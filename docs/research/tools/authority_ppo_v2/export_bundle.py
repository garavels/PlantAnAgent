"""Build an exclusive, content-verified portable research package after results close.

Standard-library only. Training artifacts and retained results are inputs, never
rewritten. The output includes selected trajectories; every remaining campaign
file is inventoried by relative path and hash as separately retained evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "plant-authority-portable-bundle-v2"
ARMS = ("narrow", "broad")
SEEDS = (101, 211, 307)
PAIRS = [(arm, seed) for arm in ARMS for seed in SEEDS]
COVERAGE = "daily-coverage-offgrid-v1"
MPC = "daily-return-mission-mpc-v1"
HELPERS = ("__init__.py", "portable.py", "evaluate.py", "analyze.py", "build_inspector.py",
           "diagnose.py", "export_bundle.py", "build_writeup.py", "campaign-evaluate.py",
           "continue_campaign.py", "orchestrate.py")
FORBIDDEN_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".ssh"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def training_plan_digest(plan: dict) -> str:
    # Training custody canonical_bytes uses sorted, pretty JSON plus a newline,
    # unlike the compact JSON digest used for the evaluation freeze.
    encoded = (json.dumps(plan, allow_nan=False, ensure_ascii=True, indent=2,
                          separators=(",", ": "), sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def relative_path(value: str | Path) -> Path:
    text = str(value)
    path = PurePosixPath(text)
    if not text or "\\" in text or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError(f"Unsafe package-relative path: {text}")
    if any(part in FORBIDDEN_PARTS or part.lower().startswith(".env") for part in path.parts):
        raise ValueError(f"Excluded package path: {text}")
    if path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"} or any(part.lower() in {"credentials", "credentials.json", "secrets.json", "id_rsa", "id_ed25519"} for part in path.parts):
        raise ValueError(f"Credential-like file excluded: {text}")
    return Path(*path.parts)


def regular_file(path: Path, *, within: Path | None = None) -> Path:
    path = path.absolute()
    if not stat.S_ISREG(path.lstat().st_mode) or path.is_symlink():
        raise ValueError(f"Input must be a regular, non-symlink file: {path}")
    if within is not None:
        base = within.resolve()
        if not path.resolve().is_relative_to(base):
            raise ValueError(f"Input escaped its admitted root: {path}")
        cursor = path
        while cursor != base:
            if cursor.is_symlink():
                raise ValueError(f"Symlink component is not admitted: {cursor}")
            cursor = cursor.parent
    return path


def pin(path: Path, *, relative_to: Path | None = None) -> dict:
    path = regular_file(path, within=relative_to)
    return {"path": path.relative_to(relative_to).as_posix() if relative_to else str(path),
            "bytes": path.stat().st_size, "sha256": sha(path)}


def verify_pin(path: Path, record: dict) -> None:
    regular_file(path)
    if sha(path) != record["sha256"] or ("bytes" in record and path.stat().st_size != record["bytes"]):
        raise ValueError(f"Input content differs from published identity: {path}")


def publish_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def copy_file(source: Path, bundle: Path, relative: str | Path, *, expected: dict | None = None) -> Path:
    target = bundle / relative_path(relative)
    source = regular_file(source)
    if expected is not None:
        verify_pin(source, expected)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            outgoing.write(block)
    if sha(target) != sha(source):
        raise ValueError(f"Copy verification failed: {target}")
    return target


def validate_analysis(analysis: dict) -> None:
    if analysis.get("schema") != "authority-ppo-v2-evaluation-analysis":
        raise ValueError("Expected the final authority analysis artifact")
    seen = set()
    for row in analysis["rows"]:
        key = (row["split"], row["controller"], row["seed"])
        if key in seen:
            raise ValueError(f"Duplicate analysis outcome: {key}")
        seen.add(key)
    labels = [COVERAGE, *(f"authority-ppo-v2-{arm}-{seed}" for arm, seed in PAIRS)]
    for split, scenarios in (("development", range(9950000, 9950008)), ("held_out", range(9960000, 9960032))):
        for label in labels:
            actual = {scenario for s, c, scenario in seen if s == split and c == label}
            if actual != set(scenarios):
                raise ValueError(f"Final analysis panel is incomplete: {split}/{label}")
    if {scenario for s, c, scenario in seen if s == "held_out" and c == MPC} != set(range(9960000, 9960008)):
        raise ValueError("Strict MPC supplementary panel is incomplete")
    if not analysis.get("evaluation_freeze_sha256"):
        raise ValueError("Final analysis lacks a held-out source/endpoint freeze")


def requirements_from_runtime(runtime: dict[str, str]) -> str:
    if not runtime or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+!-]*", version)
                          for key, version in runtime.items()):
        raise ValueError("Runtime package names/versions are malformed")
    return "# Exact direct runtime versions recorded for this campaign; uv.lock is historical.\n" + "".join(f"{key}=={runtime[key]}\n" for key in sorted(runtime))


def portable_campaign_markdown(text: str) -> str:
    """Rebase known campaign document links, including paths inside backticks."""
    for relative in ("AUTHORITY_PPO_V2_PROTOCOL.md", "AUTHORITY_PPO_V2_LEARNING_CURVES.md",
                     "benchmarks/authority_ppo_v2.json"):
        text = text.replace(f"../../{relative}", f"source/docs/research/{relative}")
    return text


def discover_freeze(campaign: Path, expected_sha: str) -> Path:
    matches = []
    for path in campaign.rglob("*freeze*.json"):
        if path.name == "endpoints-frozen.json":
            continue
        value = load_json(path)
        if value.get("schema") == "authority-ppo-v2-evaluation-freeze" and value.get("sha256") == expected_sha:
            body = {key: item for key, item in value.items() if key != "sha256"}
            if hashlib.sha256(canonical(body)).hexdigest() != expected_sha:
                raise ValueError("Evaluation freeze checksum differs")
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"Expected one matching evaluation freeze, found {len(matches)}")
    return matches[0]


def build(*, campaign: Path, source: Path, analysis_dir: Path, paper: Path, inspector: Path,
          output: Path, helpers: Path | None = None, make_zip: bool = True) -> dict:
    campaign, source, analysis_dir, paper, inspector, output = [path.resolve() for path in (campaign, source, analysis_dir, paper, inspector, output)]
    helpers = (helpers or Path(__file__).parent).resolve()
    archive = output.with_suffix(".zip")
    if output.exists() or (make_zip and (archive.exists() or archive.with_suffix(".zip.sha256").exists())):
        raise FileExistsError("The package or its archive already exists; choose a fresh destination")
    if output == campaign or output == source or source.is_relative_to(output) or output.is_relative_to(analysis_dir):
        raise ValueError("Package destination must not contain its source inputs")
    analysis = load_json(analysis_dir / "analysis.json")
    validate_analysis(analysis)
    regular_file(analysis_dir / "results.csv")
    if paper.suffix.lower() != ".pdf" or not regular_file(paper).read_bytes().startswith(b"%PDF-"):
        raise ValueError("Commission write-up must be a prepared PDF")
    writeup_build_path = regular_file(paper.with_suffix(".build.json"))
    writeup_build = load_json(writeup_build_path)
    if writeup_build.get("schema") != "authority-ppo-v2-writeup" or writeup_build.get("pages") != 2:
        raise ValueError("Commission write-up must retain the two-page writer's build provenance")
    verify_pin(paper, writeup_build["pdf"])
    verify_pin(analysis_dir / "analysis.json", writeup_build["analysis"])
    writeup_markdown = regular_file(campaign / "WRITEUP_DRAFT.md", within=campaign)
    verify_pin(writeup_markdown, writeup_build["markdown"])
    figure_records = writeup_build.get("figures")
    if not isinstance(figure_records, list) or not figure_records:
        raise ValueError("Commission write-up must retain its figure build pins")
    writeup_figures = []
    figure_names = set()
    for record in figure_records:
        path = regular_file(Path(record["path"]))
        name = relative_path(path.name)
        if name in figure_names:
            raise ValueError(f"Write-up figures would collide in the package: {name}")
        figure_names.add(name)
        verify_pin(path, record)
        writeup_figures.append((path, record))
    execution_notes = campaign / "EXECUTION_NOTES.md"
    if execution_notes.exists():
        regular_file(execution_notes, within=campaign)
    if inspector.suffix.lower() != ".html":
        raise ValueError("Inspector must be the self-contained prepared HTML artifact")
    regular_file(inspector)
    inspector_meta = inspector.with_suffix(".json")
    regular_file(inspector_meta)
    inspector_data = load_json(inspector_meta)
    if inspector_data.get("default") != "narrow-101":
        raise ValueError("Inspector default differs from the prospectively fixed case")
    runtime = load_json(campaign / "runtime-packages.json")
    requirement_text = requirements_from_runtime(runtime)
    supplied_requirements = campaign / "requirements-training.txt"
    if supplied_requirements.exists():
        # Keep the recorded file verbatim while proving its exact package pins.
        lines = [line.strip() for line in supplied_requirements.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
        supplied = dict(line.split("==", 1) for line in lines)
        if supplied != runtime:
            raise ValueError("Recorded requirements disagree with runtime-packages.json")
        requirement_text = supplied_requirements.read_text()
    plan_path = campaign / "training/plan.json"
    plan = load_json(plan_path)
    for name, version in plan["runtime"]["packages"].items():
        if runtime.get(name) != version:
            raise ValueError(f"Training and published runtime versions differ: {name}")
    endpoint_freeze_path = campaign / "training/endpoints-frozen.json"
    endpoint_freeze = load_json(endpoint_freeze_path)
    if endpoint_freeze["plan_sha256"] != training_plan_digest(plan):
        raise ValueError("Endpoint freeze does not bind the recorded training plan")
    if [(row["arm"], row["optimizer_seed"]) for row in endpoint_freeze["endpoints"]] != PAIRS:
        raise ValueError("Endpoint freeze does not contain exactly the six fixed policies")
    freeze_path = discover_freeze(campaign, analysis["evaluation_freeze_sha256"])
    evaluation_freeze = load_json(freeze_path)
    if evaluation_freeze["endpoints"] != endpoint_freeze:
        raise ValueError("Evaluation and training endpoint freezes differ")
    source_inputs: dict[str, tuple[Path, dict | None]] = {}
    for record in plan["source_files"]:
        path = regular_file(Path(record["path"]), within=source)
        verify_pin(path, record)
        source_inputs[path.relative_to(source).as_posix()] = (path, record)
    # Supplemental helpers are pinned by this publication. They cannot replace a
    # training dependency with later code from the mutable workspace.
    for name in HELPERS:
        path = helpers / name
        if path.exists():
            key = f"docs/research/tools/authority_ppo_v2/{name}"
            if key in source_inputs and sha(path) != source_inputs[key][1]["sha256"]:
                raise ValueError(f"Supplement would replace a frozen training source: {key}")
            source_inputs.setdefault(key, (regular_file(path, within=helpers), None))
    supplemental_protocol = helpers.parent.parent / "AUTHORITY_PPO_V2_LEARNING_CURVES.md"
    if supplemental_protocol.exists():
        source_inputs["docs/research/AUTHORITY_PPO_V2_LEARNING_CURVES.md"] = (regular_file(supplemental_protocol), None)
    source_inputs["docs/research/tools/authority_ppo_v2/REPRODUCE.md"] = (regular_file(helpers / "REPRODUCE.md", within=helpers), None)
    for required in ("portable.py", "evaluate.py", "analyze.py", "export_bundle.py"):
        if f"docs/research/tools/authority_ppo_v2/{required}" not in source_inputs:
            raise ValueError(f"Missing publication runtime helper: {required}")
    # Bind the evaluator and planner actually used, rather than a later editable
    # workspace version. Unchanged dependencies are already in the training set.
    for record in evaluation_freeze["sources"]:
        path = regular_file(Path(record["path"]), within=source)
        verify_pin(path, record)
        key = path.relative_to(source).as_posix()
        existing = source_inputs.get(key)
        if existing is not None and sha(existing[0]) != record["sha256"]:
            raise ValueError(f"Publication source differs from the evaluation freeze: {key}")
        source_inputs[key] = (path, record)
    for relative in ("pyproject.toml", "backend/pyproject.toml", "simulator/pyproject.toml", "docs/research/COVERAGE_PPO_V1_PROTOCOL.md"):
        path = source / relative
        if path.exists():
            source_inputs.setdefault(relative, (regular_file(path, within=source), None))
    endpoint_inputs = []
    for frozen in endpoint_freeze["endpoints"]:
        arm, seed = frozen["arm"], frozen["optimizer_seed"]
        directory = campaign / "training" / f"{arm}-seed-{seed}"
        result_path = directory / "result.json"
        artifact = directory / "final-policy.zip"
        verify_pin(result_path, frozen["result"])
        verify_pin(artifact, frozen["checkpoint"])
        result = load_json(result_path)
        if (result["arm"] != arm or result["optimizer_seed"] != seed or result["num_timesteps"] != 262144
            or not result["checkpoint_reopen_verified"] or not result["actor_mean_parameters_changed"]
            or result["initial_actor_mean_sha256"] == result["final_actor_mean_sha256"]
            or result["final_checkpoint"]["sha256"] != sha(artifact)):
            raise ValueError(f"Final endpoint is incomplete or mismatched: {arm}/{seed}")
        endpoint_inputs.append((arm, seed, directory, result))
    for seed in SEEDS:
        paired = [result for arm, opt, directory, result in endpoint_inputs if opt == seed]
        if paired[0]["initial_tensor_sha256"] != paired[1]["initial_tensor_sha256"]:
            raise ValueError(f"Paired authority arms did not start with identical tensors: {seed}")
    selected_rows = [row for row in analysis["rows"] if row["split"] == "held_out" and
                     ((row["seed"] == 9960000 and row["controller"] in [COVERAGE, *(f"authority-ppo-v2-{arm}-{seed}" for arm, seed in PAIRS)])
                      or (row["seed"] == 9960001 and row["controller"] == "authority-ppo-v2-broad-101"))]
    if len(selected_rows) != 8 or any(not row.get("recording") for row in selected_rows):
        raise ValueError("Prespecified first-case trajectories or broad replay case are missing")
    expected_inspector_keys = {"coverage", *(f"{arm}-{seed}" for arm, seed in PAIRS)}
    if set(inspector_data.get("recordings", {})) != expected_inspector_keys:
        raise ValueError("Inspector does not contain coverage and all six final endpoints")
    for row in selected_rows:
        verify_pin(Path(row["recording"]["path"]), row["recording"])
        if row["seed"] == 9960000:
            key = "coverage" if row["controller"] == COVERAGE else row["controller"].removeprefix("authority-ppo-v2-")
            displayed = inspector_data["recordings"][key]
            if (displayed["recording_sha256"] != row["recording"]["sha256"] or displayed["scenario"] != 9960000
                or displayed["contract_sha256"] != analysis["benchmark_contract_sha256"]):
                raise ValueError(f"Inspector displayed case differs from the packaged evidence: {key}")
    proof_path = campaign / "diagnosis/selected-full-weeks.json"
    proofs = load_json(proof_path)
    if {row["seed"] for row in proofs["rows"]} != {9930001, 9930011}:
        raise ValueError("Constructive proof must retain both previously exposed cases")
    for row in proofs["rows"]:
        verify_pin(Path(row["recording"]["path"]), row["recording"])
    # Inventory the original campaign before creating any output underneath it.
    inventory = []
    for path in sorted(campaign.rglob("*")):
        if path.is_file() and not path.is_relative_to(output):
            relative = path.relative_to(campaign)
            try:
                relative_path(relative.as_posix())
            except ValueError:
                continue
            inventory.append(pin(path, relative_to=campaign))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    for relative, (path, expected) in sorted(source_inputs.items()):
        copy_file(path, output, Path("source") / relative, expected=expected)
    template = helpers / "REPRODUCE.md"
    copy_file(template, output, "REPRODUCE.md")
    rendered = (output / "REPRODUCE.md").read_text().replace("@PYTHON_VERSION@", plan["runtime"]["python"].split()[0])
    # The template output is new and unpublished until the manifest is written.
    (output / "REPRODUCE.md").write_text(rendered)
    (output / "requirements-training.txt").write_text(requirement_text)
    copy_file(campaign / "runtime-packages.json", output, "runtime-packages.json")
    copy_file(paper, output, "commission-writeup.pdf", expected=writeup_build["pdf"])
    copy_file(writeup_markdown, output, "history/WRITEUP_DRAFT.md", expected=writeup_build["markdown"])
    copy_file(writeup_build_path, output, "history/writeup-build.json")
    published_figures = [
        pin(copy_file(path, output, Path("report-figures") / path.name, expected=record), relative_to=output)
        for path, record in writeup_figures
    ]
    with (output / "commission-writeup.md").open("x") as stream:
        stream.write(portable_campaign_markdown((output / "history/WRITEUP_DRAFT.md").read_text()))
    if execution_notes.exists():
        copy_file(execution_notes, output, "history/EXECUTION_NOTES.md")
        with (output / "EXECUTION_NOTES.md").open("x") as stream:
            stream.write(portable_campaign_markdown((output / "history/EXECUTION_NOTES.md").read_text()))
    copy_file(inspector, output, "inspector.html")
    copy_file(inspector_meta, output, "inspector.json")
    for path in sorted(analysis_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".png", ".svg", ".pdf", ".md"}:
            copy_file(path, output, Path("results") / path.relative_to(analysis_dir))
    for relative in ("training/plan.json", "training/endpoints-frozen.json", "campaign-start.json", "seed-audit.json", "source-initial-snapshot.json"):
        if (campaign / relative).exists():
            copy_file(campaign / relative, output, Path("history") / relative)
    copy_file(freeze_path, output, "history/evaluation-freeze.json")
    for path in sorted((campaign / "preflight").glob("*")):
        if path.is_file() and path.suffix in {".json", ".log", ".txt", ".md", ".py"}:
            copy_file(path, output, Path("history/preflight") / path.name)
    endpoints = []
    for arm, seed, directory, result in endpoint_inputs:
        prefix = Path("policies") / f"{arm}-{seed}"
        artifact = copy_file(directory / "final-policy.zip", output, prefix / "policy.zip", expected=result["final_checkpoint"])
        fields = ("schema", "arm", "optimizer_seed", "num_timesteps", "policy_identity", "constraint_state",
                  "final_tensor_sha256", "final_actor_sha256", "final_actor_mean_sha256", "initial_tensor_sha256",
                  "initial_actor_mean_sha256", "actor_mean_parameters_changed", "checkpoint_reopen_verified")
        compact = {key: result[key] for key in fields}
        compact["original_result_sha256"] = sha(directory / "result.json")
        metadata = output / prefix / "metadata.json"
        publish_json(metadata, compact)
        copy_file(directory / "result.json", output, Path("history/training") / f"{arm}-seed-{seed}" / "result.json")
        for optional in ("attempt.json", "progress.jsonl"):
            if (directory / optional).exists():
                copy_file(directory / optional, output, Path("history/training") / f"{arm}-seed-{seed}" / optional)
        endpoints.append({"arm": arm, "optimizer_seed": seed, "checkpoint": artifact.relative_to(output).as_posix(),
                          "checkpoint_sha256": sha(artifact), "metadata": metadata.relative_to(output).as_posix(),
                          "metadata_sha256": sha(metadata)})
    cases = []
    for row in selected_rows:
        path = Path(row["recording"]["path"])
        target = copy_file(path, output, Path("trajectories") / row["controller"] / path.name, expected=row["recording"])
        cases.append({"controller": row["controller"], "seed": row["seed"], "split": "held_out",
                      "recording": pin(target, relative_to=output), "minute_truth_sha256": row["minute_truth_sha256"]})
    for path in sorted((campaign / "diagnosis").iterdir()):
        if path.is_file() and not path.name.endswith(".json.gz") and path.suffix in {".json", ".md", ".csv", ".png", ".pdf", ".py"}:
            copy_file(path, output, Path("diagnosis") / path.name)
    for row in proofs["rows"]:
        path = Path(row["recording"]["path"])
        copy_file(path, output, Path("diagnosis") / path.name, expected=row["recording"])
    publish_json(output / "history/full-evidence-inventory.json", {
        "schema": "plant-authority-external-evidence-inventory-v2", "path_base": "original campaign root",
        "files": inventory, "total_bytes": sum(row["bytes"] for row in inventory),
        "scope": "Hash inventory of all retained original campaign files. Files absent from this package remain separate evidence; the inventory does not claim their inclusion.",
    })
    source_files = [pin(path, relative_to=output / "source") for path in sorted((output / "source").rglob("*")) if path.is_file()]
    bundle_files = [pin(path, relative_to=output) for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = {
        "schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_files": source_files, "endpoints": endpoints, "runtime_packages": runtime,
        "recorded_python": plan["runtime"]["python"], "recorded_runtime_is_uv_lock": False,
        "benchmark_contract_sha256": analysis["benchmark_contract_sha256"],
        "evaluation_freeze_sha256": analysis["evaluation_freeze_sha256"],
        "training_plan_sha256": training_plan_digest(plan), "training_plan_file_sha256": sha(plan_path),
        "initial_weights_paired_across_arms_verified": True,
        "writeup_figures": published_figures,
        "published_cases": cases, "bundle_files": bundle_files,
        "evaluation_outcome_counts": dict(sorted(Counter(row["split"] for row in analysis["rows"]).items())),
        "external_evidence_inventory": "history/full-evidence-inventory.json",
        "historical_path_policy": "Original plan/result paths are retained as historical evidence; portable inference verifies relative source/checkpoint identities and never calls the absolute-host training-plan gate.",
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    publish_json(output / "manifest.json", manifest)
    verify_bundle(output)
    report = {"bundle": str(output), "manifest_sha256": manifest["manifest_sha256"],
              "files": len(bundle_files) + 1, "endpoints": len(endpoints), "selected_trajectories": len(cases)}
    if make_zip:
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    info = zipfile.ZipInfo((Path(output.name) / path.relative_to(output)).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    zipped.writestr(info, path.read_bytes())
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("Portable archive CRC verification failed")
        digest = sha(archive)
        with archive.with_suffix(".zip.sha256").open("x") as stream:
            stream.write(f"{digest}  {archive.name}\n")
        report.update(archive=str(archive), archive_sha256=digest, archive_bytes=archive.stat().st_size)
    return report


def verify_bundle(bundle: Path) -> dict:
    bundle = bundle.resolve()
    manifest = load_json(bundle / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unknown package schema")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if hashlib.sha256(canonical(body)).hexdigest() != manifest["manifest_sha256"]:
        raise ValueError("Package manifest checksum differs")
    if [(item["arm"], item["optimizer_seed"]) for item in manifest["endpoints"]] != PAIRS:
        raise ValueError("Package does not contain exactly six unique endpoints")
    admitted = set()
    for item in manifest["bundle_files"]:
        relative = relative_path(item["path"])
        if str(relative) in admitted:
            raise ValueError("Duplicate package inventory entry")
        admitted.add(str(relative))
        path = regular_file(bundle / relative, within=bundle)
        verify_pin(path, item)
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file() and path != bundle / "manifest.json"}
    if actual != admitted:
        raise ValueError("Package contains an unlisted or missing file")
    for item in manifest["source_files"]:
        verify_pin(regular_file(bundle / "source" / relative_path(item["path"]), within=bundle / "source"), item)
    for item in manifest["endpoints"]:
        for key in ("checkpoint", "metadata"):
            verify_pin(regular_file(bundle / relative_path(item[key]), within=bundle), {"sha256": item[key + "_sha256"]})
    return {"verified": True, "files": len(admitted) + 1, "manifest_sha256": manifest["manifest_sha256"]}


def _mock_build_fixture(root: Path) -> None:
    """Exercise the complete exporter with synthetic bytes and no model imports."""
    campaign, source, helpers = root / "campaign", root / "snapshot", root / "helpers"
    for path in (campaign, source, helpers):
        path.mkdir()
    training_source = source / "docs/research/tools/authority_ppo_v2/train.py"
    training_source.parent.mkdir(parents=True)
    training_source.write_text("# synthetic fixture only\n")
    for name in ("portable.py", "evaluate.py", "analyze.py", "export_bundle.py"):
        (helpers / name).write_text("# synthetic helper only\n")
    (helpers / "REPRODUCE.md").write_text("Synthetic package, Python @PYTHON_VERSION@.\n")
    evaluator_source = source / "docs/research/tools/authority_ppo_v2/evaluate.py"
    evaluator_source.write_bytes((helpers / "evaluate.py").read_bytes())
    runtime = {"numpy": "0.0.fixture"}
    publish_json(campaign / "runtime-packages.json", runtime)
    plan = {"runtime": {"python": "3.12.fixture", "packages": runtime}, "source_files": [pin(training_source)]}
    plan_path = campaign / "training/plan.json"
    publish_json(plan_path, plan)
    endpoints = []
    for arm, seed in PAIRS:
        directory = campaign / "training" / f"{arm}-seed-{seed}"
        directory.mkdir()
        artifact = directory / "final-policy.zip"
        artifact.write_bytes(f"Synthetic checkpoint, not loadable: {arm}/{seed}".encode())
        result = {"schema": "coverage-authority-full-mission-ppo-v2", "arm": arm, "optimizer_seed": seed,
                  "num_timesteps": 262144, "policy_identity": {}, "constraint_state": {},
                  "initial_tensor_sha256": f"paired-{seed}", "final_tensor_sha256": f"final-{arm}-{seed}",
                  "final_actor_sha256": "actor", "initial_actor_mean_sha256": "initial-mean",
                  "final_actor_mean_sha256": "final-mean", "checkpoint_reopen_verified": True,
                  "actor_mean_parameters_changed": True, "final_checkpoint": pin(artifact)}
        publish_json(directory / "result.json", result)
        endpoints.append({"arm": arm, "optimizer_seed": seed, "result": pin(directory / "result.json"), "checkpoint": pin(artifact)})
    endpoint_freeze = {"endpoints": endpoints, "plan_sha256": training_plan_digest(plan)}
    publish_json(campaign / "training/endpoints-frozen.json", endpoint_freeze)
    evaluation_freeze = {"schema": "authority-ppo-v2-evaluation-freeze", "endpoints": endpoint_freeze,
                         "sources": [pin(training_source), pin(evaluator_source)]}
    evaluation_freeze["sha256"] = hashlib.sha256(canonical(evaluation_freeze)).hexdigest()
    publish_json(campaign / "evaluation-freeze.json", evaluation_freeze)
    labels = [COVERAGE, *(f"authority-ppo-v2-{arm}-{seed}" for arm, seed in PAIRS)]
    rows = []
    for split, scenarios in (("development", range(9950000, 9950008)), ("held_out", range(9960000, 9960032))):
        for label in labels:
            for scenario in scenarios:
                row = {"split": split, "controller": label, "seed": scenario, "recording": None}
                if split == "held_out" and (scenario == 9960000 or (scenario == 9960001 and label == "authority-ppo-v2-broad-101")):
                    path = campaign / "evaluations" / label / f"{scenario}.json.gz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic recording fixture; no physical scenario opened")
                    row.update(recording=pin(path), minute_truth_sha256="synthetic")
                rows.append(row)
    rows.extend({"split": "held_out", "controller": MPC, "seed": scenario, "recording": None} for scenario in range(9960000, 9960008))
    analysis_dir = campaign / "analysis"
    publish_json(analysis_dir / "analysis.json", {"schema": "authority-ppo-v2-evaluation-analysis", "rows": rows,
                 "evaluation_freeze_sha256": evaluation_freeze["sha256"], "benchmark_contract_sha256": "synthetic"})
    (analysis_dir / "results.csv").write_text("synthetic,fixture\n")
    paper, inspector = campaign / "paper.pdf", campaign / "inspector.html"
    paper.write_bytes(b"%PDF-1.4 synthetic fixture, no publication\n")
    writeup = campaign / "WRITEUP_DRAFT.md"
    writeup.write_text("[Protocol](../../AUTHORITY_PPO_V2_PROTOCOL.md).\n"
                       "[Contract](../../benchmarks/authority_ppo_v2.json).\n"
                       "[Diagnosis](diagnosis/README.md). [External](https://example.org/paper).\n")
    notes = campaign / "EXECUTION_NOTES.md"
    notes.write_text("Protocol: `../../AUTHORITY_PPO_V2_PROTOCOL.md`; "
                     "addition: `../../AUTHORITY_PPO_V2_LEARNING_CURVES.md`.\n"
                     "Evidence: `diagnosis/README.md`.\n")
    figures = []
    for name in ("writeup-paired-wear.png", "writeup-diagnostic.png"):
        path = campaign / "writeup-figures" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic binary figure fixture: " + name.encode())
        figures.append(path)
    writeup_build = {"schema": "authority-ppo-v2-writeup", "pages": 2, "pdf": pin(paper),
                     "analysis": pin(analysis_dir / "analysis.json"), "markdown": pin(writeup),
                     "figures": [pin(path) for path in figures]}
    publish_json(paper.with_suffix(".build.json"), writeup_build)
    inspector.write_text("<html>Synthetic fixture</html>")
    displayed = {("coverage" if row["controller"] == COVERAGE else row["controller"].removeprefix("authority-ppo-v2-")):
                 {"recording_sha256": row["recording"]["sha256"], "scenario": 9960000, "contract_sha256": "synthetic"}
                 for row in rows if row["split"] == "held_out" and row["seed"] == 9960000 and row["recording"]}
    publish_json(inspector.with_suffix(".json"), {"default": "narrow-101", "recordings": displayed})
    proof_rows = []
    for scenario in (9930001, 9930011):
        path = campaign / "diagnosis" / f"{scenario}-week.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic proof recording fixture")
        proof_rows.append({"seed": scenario, "recording": pin(path)})
    publish_json(campaign / "diagnosis/selected-full-weeks.json", {"rows": proof_rows})
    output = campaign / "deliverables/research-package"
    for field in ("pdf", "analysis", "markdown"):
        mismatched = {**writeup_build, field: {**writeup_build[field], "sha256": "0" * 64}}
        paper.with_suffix(".build.json").write_text(json.dumps(mismatched))
        try:
            build(campaign=campaign, source=source, analysis_dir=analysis_dir,
                  paper=paper, inspector=inspector, output=output, helpers=helpers)
        except ValueError as error:
            assert "Input content differs" in str(error)
        else:
            raise AssertionError(f"Mismatched write-up {field} identity accepted")
        assert not output.exists()
    for records, message in (
        ([{**writeup_build["figures"][0], "sha256": "0" * 64}, writeup_build["figures"][1]], "Input content differs"),
        ([writeup_build["figures"][0], writeup_build["figures"][0]], "figures would collide"),
    ):
        paper.with_suffix(".build.json").write_text(json.dumps({**writeup_build, "figures": records}))
        try:
            build(campaign=campaign, source=source, analysis_dir=analysis_dir,
                  paper=paper, inspector=inspector, output=output, helpers=helpers)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("Mismatched or colliding write-up figure accepted")
        assert not output.exists()
    paper.with_suffix(".build.json").write_text(json.dumps(writeup_build))
    report = build(campaign=campaign, source=source, analysis_dir=analysis_dir,
                   paper=paper, inspector=inspector, output=output, helpers=helpers)
    assert report["endpoints"] == 6 and report["selected_trajectories"] == 8
    assert Path(report["archive"]).exists()
    moved = root / "moved-complete-package"
    output.rename(moved)
    assert verify_bundle(moved)["verified"]
    assert "3.12.fixture" in (moved / "REPRODUCE.md").read_text()
    assert (moved / "history/WRITEUP_DRAFT.md").read_bytes() == writeup.read_bytes()
    assert (moved / "history/EXECUTION_NOTES.md").read_bytes() == notes.read_bytes()
    assert (moved / "history/writeup-build.json").read_bytes() == paper.with_suffix(".build.json").read_bytes()
    published_figures = load_json(moved / "manifest.json")["writeup_figures"]
    assert {item["path"] for item in published_figures} == {f"report-figures/{path.name}" for path in figures}
    for path in figures:
        assert (moved / "report-figures" / path.name).read_bytes() == path.read_bytes()
    tampered_figure = moved / "report-figures" / figures[0].name
    original_figure_bytes = tampered_figure.read_bytes()
    tampered_figure.write_bytes(original_figure_bytes + b"altered")
    try:
        verify_bundle(moved)
    except ValueError as error:
        assert "Input content differs" in str(error)
    else:
        raise AssertionError("Packaged figure alteration went undetected")
    tampered_figure.write_bytes(original_figure_bytes)
    assert verify_bundle(moved)["verified"]
    portable_writeup = (moved / "commission-writeup.md").read_text()
    assert "(source/docs/research/AUTHORITY_PPO_V2_PROTOCOL.md)" in portable_writeup
    assert "(source/docs/research/benchmarks/authority_ppo_v2.json)" in portable_writeup
    assert "(diagnosis/README.md)" in portable_writeup and "(https://example.org/paper)" in portable_writeup
    portable_notes = (moved / "EXECUTION_NOTES.md").read_text()
    assert "`source/docs/research/AUTHORITY_PPO_V2_PROTOCOL.md`" in portable_notes
    assert "`source/docs/research/AUTHORITY_PPO_V2_LEARNING_CURVES.md`" in portable_notes
    assert "`diagnosis/README.md`" in portable_notes


def self_test() -> dict:
    """Synthetic confinement/copy/manifest tests only; no real policy is loaded."""
    with tempfile.TemporaryDirectory(prefix="authority-package-fixture-") as temporary:
        root = Path(temporary).resolve()
        source = root / "input.txt"
        source.write_text("synthetic fixture\n")
        bundle = root / "movable"
        bundle.mkdir()
        copy_file(source, bundle, "source/fixture.py")
        try:
            copy_file(source, bundle, "source/fixture.py")
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing artifact was overwritten")
        for bad in ("../escape", "/absolute", ".env", "source/.env.secret", "source/.git/config", "source/key.pem", "a\\b"):
            try:
                relative_path(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Unsafe path accepted: {bad}")
        link = root / "link.txt"
        link.symlink_to(source)
        try:
            copy_file(link, bundle, "link.txt")
        except ValueError:
            pass
        else:
            raise AssertionError("Symlink copied")
        endpoints = []
        for arm, seed in PAIRS:
            prefix = f"policies/{arm}-{seed}"
            checkpoint = copy_file(source, bundle, prefix + "/policy.zip")
            metadata = copy_file(source, bundle, prefix + "/metadata.json")
            endpoints.append({"arm": arm, "optimizer_seed": seed, "checkpoint": checkpoint.relative_to(bundle).as_posix(),
                              "checkpoint_sha256": sha(checkpoint), "metadata": metadata.relative_to(bundle).as_posix(), "metadata_sha256": sha(metadata)})
        manifest = {"schema": SCHEMA, "endpoints": endpoints, "source_files": [pin(bundle / "source/fixture.py", relative_to=bundle / "source")],
                    "bundle_files": [pin(path, relative_to=bundle) for path in sorted(bundle.rglob("*")) if path.is_file()]}
        manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
        publish_json(bundle / "manifest.json", manifest)
        assert verify_bundle(bundle)["verified"]
        moved = root / "different-location"
        bundle.rename(moved)
        assert verify_bundle(moved)["verified"]
        (moved / "source/fixture.py").write_text("tampered")
        try:
            verify_bundle(moved)
        except ValueError:
            pass
        else:
            raise AssertionError("Content alteration went undetected")
        _mock_build_fixture(root)
    return {"synthetic_self_test": "passed", "checks": ["exclusive copy", "path confinement", "credential exclusions", "symlink rejection", "six-endpoint manifest", "movable relative inventory", "content alteration rejection", "write-up PDF/analysis/markdown provenance rejection", "write-up figure pin and collision rejection", "exact moved figure copies and alteration rejection", "portable editable document links and exact history", "complete synthetic export and ZIP", "moved complete package verification"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--campaign", type=Path, default=ROOT / "docs/research/results/authority_ppo_20260917")
    parser.add_argument("--source", type=Path, default=ROOT / "runs/authority-source-20260917")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--paper", type=Path)
    parser.add_argument("--inspector", type=Path)
    parser.add_argument("--helpers", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.verify:
        result = verify_bundle(args.verify)
    else:
        if args.analysis_dir is None or args.paper is None or args.inspector is None:
            parser.error("Final export requires --analysis-dir, --paper and --inspector")
        result = build(campaign=args.campaign, source=args.source, analysis_dir=args.analysis_dir,
                       paper=args.paper, inspector=args.inspector, helpers=args.helpers,
                       output=args.output or args.campaign / "deliverables/research-package", make_zip=not args.no_zip)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
