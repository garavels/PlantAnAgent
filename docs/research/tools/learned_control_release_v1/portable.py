"""Authenticate and replay immutable learned-control evidence after relocation.

Original JSON, checkpoint sidecars, and gate bytes are never rewritten. A
verified original-root-to-package-root mapping resolves their absolute paths.
Replay must run in a fresh interpreter importing this package's own sources.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from docs.research.tools.wholeplant_ppo_v1.portable import canonical, digest, file_pin, file_sha

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "learned-control-portable-v1"
MANIFEST = "learned-control-manifest.json"


def same(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError("Verification differs: " + label)


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def member_path(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Package member must be a safe relative path")
    path = root / relative
    if any((root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts)+1)):
        raise ValueError("Package member uses a symbolic link")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Package member escapes its root")
    return path


def iter_pins(value):
    if isinstance(value, dict):
        if ({"path", "bytes", "sha256"}.issubset(value) and isinstance(value["path"], str)
                and Path(value["path"]).is_absolute()):
            yield {k: value[k] for k in ("path", "bytes", "sha256")}
        for item in value.values():
            yield from iter_pins(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_pins(item)


class Evidence:
    def __init__(self, results, *, root=ROOT):
        self.root = Path(root).resolve()
        self.results = Path(results).resolve()
        if not self.results.is_relative_to(self.root):
            raise ValueError("Results must belong to the package root")
        self.freeze_path = self.results / "freeze.json"
        self.spec = json.loads(self.freeze_path.read_text())
        suffix = "/docs/research/tools/learned_control_v1/environment.py"
        originals = [p["path"][:-len(suffix)] for p in self.spec["sources"] if p["path"].endswith(suffix)]
        if len(originals) != 1:
            raise ValueError("No unique original source root")
        self.original_root = Path(originals[0])
        self.verify_sources()

    def path_for_original(self, path):
        try:
            relative = Path(path).relative_to(self.original_root)
        except ValueError as exc:
            raise ValueError("Original evidence path is outside its workspace") from exc
        return member_path(self.root, relative)

    def resolve(self, pin):
        path = self.path_for_original(pin["path"])
        if not path.is_file() or path.stat().st_size != pin["bytes"] or file_sha(path) != pin["sha256"]:
            raise ValueError("Pinned evidence changed: " + pin["path"])
        return path

    def selection(self):
        if self.spec["split"] != "final":
            return None
        gate_path = self.resolve(self.spec["gate"])
        gate = json.loads(gate_path.read_text())
        if (gate.get("schema") != "learned-control-v1-evaluation-gate"
                or gate.get("selected_recipe_frozen") is not True or "final" not in gate.get("allowed_splits", [])):
            raise ValueError("Final gate does not freeze a selected recipe")
        selected_pin = self.spec["selected_recipe"]
        selected_path = self.resolve(selected_pin)
        path = Path(gate["selected_recipe"]["path"])
        if not path.is_absolute():
            path = Path(self.spec["gate"]["path"]).parent / path
        same(str(path), selected_pin["path"], "gate selected-recipe path")
        same(gate["selected_recipe"]["sha256"], selected_pin["sha256"], "gate selected-recipe hash")
        selected = json.loads(selected_path.read_text())
        from docs.research.tools.learned_control_v1.evaluate import CONTROLLER_IDENTITY_FIELDS
        identity = lambda c: {k: c[k] for k in CONTROLLER_IDENTITY_FIELDS if k in c}
        actual = {k: c for k, c in self.spec["controllers"].items() if c["kind"] == "learned"}
        chosen = {c["label"]: c for c in selected["controllers"]}
        same(set_as_list(actual), set_as_list(chosen), "all selected learned controllers")
        for label, chosen_policy in chosen.items():
            same(identity(actual[label]), identity(chosen_policy), "selected controller specification")
            if chosen_policy.get("checkpoint_sha256"):
                same(chosen_policy["checkpoint_sha256"], actual[label]["checkpoint_files"]["checkpoint"]["sha256"], "selected checkpoint hash")
        champion = selected["baseline_champion"]
        same(identity(self.spec["controllers"][champion["label"]]), identity(champion), "selected baseline champion")
        for field in ("seeds", "conditions", "primary"):
            same(selected[field], self.spec[field], "selected " + field)
        same(self.spec["primary"]["reference"], champion["label"], "primary baseline champion")
        same(self.spec["primary"].get("recipe", self.spec["primary"].get("group")), selected["recipe_id"], "primary selected recipe")
        return gate, selected

    def verify_sources(self):
        from docs.research.tools.learned_control_v1 import evaluate
        if self.spec.get("schema") != evaluate.SCHEMA:
            raise ValueError("Unknown evaluation schema")
        same(self.spec["sha256"], digest({k: v for k, v in self.spec.items() if k != "sha256"}), "freeze digest")
        evaluate._validate_grid(self.spec["split"], self.spec["seeds"], self.spec["conditions"])
        expected = [{"seed": s, "condition": c, "controller": p} for s in self.spec["seeds"]
                    for c in self.spec["conditions"] for p in self.spec["controllers"]]
        same(self.spec["cases"], expected, "complete frozen case grid")
        same(self.spec["expected_cases"], len(expected), "expected case count")
        for pin in [self.spec["original_evaluation"], *self.spec["sources"]]:
            self.resolve(pin)
        for controller in self.spec["controllers"].values():
            for pin in controller.get("checkpoint_files", {}).values():
                self.resolve(pin)
            if controller["kind"] != "baseline":
                same(controller["checkpoint"], controller["checkpoint_files"]["checkpoint"]["path"], "controller checkpoint path")
        self.selection()

    def read(self, seed, condition, controller):
        if {"seed": seed, "condition": condition, "controller": controller} not in self.spec["cases"]:
            raise ValueError("Case is outside the frozen panel")
        relative = f"cases/{condition}--{controller}--{seed}/result.json"
        path = member_path(self.results, relative)
        row = json.loads(path.read_text())
        same([row[k] for k in ("seed", "condition", "controller")], [seed, condition, controller], "case identity")
        same(row["freeze_sha256"], self.spec["sha256"], "case freeze")
        with gzip.open(self.resolve(row["recording"]), "rt") as stream:
            recording = json.load(stream)
        same(recording["summary"], {k: v for k, v in row.items() if k != "recording"}, "recording/result summary")
        return row, recording

    @contextmanager
    def relocated_gate(self):
        """Resolve only authenticated gate dependencies; physical methods are untouched."""
        if self.root != ROOT:
            raise ValueError("Replay requires a fresh interpreter importing the relocated root")
        from docs.research.tools.learned_control_v1 import environment
        original = environment._verify_final_gate
        if self.spec["split"] != "final":
            yield
            return
        gate, _ = self.selection()
        same(gate["environment_identity_sha256"], environment.environment_identity_sha256(), "replay environment identity")
        same(gate["source_identity_sha256"], environment.source_identity_sha256(), "replay source identity")
        admitted = self.resolve(self.spec["gate"])

        def verify(path):
            same(str(Path(path).resolve()), str(admitted), "replay gate path")
            self.selection()
            return self.spec["gate"]["sha256"]

        environment._verify_final_gate = verify
        try:
            yield
        finally:
            environment._verify_final_gate = original


def set_as_list(value):
    return sorted(value)


def load_policy(evidence, controller, env):
    from docs.research.tools.learned_control_v1 import evaluate
    cs = deepcopy(evidence.spec["controllers"][controller])
    if cs["kind"] != "baseline":
        cs["checkpoint"] = str(evidence.resolve(cs["checkpoint_files"]["checkpoint"]))
    policy = evaluate.load_controller(cs, env)
    if cs["kind"] != "baseline" and cs.get("optimizer_seed") is not None:
        same(int(policy.seed), cs["optimizer_seed"], "loaded optimizer seed")
    return policy


def inference(policy, kind, observation):
    observation = np.asarray(observation, dtype=np.float32)
    evidence = {"observation": observation.tolist(), "observation_sha256": digest(observation.tolist())}
    if kind == "baseline":
        action, details = policy.inference(observation.copy())
        evidence["baseline_inference"] = details
    else:
        details = policy.deterministic_details(observation.copy())
        action = details["clipped_action"]
        evidence.update(normalized_observation=np.asarray(details["normalized_input"]).tolist(),
                        raw_mean=np.asarray(details["unclipped_mean"]).tolist())
    evidence["action"] = np.asarray(action).tolist()
    return evidence


def verify_recording(row, recording, *, first_seed, policy=None, controller_kind=None):
    from docs.research.tools.learned_control_v1 import analyze
    if not analyze.audited(row):
        raise ValueError("Case did not complete every physical/economic audit")
    if row["decisions"] != 672 or row["physical_states"] != 10081 or row["minute_audit"].get("states_checked") != 10081:
        raise ValueError("Case geometry differs")
    commands, states = recording["decision_trace"], recording["minute_truth_states"]
    if len(commands) != 672 or len(states) != (10081 if row["seed"] == first_seed else 0):
        raise ValueError("Prescribed inference or physical retention differs")
    witness, reward_sum = hashlib.sha256(), 0.0
    for index, command in enumerate(commands):
        same(command["decision_index"], index, "decision order")
        details = command["evidence"]
        observation, action = np.asarray(details["observation"]), np.asarray(details["action"])
        if observation.shape != (149,) or action.shape != (7,) or not np.isfinite(observation).all() or not np.isfinite(action).all():
            raise ValueError("Invalid stored observation or action")
        same(details["observation_sha256"], digest(details["observation"]), "stored observation digest")
        same(details["action"], command["action"], "stored action copies")
        if "normalized_observation" in details:
            normalized, mean = np.asarray(details["normalized_observation"]), np.asarray(details["raw_mean"])
            if normalized.shape != (149,) or mean.shape != (7,) or not np.isfinite(normalized).all() or not np.isfinite(mean).all():
                raise ValueError("Invalid learned inference witness")
            same(np.clip(mean, -1, 1).tolist(), action.tolist(), "deterministic actor clipping")
        else:
            same(details["baseline_inference"]["actual_observation"], details["observation"], "baseline actual input")
        if policy is not None:
            same(inference(policy, controller_kind, observation), details, "recomputed full inference witness")
        witness.update(canonical(details))
        reward_sum += command["reward_eur"]
    same(witness.hexdigest(), row["input_action_witness_sha256"], "complete input/action witness")
    if abs(reward_sum-row["economic_metrics"]["net_operating_value_eur"]) > 1e-6:
        raise ValueError("Stored rewards do not reconcile with economic total")
    settlements = row["delivery"]["settlements"]
    if len(settlements) != 28 or any(d["due_step"] != (i+1)*360 or d["requested_kg"] != 180 for i, d in enumerate(settlements)):
        raise ValueError("Delivery schedule differs")
    same(row["every_delivery_passed"], all(d["unmet_kg"] <= 1e-9 for d in settlements), "delivery success classification")
    if abs(sum(d["unmet_kg"] for d in settlements)-row["economic_metrics"]["unmet_kg"]) > 1e-6:
        raise ValueError("Unmet delivery mass does not reconcile")
    if states:
        physical_hash = hashlib.sha256()
        for index, state in enumerate(states):
            same(state["step"], index, "physical step order")
            raw = canonical(state)
            physical_hash.update(len(raw).to_bytes(8, "big"))
            physical_hash.update(raw)
        same(physical_hash.hexdigest(), row["minute_audit"]["stream_sha256"], "physical stream digest")
        same(digest(states), row["retained_minute_truth_sha256"], "retained state digest")
        same(states[0], row["initial_state"], "initial retained state")
        same(states[-1], row["final_state"], "final retained state")
    return {"decisions": len(commands), "retained_states": len(states), "inference_recomputed": policy is not None}


def verify_all(evidence, output, *, recompute_inference=True):
    from docs.research.tools.learned_control_v1 import analyze, environment
    from docs.research.tools.wholeplant_ppo_v1 import training
    from docs.research.tools.tail_stress_v1.environment import condition_spec
    training.configure_torch()
    summary_path = evidence.results / "summary.json"
    summary = json.loads(summary_path.read_text())
    same(summary["freeze"], {**file_pin(evidence.freeze_path), "path": str(evidence.original_root / evidence.freeze_path.relative_to(evidence.root))}, "summary freeze pin")
    index = analyze.validate_rows(summary["rows"], evidence.spec)
    if not summary["all_executions_and_audits_passed"] or summary["case_count"] != len(index):
        raise ValueError("Summary is incomplete or unaudited")
    counts = {"cases": 0, "decisions": 0, "retained_states": 0, "exact_control_prefixes": 0, "exact_physical_prefixes": 0}
    for controller, cs in evidence.spec["controllers"].items():
        policy = env = None
        try:
            if recompute_inference:
                env = environment.FixedEnv(condition="ordinary", split="train")
                policy = load_policy(evidence, controller, env)
            for seed in evidence.spec["seeds"]:
                control_saved = evidence.read(seed, "control", controller)[1] if "control" in evidence.spec["conditions"] else None
                for condition in evidence.spec["conditions"]:
                    row, recording = evidence.read(seed, condition, controller)
                    same(row, index[(seed, condition, controller)], "case and aggregate summary")
                    result = verify_recording(row, recording, first_seed=evidence.spec["seeds"][0], policy=policy, controller_kind=cs["kind"])
                    counts["cases"] += 1
                    counts["decisions"] += result["decisions"]
                    counts["retained_states"] += result["retained_states"]
                    if control_saved is not None and condition not in ("ordinary", "control"):
                        onset = int(condition_spec(condition)["onset_h"]*4)
                        same(recording["decision_trace"][:onset], control_saved["decision_trace"][:onset], "paired pre-event inference and reward prefix")
                        same(recording["decision_trace"][onset]["evidence"], control_saved["decision_trace"][onset]["evidence"], "no privileged action warning at outage onset")
                        counts["exact_control_prefixes"] += 1
                        if recording["minute_truth_states"]:
                            same(recording["minute_truth_states"][:onset*15+1], control_saved["minute_truth_states"][:onset*15+1], "paired pre-event physical prefix")
                            counts["exact_physical_prefixes"] += 1
                print(json.dumps({"verified_controller": controller, "seed": seed, "cases_checked": counts["cases"]}), flush=True)
        finally:
            if policy is not None and hasattr(policy, "close"):
                policy.close()
            if env is not None:
                env.close()
    evidence.verify_sources()
    report = {"schema": SCHEMA + "-integrity", "verified": True, "all_inference_recomputed": recompute_inference,
              **counts, "freeze_sha256": evidence.spec["sha256"], "sources_and_checkpoints_unchanged": True,
              "completed_at": datetime.now(timezone.utc).isoformat(), "verifier": file_pin(__file__),
              "freeze": file_pin(evidence.freeze_path), "summary": file_pin(summary_path)}
    publish(Path(output) / "review.json", report)
    return report


def replay_one(evidence, condition, controller, *, deadline=None):
    from docs.research.tools.learned_control_v1 import environment, evaluate
    from docs.research.tools.wholeplant_ppo_v1 import evaluate as audit_tools
    from docs.research.tools.tail_stress_v1.audit import MinuteAudit as TailMinuteAudit
    seed = evidence.spec["seeds"][0]
    row, saved = evidence.read(seed, condition, controller)
    verify_recording(row, saved, first_seed=seed)
    env = policy = None
    started = time.monotonic()
    with evidence.relocated_gate():
        try:
            env = environment.FixedEnv(condition=condition, split=evidence.spec["split"],
                gate_path=evidence.resolve(evidence.spec["gate"]) if evidence.spec.get("gate") else None, record_minutes=True)
            observation, _ = env.reset(seed=seed)
            initial = env.simulator.state.to_dict()
            same(initial, row["initial_state"], "replay initial state")
            audit = audit_tools.MinuteAudit(env) if condition == "ordinary" else TailMinuteAudit(env)
            operating = audit_tools.OperatingAudit(env)
            audit.add(initial)
            operating.add(initial)
            policy = load_policy(evidence, controller, env)
            cs = evidence.spec["controllers"][controller]
            witness, reward_sum, checked = hashlib.sha256(), 0.0, 1
            for index, expected in enumerate(saved["decision_trace"]):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Replay budget exhausted")
                if not np.array_equal(observation, env.current_policy_observation):
                    raise ValueError("Replay input differs from actual public observation")
                details = inference(policy, cs["kind"], observation)
                same(details, expected["evidence"], "replayed actual policy inference")
                action = np.asarray(details["action"], dtype=np.float32)
                witness.update(canonical(details))
                observation, reward, terminated, truncated, info = env.step(action)
                same(float(reward), expected["reward_eur"], "replayed reward")
                reward_sum += float(reward)
                trace = env.last_minute_truth_states
                audit_tools.audit_intervention(env, action, info, trace)
                for state in trace[1:]:
                    same(state.to_dict(), saved["minute_truth_states"][checked], "replayed physical minute")
                    audit.add(state)
                    operating.add(state)
                    checked += 1
                if terminated != (index == 671) or truncated:
                    raise ValueError("Replayed episode boundary differs")
                event = row["event_metrics"]
                for name, boundary in (("pre_event_state", event["onset_h"]), ("post_event_state", event["end_h"])):
                    if boundary is not None and env.physics_steps_taken == round(boundary*60):
                        same(env.simulator.state.to_dict(), event[name], "replayed event state")
            final = env.simulator.state.to_dict()
            actual = {"final_state": final, "delivery": env.simulator.delivery_audit(), "economic_metrics": env.economic_metrics.to_dict(),
                "physical_metrics": env.metrics.to_dict(), "operating_metrics": operating.totals, "minute_audit": audit.result(),
                "economic_audit": audit_tools.economic_audit(env, initial, final, reward_sum, operating),
                "input_action_witness_sha256": witness.hexdigest(),
                "physical_solar_forcing_sha256": digest([float(v) for v in env.frozen_source.expected_solar_kw])}
            for key, value in actual.items():
                same(value, row[key], "replayed " + key)
            return {"schema": SCHEMA + "-replay", "verified": True, "scientific_outcome_attempt": False,
                "seed": seed, "condition": condition, "controller": controller, "decisions": 672, "physical_states": checked,
                "all_inputs_actions_rewards_and_states_exact": True, "normalizer_and_actor_exact": cs["kind"] != "baseline",
                "freeze_sha256": evidence.spec["sha256"], "recording": row["recording"],
                "wall_time_s": time.monotonic()-started, "completed_at": datetime.now(timezone.utc).isoformat(), "verifier": file_pin(__file__)}
        finally:
            if policy is not None and hasattr(policy, "close"):
                policy.close()
            if env is not None:
                env.close()


def replay_panel(evidence, output, *, max_seconds=1200, conditions=None, controllers=None):
    from docs.research.tools.wholeplant_ppo_v1 import training
    training.configure_torch()
    conditions = list(evidence.spec["conditions"] if conditions is None else conditions)
    controllers = list(evidence.spec["controllers"] if controllers is None else controllers)
    if len(set(conditions)) != len(conditions) or len(set(controllers)) != len(controllers):
        raise ValueError("Replay identities must be unique")
    deadline, receipts = time.monotonic()+max_seconds, []
    for condition in conditions:
        for controller in controllers:
            result = replay_one(evidence, condition, controller, deadline=deadline)
            path = Path(output) / f"replay-{condition}--{controller}.json"
            publish(path, result)
            receipts.append(file_pin(path))
            print(json.dumps({"replayed": len(receipts), "condition": condition, "controller": controller}), flush=True)
    evidence.verify_sources()
    report = {"schema": SCHEMA + "-replay-panel", "verified": True, "freeze_sha256": evidence.spec["sha256"],
              "complete_prescribed_panel": conditions == evidence.spec["conditions"] and controllers == list(evidence.spec["controllers"]),
              "case_count": len(receipts), "receipts": receipts, "completed_at": datetime.now(timezone.utc).isoformat()}
    publish(Path(output) / "replay-summary.json", report)
    return report


def verify_bundle(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest.get("schema") != SCHEMA + "-bundle":
        raise ValueError("Unknown learned-control bundle")
    same(manifest["sha256"], digest({k: v for k, v in manifest.items() if k != "sha256"}), "bundle manifest digest")
    seen, originals = {}, {}
    for item in manifest["files"]:
        if item["path"] in seen or item.get("original_path") in originals:
            raise ValueError("Duplicate bundled member or original path")
        path = member_path(root, item["path"])
        if path.stat().st_size != item["bytes"] or file_sha(path) != item["sha256"]:
            raise ValueError("Bundle member bytes changed: " + item["path"])
        seen[item["path"]] = item
        if item.get("original_path"):
            originals[item["original_path"]] = item
    for pin in manifest["scientific_bindings"]:
        item = originals.get(pin["path"])
        if item is None or any(item[k] != pin[k] for k in ("bytes", "sha256")):
            raise ValueError("Bundle lacks an authentic scientific binding")
    return manifest


def require_release_reviews(evidence, integrity_path, replay_path):
    integrity, replay = (json.loads(Path(p).read_text()) for p in (integrity_path, replay_path))
    if (not integrity.get("verified") or not integrity.get("all_inference_recomputed")
            or integrity.get("cases") != evidence.spec["expected_cases"]
            or integrity.get("decisions") != 672*evidence.spec["expected_cases"]
            or integrity.get("freeze_sha256") != evidence.spec["sha256"]):
        raise ValueError("Complete independently recomputed inference review is required")
    expected = {(evidence.spec["seeds"][0], c, p) for c in evidence.spec["conditions"] for p in evidence.spec["controllers"]}
    if (not replay.get("verified") or not replay.get("complete_prescribed_panel")
            or replay.get("case_count") != len(expected) or replay.get("freeze_sha256") != evidence.spec["sha256"]):
        raise ValueError("Complete prescribed physical replay panel is required")
    actual = []
    for pin in replay["receipts"]:
        receipt = json.loads(evidence.resolve(pin).read_text())
        if (not receipt.get("verified") or not receipt.get("all_inputs_actions_rewards_and_states_exact")
                or receipt.get("freeze_sha256") != evidence.spec["sha256"]
                or receipt.get("decisions") != 672 or receipt.get("physical_states") != 10081):
            raise ValueError("Prescribed replay receipt is invalid")
        actual.append((receipt["seed"], receipt["condition"], receipt["controller"]))
        if (evidence.spec["controllers"][receipt["controller"]]["kind"] != "baseline"
                and not receipt.get("normalizer_and_actor_exact")):
            raise ValueError("Learned replay lacks exact normalizer and actor evidence")
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Prescribed replay receipt identities differ")
    return integrity, replay


def checkpoint_plan(metadata_path, metadata, root):
    """Find the nearby frozen plan by its canonical hash, never by name alone."""
    if not {"plan_sha256", "checkpoint", "normalizer"}.issubset(metadata):
        return None
    root = Path(root).resolve()
    for parent in Path(metadata_path).resolve().parents:
        if not parent.is_relative_to(root):
            break
        candidate = parent / "plan.json"
        if candidate.is_file():
            plan = json.loads(candidate.read_text())
            if digest(plan) == metadata["plan_sha256"]:
                return candidate
    return None


def create_bundle(evidence, output, *, include_paths=(), max_uncompressed_bytes=8_000_000_000,
                  integrity_path=None, replay_path=None):
    """Package explicit final evidence/dependencies; never scan all pilot runs."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if evidence.spec["split"] != "final":
        raise ValueError("Release packaging requires the completed final panel")
    integrity_path = Path(integrity_path or evidence.results / "postflight/integrity/review.json")
    replay_path = Path(replay_path or evidence.results / "postflight/replay/replay-summary.json")
    require_release_reviews(evidence, integrity_path, replay_path)
    paths = set()
    for path in evidence.results.rglob("*"):
        if path.is_file() and not path.is_symlink() and not any(p in {"__pycache__", ".build", ".layout"} for p in path.relative_to(evidence.results).parts):
            paths.add(path)
    for path in include_paths:
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError("Extra package paths must be explicit regular files")
        paths.add(path)
    paths.add(Path(__file__).resolve())
    paths.add(Path(__file__).with_name("__init__.py"))
    # Older freeze source lists intentionally excluded release helpers. The
    # evaluator imports this hash/path helper, so it is a runtime dependency.
    paths.add(evidence.root / "docs/research/tools/wholeplant_ppo_v1/portable.py")
    paths.update((integrity_path.resolve(), replay_path.resolve()))
    if evidence.spec.get("selected_recipe"):
        paths.add(evidence.resolve(evidence.spec["selected_recipe"]))
    if evidence.spec.get("gate"):
        paths.add(evidence.resolve(evidence.spec["gate"]))
    for controller in evidence.spec["controllers"].values():
        if controller["kind"] == "learned":
            metadata_path = evidence.resolve(controller["checkpoint_files"]["metadata"])
            plan = checkpoint_plan(metadata_path, json.loads(metadata_path.read_text()), evidence.root)
            if plan is None:
                raise ValueError("Selected learned checkpoint lacks its authenticated training plan")
            paths.add(plan)
    pending = [evidence.freeze_path, *paths]
    scanned, bindings = set(), {}
    while pending:
        path = pending.pop()
        if path in scanned:
            continue
        scanned.add(path)
        if path.suffix != ".json":
            continue
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            plan = checkpoint_plan(path, payload, evidence.root)
            if plan is not None and plan not in paths:
                paths.add(plan)
                pending.append(plan)
        for pin in iter_pins(payload):
            resolved = evidence.resolve(pin)
            prior = bindings.get(pin["path"])
            if prior is not None and prior != pin:
                raise ValueError("Conflicting scientific file pins")
            bindings[pin["path"]] = pin
            if resolved not in paths:
                paths.add(resolved)
                pending.append(resolved)
    files = []
    for path in sorted(paths):
        if not path.is_relative_to(evidence.root):
            raise ValueError("Package file is outside the evidence root")
        relative = path.relative_to(evidence.root)
        member_path(evidence.root, relative)
        pin = file_pin(path)
        files.append({"path": relative.as_posix(), "original_path": str(evidence.original_root / relative),
                      "bytes": pin["bytes"], "sha256": pin["sha256"]})
    total = sum(item["bytes"] for item in files)
    if total > max_uncompressed_bytes:
        raise ValueError(f"Explicit package exceeds disk budget: {total} bytes")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < total + 512_000_000:
        raise ValueError("Insufficient free space for bounded release package")
    manifest = {"schema": SCHEMA + "-bundle", "created_at": datetime.now(timezone.utc).isoformat(),
                "files": files, "scientific_bindings": list(bindings.values()), "results": evidence.results.relative_to(evidence.root).as_posix(),
                "uncompressed_bytes": total, "local_review_only": True, "freeze_sha256": evidence.spec["sha256"]}
    manifest["sha256"] = digest(manifest)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for item in files:
            compression = zipfile.ZIP_STORED if item["path"].endswith((".gz", ".zip")) else zipfile.ZIP_DEFLATED
            archive.write(evidence.root / item["path"], item["path"], compress_type=compression)
        archive.writestr(MANIFEST, json.dumps(manifest, indent=2)+"\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Release archive failed CRC verification")
        for item in files:
            with archive.open(item["path"]) as stream:
                sha = hashlib.sha256()
                for block in iter(lambda: stream.read(1024*1024), b""):
                    sha.update(block)
            if sha.hexdigest() != item["sha256"]:
                raise ValueError("Archived member differs")
    return {"archive": file_pin(output), "files": len(files), "uncompressed_bytes": total,
            "manifest_sha256": manifest["sha256"], "all_archived_member_hashes_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "replay", "manifest", "bundle"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--include", type=Path, nargs="*", default=[])
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--controllers", nargs="+")
    parser.add_argument("--stored-witnesses-only", action="store_true")
    args = parser.parse_args()
    if args.command == "manifest":
        result = verify_bundle(args.root)
        print(json.dumps({"verified": True, "files": len(result["files"]), "manifest_sha256": result["sha256"]}))
        return
    if args.results is None or args.output is None:
        parser.error("--results and --output are required")
    evidence = Evidence(args.results, root=args.root)
    if args.command == "verify":
        result = verify_all(evidence, args.output, recompute_inference=not args.stored_witnesses_only)
    elif args.command == "replay":
        result = replay_panel(evidence, args.output, max_seconds=args.max_seconds, conditions=args.conditions, controllers=args.controllers)
    else:
        result = create_bundle(evidence, args.output, include_paths=args.include, max_uncompressed_bytes=args.max_bytes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
