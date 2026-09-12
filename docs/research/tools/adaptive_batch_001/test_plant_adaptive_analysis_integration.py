"""Closed fabricated artifacts through audit and descriptive report; no resets."""

from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path

from plant_backend.rl.v2.metrics import EpisodeMetricsV2


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


graph = load(
    "closure_fixtures_for_integration",
    str(Path(__file__).with_name("test_plant_adaptive_closure_audit.py")),
)
report_fixtures = load(
    "report_fixtures_for_integration",
    str(Path(__file__).with_name("test_plant_adaptive_report.py")),
)
audit, reporter = graph.audit, report_fixtures.reporter


def test_closed_artifact_graph_passes_audit_and_report_without_subset_wins(
    tmp_path, monkeypatch
):
    original = audit.read_regular

    def no_live(path):
        assert not str(path).startswith(str(audit.batch.ROOT))
        return original(path)

    monkeypatch.setattr(audit, "read_regular", no_live)
    root = graph.fabricate(tmp_path / "batch")
    for arm in audit.batch.ARMS:
        diagnostics = report_fixtures.fake_training_data()["diagnostics"]
        for index, record in enumerate(diagnostics, 1):
            record.update(
                schema="plant-research-ppo-rollout-diagnostics-v1",
                algorithm_id=arm.algorithm,
            )
            graph.write(
                root
                / "training"
                / arm.name
                / "diagnostics"
                / f"diagnostic-{index:03d}.json",
                record,
            )
    for seed in audit.batch.DEV_SEEDS:
        hdir = root / "evaluations" / f"{seed}-H"
        href = json.loads(gzip.decompress((hdir / "episode.json.gz").read_bytes()))
        for actor in audit.ACTORS:
            directory = root / "evaluations" / f"{seed}-{actor}"
            episode = json.loads(
                gzip.decompress((directory / "episode.json.gz").read_bytes())
            )
            if actor == "B-pi-scaled":
                model_actor = episode["summary"]["actor"]
                episode = deepcopy(href)
                episode["summary"].update(actor=model_actor, policy_id=actor)
            summary = episode["summary"]
            complete = summary["execution_complete"]
            metrics = EpisodeMetricsV2(
                decisions=summary["decisions"],
                physical_steps=summary["physical_steps"],
                hours=168.0 if complete else 0.0,
                requested_kg=6720.0 if complete else 0.0,
                delivered_kg=6000.0 if complete else 0.0,
                unmet_kg=720.0 if complete else 0.0,
                commitments_settled=28 if complete else 0,
                commitments_met=24 if complete else 0,
            ).to_dict()
            summary["physical_metrics"] = metrics
            benefit = 1000.0 if actor == "B-pi-scaled" else 0.0
            summary["economic_metrics"] = {
                "net_operating_value_eur": -10000.0 + benefit,
                "components_eur": {"service": 2000.0, "operation": -12000.0 + benefit},
            }
            summary.update(fallback_decisions=0, recovery_decisions=0)
            raw = audit.canonical_bytes(episode)
            archive = directory / "episode.json.gz"
            archive.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
            graph.write(
                directory / "summary.json",
                {
                    "actor": actor,
                    "seed": seed,
                    "summary": summary,
                    "archive": graph.identity(archive),
                },
            )
            replay = json.loads((directory / "replay.json").read_text())
            replay.update(
                evidence_sha256=audit.sha(raw),
                decisions=summary["decisions"],
                physical_steps=summary["physical_steps"],
                execution_complete=complete,
                scope="complete_episode" if complete else "retained_physical_prefix",
            )
            graph.write(directory / "replay.json", replay)
    packet = audit.audit_closed_batch(root, fixture=True)
    report = reporter.make_report(packet)
    assert (
        packet["integrity_verified"] and packet["integrity"]["evaluation_slots"] == 80
    )
    assert report["actors"]["H"]["aggregate"]["delivered_kg"]["mean"] == 6000.0
    assert (
        report["actors"]["B-pi-scaled"]["aggregate"]["delivered_kg"]["mean"] == 6000.0
    )
    pair = next(
        p
        for p in report["operational_comparisons"]
        if p["candidate"] == "B-pi-scaled" and p["comparator"] == "H"
    )
    assert pair["paired_differences"]["net_operating_value_eur"]["mean"] == 1000.0
    assert pair["candidate_all_cases_hard_safety_passed"] is False
    assert pair["confirmation_or_material_benefit_claim"] is False
    assert report["actors"]["E-beta"]["aggregate"] is None
    assert "unavailable" in reporter.markdown(report)
