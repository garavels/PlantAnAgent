"""Small deterministic arithmetic and field-extraction fixtures only."""

import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

MODULE = Path(__file__).with_name("batch009_economic_audit.py")
spec = importlib.util.spec_from_file_location("batch009_audit", MODULE)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def reward_row(index, components):
    c = {k: 0.0 for k in audit.COMPONENTS}
    c.update(components)
    value = sum(c.values())
    return {
        "decision_index": index,
        "reward_eur": value,
        "info": {"reward": {"total_eur": value, "components_eur": c}},
        "nested": {
            "text": "A string with }, and \\n brackets [] {}",
            "list": [{"a": 1}],
        },
    }


class ArithmeticTests(unittest.TestCase):
    def test_signs_and_rescore(self):
        c = {k: 0.0 for k in audit.COMPONENTS}
        c.update(
            delivered_service_value=40,
            unmet_delivery_penalty=-20,
            battery_throughput_reserve=-5,
        )
        self.assertEqual(audit.rescore(c, {}), 15)
        self.assertEqual(audit.rescore(c, {"battery_throughput_reserve": 2}), 10)
        self.assertEqual(audit.rescore(c, {"unmet_delivery_penalty": 0}), 35)

    def test_break_even(self):
        self.assertEqual(audit.break_even(-60, 80)["multiplier"], 1.75)
        self.assertEqual(audit.break_even(-60, -200)["multiplier"], 0.7)
        self.assertEqual(audit.break_even(-60, 0)["status"], "unchanged_slope")
        self.assertEqual(audit.break_even(-60, -20)["status"], "negative_factor_only")

    def test_ratios(self):
        self.assertEqual(audit.ratio(30, 10), 3)
        self.assertIsNone(audit.ratio(30, 0))

    def test_discount_first_reward_at_zero(self):
        rows = [
            reward_row(0, {"delivered_service_value": 20}),
            reward_row(1, {"unmet_delivery_penalty": -10}),
        ]
        r = audit.summarize_history(iter(rows), [0.5, 1.0])
        self.assertEqual(r["discounted"]["0.5"]["total_eur"], 15)
        self.assertEqual(r["discounted"]["1.0"]["total_eur"], 10)
        self.assertEqual(r["discounted"]["0.5"]["unmet_delivery_penalty"], -5)
        self.assertEqual(r["daily_undiscounted_components_eur"][0]["total_eur"], 10)
        self.assertEqual(
            r["daily_discounted_components_eur"]["0.5"][0]["total_eur"], 15
        )

    def test_day_boundary(self):
        rows = [reward_row(i, {"battery_throughput_reserve": -1}) for i in range(97)]
        r = audit.summarize_history(iter(rows), [1.0])
        self.assertEqual(r["daily_undiscounted_components_eur"][0]["total_eur"], -96)
        self.assertEqual(r["daily_undiscounted_components_eur"][1]["total_eur"], -1)

    def test_reward_extraction_rejects_mismatch(self):
        row = reward_row(0, {"delivered_service_value": 10})
        row["reward_eur"] = 11
        with self.assertRaises(ValueError):
            audit.extract_reward(row)

    def test_reward_extraction_rejects_missing(self):
        row = reward_row(0, {})
        del row["info"]["reward"]["components_eur"]["battery_throughput_reserve"]
        with self.assertRaises(ValueError):
            audit.extract_reward(row)

    def test_nonfinite_and_bool(self):
        for value in [float("nan"), float("inf"), True, "3"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.finite(value)

    def test_unknown_factor(self):
        with self.assertRaises(ValueError):
            audit.rescore({}, {"bogus": 1})

    def test_stats_keeps_all_values(self):
        r = audit.stats([-2, 0, 5])
        self.assertEqual(r["values"], [-2, 0, 5])
        self.assertEqual(r["mean"], 1)
        self.assertEqual((r["negative_count"], r["positive_count"]), (1, 1))


class HistoryTests(unittest.TestCase):
    def stream(self, rows):
        return io.BytesIO(
            gzip.compress(
                json.dumps(
                    {"config": {}, "history": rows, "summary": {}}, indent=2
                ).encode()
            )
        )

    def test_stream_decision_objects(self):
        rows = [reward_row(i, {"delivered_service_value": i}) for i in range(3)]
        self.assertEqual(list(audit.iter_history(self.stream(rows))), rows)

    def test_empty_history(self):
        self.assertEqual(list(audit.iter_history(self.stream([]))), [])

    def test_noncontiguous_order(self):
        with self.assertRaises(ValueError):
            list(audit.iter_history(self.stream([reward_row(1, {})])))

    def test_bad_layout_rejected(self):
        stream = io.BytesIO(gzip.compress(b'{"history": []}'))
        with self.assertRaises(ValueError):
            list(audit.iter_history(stream))

    def test_bad_gzip_crc(self):
        contents = self.stream([reward_row(0, {})]).getvalue()
        contents = contents[:-8] + b"\x00" * 8
        with self.assertRaises((OSError, EOFError)):
            list(audit.iter_history(io.BytesIO(contents)))

    def test_decompressed_limit(self):
        previous = audit.MAX_DECOMPRESSED
        try:
            audit.MAX_DECOMPRESSED = 3
            with self.assertRaises(ValueError):
                list(audit.iter_history(self.stream([])))
        finally:
            audit.MAX_DECOMPRESSED = previous


class CustodyTests(unittest.TestCase):
    def test_valid_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / "input.json"
            path.write_text("{}")
            with audit.verified_file(audit.file_record(path), 20) as stream:
                self.assertEqual(stream.read(), b"{}")

    def test_bad_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / "input.json"
            path.write_text("{}")
            r = audit.file_record(path)
            r["sha256"] = "0" * 64
            with self.assertRaises(ValueError), audit.verified_file(r, 20):
                pass

    def test_changed_during_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / "input.json"
            path.write_text("{}")
            with (
                self.assertRaises(ValueError),
                audit.verified_file(audit.file_record(path), 20),
            ):
                path.write_text("[]")

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / "input.json"
            path.write_text("{}")
            link = path.with_name("link")
            link.symlink_to(path)
            record = audit.file_record(path)
            record["path"] = str(link)
            with self.assertRaises(ValueError), audit.verified_file(record, 20):
                pass


class SummaryExtractionTests(unittest.TestCase):
    def fixture(self):
        slots = []
        for seed in audit.SEEDS:
            economic = {
                "components_eur": {k: 0.0 for k in audit.COMPONENTS},
                "net_operating_value_eur": 0.0,
                "battery_throughput_kwh": 12.0,
                "accepted_solar_kwh": 100.0,
                "delivered_kg": 4.0,
            }
            physical = {
                "electrolyser_nonproductive_energy_kwh": 6.0,
                "accepted_process_energy_kwh": 80.0,
                "ch4_produced_kg": 2.0,
            }
            slots.append(
                {
                    "seed": seed,
                    "eligible_for_descriptive_comparison": True,
                    "failures": [],
                    "summary": {
                        "decisions": 672,
                        "execution_complete": True,
                        "failure": None,
                        "economic_metrics": economic,
                        "physical_metrics": physical,
                    },
                }
            )
        return {"actors": {"OperatingMPC": {"slots": slots}}}

    def test_intensity_extraction_uses_produced_and_delivered(self):
        result = audit.extract_actor("OperatingMPC", self.fixture())
        proxies = result["aggregate"]["physical_intensity_proxies"]
        self.assertEqual(
            proxies["battery_throughput_kwh_per_ch4_produced_kg"]["ratio_of_totals"],
            6.0,
        )
        self.assertEqual(
            proxies["accepted_process_energy_kwh_per_delivered_kg"]["ratio_of_totals"],
            20.0,
        )
        self.assertEqual(len(result["rows"]), 8)

    def test_unstarted_failure_remains_excluded(self):
        fixture = self.fixture()
        row = fixture["actors"]["OperatingMPC"]["slots"][0]
        row["eligible_for_descriptive_comparison"] = False
        row["failures"] = ["stopped_before_first_decision"]
        row["summary"].update(decisions=0, execution_complete=False, failure="stopped")
        row["summary"]["economic_metrics"]["components_eur"] = {}
        result = audit.extract_actor("OperatingMPC", fixture)
        self.assertFalse(result["complete_panel"])
        self.assertIsNone(result["aggregate"])
        self.assertEqual(
            result["rows"][0]["failures"], ["stopped_before_first_decision"]
        )
        self.assertEqual(
            result["rows"][0]["components_eur"]["unmet_delivery_penalty"], 0
        )

    def test_partial_component_surface_rejected(self):
        fixture = self.fixture()
        del fixture["actors"]["OperatingMPC"]["slots"][0]["summary"][
            "economic_metrics"
        ]["components_eur"]["unmet_delivery_penalty"]
        with self.assertRaises((ValueError, KeyError)):
            audit.extract_actor("OperatingMPC", fixture)

    def test_wrong_completion_count_rejected(self):
        fixture = self.fixture()
        fixture["actors"]["OperatingMPC"]["slots"][0]["summary"]["decisions"] = 671
        with self.assertRaises(ValueError):
            audit.extract_actor("OperatingMPC", fixture)


if __name__ == "__main__":
    unittest.main()
