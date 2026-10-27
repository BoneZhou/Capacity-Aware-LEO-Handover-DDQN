from __future__ import annotations

import argparse
import csv
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "scripts"))

from config_loader import (  # noqa: E402
    apply_training_defaults,
    experiment_config,
    method_config,
    reward_config_for,
    simulation_config,
)
from plot_paper_figures import validate_schema  # noqa: E402


class CentralizedConfigurationTests(unittest.TestCase):
    def test_formal_scenarios(self):
        scenarios = experiment_config()
        self.assertEqual(scenarios["ue_sensitivity"]["ue_counts"], [10, 15, 20, 25, 30])
        self.assertEqual(scenarios["ue_sensitivity"]["satellite_capacity"], 5)
        self.assertEqual(scenarios["capacity_sensitivity"]["capacities"], [1, 3, 5, 7, 9])
        self.assertEqual(scenarios["capacity_sensitivity"]["number_of_ues"], 25)

    def test_reward_defaults_and_cli_priority(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lambda_handover", type=float, default=999.0)
        parser.add_argument("--lambda_block", type=float, default=999.0)
        parser.add_argument("--lambda_throughput", type=float, default=999.0)
        parser.add_argument("--stay_bonus", type=float, default=999.0)
        parser.add_argument("--lookahead", type=int, default=999)
        parser.add_argument("--lookahead_penalty", type=float, default=999.0)
        parser.add_argument("--topk", type=int, default=999)
        parser.add_argument("--sticky_prob", type=float, default=999.0)
        parser.add_argument("--seed", type=int, default=999)
        parser.add_argument("--eval_order_seed", type=int, default=999)
        apply_training_defaults(parser, "standard_ddqn")
        defaults = parser.parse_args([])
        self.assertEqual(defaults.lambda_handover, 3.0)
        self.assertEqual(defaults.lambda_block, 12.0)
        self.assertEqual(defaults.lambda_throughput, 2.0)
        self.assertEqual(parser.parse_args(["--lambda_block", "7"]).lambda_block, 7.0)

    def test_reward_formula_regression(self):
        cfg = reward_config_for("proposed")
        normalized_rate, blocked, handover, stayed, lose_soon = 0.25, 1, 0, 0, 1
        before = -12.0 * blocked - 3.0 * handover + 2.0 * normalized_rate + 0.05 * stayed - 0.1 * lose_soon
        after = (-cfg["lambda_block"] * blocked - cfg["lambda_handover"] * handover
                 + cfg["lambda_throughput"] * normalized_rate + cfg["stay_bonus"] * stayed
                 - cfg["lookahead_penalty"] * lose_soon)
        self.assertEqual(before, after)
        self.assertEqual(cfg["rate_normalization_max_mbps"], 2946.289)

    def test_simulation_and_strategy_separation(self):
        sim = simulation_config()
        methods = method_config()
        self.assertEqual(sim["time"]["evaluated_slots"], 3599)
        self.assertEqual(methods["rl"]["top_k"], 3)
        self.assertEqual(methods["non_rl"]["MAC"]["minimum_stay_slots"], 150)
        self.assertNotIn("top_k", methods["non_rl"]["MAC"])

    def test_output_paths_are_repository_relative(self):
        for item in experiment_config().values():
            if isinstance(item, dict) and "output_directory" in item:
                output = Path(item["output_directory"])
                self.assertFalse(output.is_absolute())
                self.assertEqual(output.parts[0], "outputs")

    def test_plotting_schema_validator(self):
        folder = ROOT / "tests" / "_tmp_schema_test"
        path = folder / "sample.csv"
        folder.mkdir(exist_ok=False)
        try:
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["method", "ue_count", "capacity"])
                writer.writeheader()
                writer.writerow({"method": "Proposed", "ue_count": 25, "capacity": 5})
            validate_schema(path, {"method", "ue_count", "capacity"})
            with self.assertRaises(ValueError):
                validate_schema(path, {"method", "missing_metric"})
        finally:
            if path.exists():
                path.unlink()
            if folder.exists():
                folder.rmdir()


if __name__ == "__main__":
    unittest.main()
