"""Validate prerequisites and run the fixed-scenario capacity-state ablation."""
from __future__ import annotations

import argparse
from common import load_json, run_evaluator, validate_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the expensive evaluation after validation")
    args = parser.parse_args()
    scenario = load_json("experiment_scenarios.json")["capacity_aware_ablation"]
    validate_inputs(scenario["methods"])
    print(f"validated UE={scenario['number_of_ues']}, capacity={scenario['satellite_capacity']}")
    if args.execute:
        run_evaluator("evaluate_capacity_ablation.py")
    else:
        print("validation only; pass --execute to run the formal evaluation")


if __name__ == "__main__":
    main()
