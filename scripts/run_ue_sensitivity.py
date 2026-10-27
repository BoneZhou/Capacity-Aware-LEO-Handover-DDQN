"""Validate prerequisites and run the formal UE-sensitivity evaluator."""
from __future__ import annotations

import argparse
from common import load_json, run_evaluator, validate_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the expensive evaluation after validation")
    args = parser.parse_args()
    scenario = load_json("experiment_scenarios.json")["ue_sensitivity"]
    validate_inputs(["Proposed", "Dueling DDQN (No-MSHBO)", "DDQN"])
    print(f"validated UE={scenario['ue_counts']}, capacity={scenario['satellite_capacity']}")
    if args.execute:
        run_evaluator("evaluate_rl_multiseed_lambda3.py")
    else:
        print("validation only; pass --execute to run the formal evaluation")


if __name__ == "__main__":
    main()
