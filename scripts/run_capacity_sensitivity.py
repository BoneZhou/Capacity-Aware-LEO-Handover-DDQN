"""Validate prerequisites and run the formal capacity-sensitivity evaluator."""
from __future__ import annotations

import argparse
from common import load_json, run_evaluator, validate_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the expensive evaluation after validation")
    args = parser.parse_args()
    scenario = load_json("experiment_scenarios.json")["capacity_sensitivity"]
    validate_inputs(["Proposed", "Dueling DDQN (No-MSHBO)", "DDQN", "Proposed w/o Remaining-Capacity State"])
    print(f"validated UE={scenario['number_of_ues']}, capacities={scenario['capacities']}")
    if args.execute:
        run_evaluator("evaluate_capacity_sensitivity.py")
    else:
        print("validation only; pass --execute to run the formal evaluation")


if __name__ == "__main__":
    main()
