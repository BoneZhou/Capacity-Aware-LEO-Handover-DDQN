from __future__ import annotations

import argparse
import ast
import csv
import hashlib
from pathlib import Path


ACTION_TO_RATE_ID = (
    36, 27, 18, 10, 2, 44, 35, 26, 17, 9,
    1, 43, 34, 25, 16, 8, 0, 42, 33, 24,
    15, 7, 49, 41, 32, 23, 14, 6, 48, 40,
    31, 22, 13, 5, 47, 39, 30, 21, 12, 4,
    46, 38, 29, 3, 11, 19, 20, 28, 37, 45,
)
METHODS = ("GBW2", "MAC", "MVT", "MSH", "MSHBO")
UE_COUNTS = (10, 15, 20, 25, 30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_mapping(project_root: Path) -> list[dict[str, object]]:
    input_lines = (project_root / "input.txt").read_text(encoding="utf-8").splitlines()
    throughput_input_lines = (
        project_root / "throughput" / "input.txt"
    ).read_text(encoding="utf-8").splitlines()
    satellite_count = int(input_lines[0].split()[0])
    visibility = [tuple(map(int, line.split()[:2])) for line in input_lines[1:1 + satellite_count]]
    if input_lines[1:1 + satellite_count] != throughput_input_lines[1:1 + satellite_count]:
        raise RuntimeError("Root and throughput input.txt satellite rows differ")

    if sha256(project_root / "input_angle.txt") != sha256(project_root / "throughput" / "input_angle.txt"):
        raise RuntimeError("Root and throughput input_angle.txt files differ")

    rl_source = (project_root / "Dueling_DDQN" / "Dueling_ddqn.py").read_text(encoding="utf-8")
    rl_tree = ast.parse(rl_source)
    rl_mapping = None
    for node in rl_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "INPUT_ACTION_TO_RATE_ID" for target in node.targets):
            if isinstance(node.value, ast.Call) and node.value.args:
                rl_mapping = tuple(ast.literal_eval(node.value.args[0]))
            break
    if rl_mapping != ACTION_TO_RATE_ID:
        raise RuntimeError("Audit mapping differs from Dueling_DDQN INPUT_ACTION_TO_RATE_ID")

    positive_times: dict[int, list[int]] = {satellite_id: [] for satellite_id in range(satellite_count)}
    with (project_root / "input_angle.txt").open("r", encoding="utf-8") as stream:
        for line in stream:
            satellite_id_text, time_text, rate_text = line.split()
            if float(rate_text) > 0.0:
                positive_times[int(satellite_id_text)].append(int(time_text))

    if satellite_count != len(ACTION_TO_RATE_ID):
        raise RuntimeError(f"Expected {len(ACTION_TO_RATE_ID)} satellites, got {satellite_count}")
    if sorted(ACTION_TO_RATE_ID) != list(range(satellite_count)):
        raise RuntimeError("Action-to-rate mapping is not a permutation of the rate-table Satellite IDs")

    rows: list[dict[str, object]] = []
    for action_id, (visible_begin, visible_end) in enumerate(visibility):
        rate_id = ACTION_TO_RATE_ID[action_id]
        mapped_times = positive_times.get(rate_id, [])
        direct_times = positive_times.get(action_id, [])
        if not mapped_times:
            raise RuntimeError(f"Mapped Satellite ID {rate_id} has no positive-rate samples")
        mapped_span = (min(mapped_times), max(mapped_times) + 1)
        direct_span = (min(direct_times), max(direct_times) + 1) if direct_times else (None, None)
        rows.append(
            {
                "policy_action_id": action_id,
                "rate_table_satellite_id": rate_id,
                "input_visible_begin": visible_begin,
                "input_visible_end_exclusive": visible_end,
                "mapped_rate_positive_begin": mapped_span[0],
                "mapped_rate_positive_end_exclusive": mapped_span[1],
                "mapped_interval_matches": mapped_span == (visible_begin, visible_end),
                "direct_rate_positive_begin": direct_span[0],
                "direct_rate_positive_end_exclusive": direct_span[1],
                "direct_interval_matches": direct_span == (visible_begin, visible_end),
            }
        )

    mismatches = [row for row in rows if not row["mapped_interval_matches"]]
    if mismatches:
        raise RuntimeError(f"Mapping validation failed for {len(mismatches)} action IDs")
    return rows


def compare_results(project_root: Path, output_dir: Path) -> list[dict[str, object]]:
    old_rows = read_csv(project_root / "throughput" / "result_throughput.csv")
    new_rows = read_csv(output_dir / "nonrl_throughput_c5_corrected.csv")

    old_lookup = {
        (("GBW2" if row["algorithm"] == "GBW" else row["algorithm"]), int(row["m"])):
            float(row["avg_throughput_all_users_mbps"])
        for row in old_rows
        if int(row["c"]) == 5
    }
    new_lookup = {
        (row["algorithm"], int(row["m"])): float(row["avg_throughput_all_users_mbps"])
        for row in new_rows
        if int(row["c"]) == 5
    }

    expected = {(method, ue_count) for method in METHODS for ue_count in UE_COUNTS}
    if set(new_lookup) != expected:
        raise RuntimeError(f"Corrected output scenarios differ from expected set: {set(new_lookup) ^ expected}")
    missing_old = expected - set(old_lookup)
    if missing_old:
        raise RuntimeError(f"Legacy result is missing scenarios: {sorted(missing_old)}")

    rates: dict[tuple[int, int], float] = {}
    with (project_root / "input_angle.txt").open("r", encoding="utf-8") as stream:
        for line in stream:
            satellite_id_text, time_text, rate_text = line.split()
            rates[(int(satellite_id_text), int(time_text))] = float(rate_text)

    rows: list[dict[str, object]] = []
    for method in METHODS:
        for ue_count in UE_COUNTS:
            old_value = old_lookup[(method, ue_count)]
            new_value = new_lookup[(method, ue_count)]
            policy_path = (
                project_root / "throughput" / method / f"{method}_m{ue_count}_c5_policy.csv"
            )
            direct_sum = 0.0
            mapped_sum = 0.0
            valid_rows = 0
            seen: set[tuple[int, int]] = set()
            for policy_row in read_csv(policy_path):
                time = int(policy_row["t"])
                user = int(policy_row["user"])
                action_id = int(policy_row["sat"])
                if not 0 <= time < 3599:
                    continue
                if not 0 <= user < ue_count:
                    raise RuntimeError(f"Invalid user {user} in {policy_path}")
                if (time, user) in seen:
                    raise RuntimeError(f"Duplicate (time,user)=({time},{user}) in {policy_path}")
                seen.add((time, user))
                valid_rows += 1
                if action_id == -1:
                    continue
                if not 0 <= action_id < len(ACTION_TO_RATE_ID):
                    raise RuntimeError(f"Unmappable Satellite ID {action_id} in {policy_path}")
                direct_sum += rates[(action_id, time)]
                mapped_sum += rates[(ACTION_TO_RATE_ID[action_id], time)]

            expected_rows = ue_count * 3599
            if valid_rows != expected_rows or len(seen) != expected_rows:
                raise RuntimeError(
                    f"Incomplete policy window in {policy_path}: {valid_rows}/{expected_rows} rows"
                )
            direct_same_window = direct_sum / expected_rows
            mapped_independent = mapped_sum / expected_rows
            legacy_direct_3600 = direct_sum / (ue_count * 3600)
            if abs(mapped_independent - new_value) > 1e-6:
                raise RuntimeError(
                    f"Independent mapped result differs from C++ for {method}, UE={ue_count}"
                )
            if abs(legacy_direct_3600 - old_value) > 1e-6:
                raise RuntimeError(
                    f"Legacy result does not match direct indexing for {method}, UE={ue_count}"
                )
            rows.append(
                {
                    "method": method,
                    "ue_count": ue_count,
                    "capacity": 5,
                    "throughput_before_legacy_mbps": f"{old_value:.6f}",
                    "direct_index_same_3599_window_mbps": f"{direct_same_window:.6f}",
                    "throughput_after_mbps": f"{new_value:.6f}",
                    "mapping_only_difference_mbps": f"{new_value - direct_same_window:.6f}",
                    "total_difference_vs_legacy_mbps": f"{new_value - old_value:.6f}",
                    "total_difference_vs_legacy_percent": f"{(new_value / old_value - 1.0) * 100.0:.6f}",
                    "legacy_matches_direct_index_3600": True,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the non-RL throughput mapping fix")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_rows = validate_mapping(project_root)
    comparison_rows = compare_results(project_root, output_dir)
    write_csv(output_dir / "mapping_validation.csv", list(mapping_rows[0]), mapping_rows)
    write_csv(output_dir / "throughput_before_after_c5.csv", list(comparison_rows[0]), comparison_rows)

    source_paths = [
        project_root / "input.txt",
        project_root / "input_angle.txt",
        project_root / "throughput" / "input.txt",
        project_root / "throughput" / "input_angle.txt",
        project_root / "throughput" / "result_throughput.cpp",
        project_root / "throughput" / "result_throughput.csv",
        project_root / "Dueling_DDQN" / "Dueling_ddqn.py",
    ]
    for method in METHODS:
        for ue_count in UE_COUNTS:
            source_paths.append(
                project_root / "throughput" / method / f"{method}_m{ue_count}_c5_policy.csv"
            )
    source_rows = [
        {"source_file": str(path.resolve()), "sha256": sha256(path)} for path in source_paths
    ]
    write_csv(output_dir / "source_files.csv", ["source_file", "sha256"], source_rows)

    direct_matches = sum(bool(row["direct_interval_matches"]) for row in mapping_rows)
    summary_lines = [
        "Non-RL Baseline Throughput Mapping Fix",
        "",
        "Scope: throughput only; no handover or blocking calculation was run or changed.",
        "Policies: existing GBW2, MAC, MVT, MSH, and MSHBO policy CSV files; capacity=5.",
        "UE counts: 10, 15, 20, 25, 30.",
        "Valid slots: t=0..3598 (3599 slots), matching the formal RL evaluator.",
        "Formula: sum of raw successful-link Mbps / (UE count * 3599). BLOCK=-1 contributes 0.",
        f"Mapping validation: {len(mapping_rows)}/{len(mapping_rows)} mapped visibility intervals match input.txt.",
        f"Direct-ID validation: {direct_matches}/{len(mapping_rows)} intervals match without mapping.",
        "Legacy CSV check: all 25 c=5 values match direct Satellite ID indexing with its legacy 3600-slot divisor.",
        "Mapping source: the same 50-entry action-to-rate Satellite ID permutation used by Dueling_DDQN.",
        "Rate-table check: root and throughput input_angle.txt SHA-256 values are identical.",
        "Visibility check: root and throughput input.txt satellite rows are identical.",
        "All 25 requested scenarios were recalculated successfully.",
        "",
        "Outputs:",
        f"  Corrected throughput: {output_dir / 'nonrl_throughput_c5_corrected.csv'}",
        f"  Before/after comparison: {output_dir / 'throughput_before_after_c5.csv'}",
        f"  Mapping evidence: {output_dir / 'mapping_validation.csv'}",
        f"  Source hashes: {output_dir / 'source_files.csv'}",
    ]
    (output_dir / "audit_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
