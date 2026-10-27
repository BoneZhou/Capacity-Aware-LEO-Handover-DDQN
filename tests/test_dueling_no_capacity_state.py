import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import Dueling_ddqn as original
import Dueling_ddqn_no_capacity_state as ablated


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent
INPUT = ROOT / "input.txt"
RATE = ROOT / "input_angle.txt"
EXPERT = ROOT / "throughput" / "MSHBO" / "MSHBO_m25_c5_policy.csv"
OUTPUT = PROJECT / "runs" / "no_capacity_state_smoke_test"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_original_files():
    candidates = [
        PROJECT / "Dueling_ddqn.py",
        PROJECT / "checkpoints" / "dqn_imi.pt",
        PROJECT / "policy_actions_dqn_251.csv",
        PROJECT / "runs" / "proposed_persistent_topk_seed42_300ep" / "checkpoints" / "dqn_imi.pt",
        PROJECT / "runs" / "proposed_persistent_topk_seed42_300ep" / "policy_actions_dqn_251.csv",
        PROJECT / "runs" / "proposed_persistent_topk_seed42_300ep" / "run_metadata.json",
        PROJECT / "runs" / "proposed_persistent_topk_seed42_300ep" / "training_log.csv",
        INPUT,
        RATE,
        EXPERT,
    ]
    return [path for path in candidates if path.exists()]


def build_env(module):
    return module.HandoverEnv(
        str(INPUT),
        str(RATE),
        seed=42,
        training_order_seed=42,
        demonstration_order_seed=42,
        evaluation_order_seed=42,
    )


def write_synthetic_inputs(directory):
    directory.mkdir(parents=True, exist_ok=False)
    input_path = directory / "input.txt"
    rate_path = directory / "input_angle.txt"
    expert_path = directory / "mshbo.csv"

    with open(input_path, "x", encoding="utf-8") as handle:
        handle.write("50 2 0 3\n")
        for _ in range(50):
            handle.write("0 3 2\n")

    with open(rate_path, "x", encoding="utf-8") as handle:
        for satellite in range(50):
            for time_slot in range(3):
                rate = ablated.RATE_NORMALIZATION_MAX_MBPS if (satellite, time_slot) == (0, 0) else 100.0
                handle.write(f"{satellite} {time_slot} {rate}\n")

    with open(expert_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "user", "sat"])
        for time_slot in range(3):
            writer.writerow([time_slot, 0, 0])
            writer.writerow([time_slot, 1, 1])

    return input_path, rate_path, expert_path


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite smoke-test output: {OUTPUT}")

    tracked = tracked_original_files()
    hashes_before = {str(path): sha256(path) for path in tracked}

    original_env = build_env(original)
    ablated_env = build_env(ablated)
    original_env.reset_multi_user_episode()
    ablated_env.reset_multi_user_episode()

    original_state = original_env.make_state(0, 0)
    ablated_state = ablated_env.make_state(0, 0)
    assert original_state.shape == (251,)
    assert ablated_state.shape == (201,)
    assert np.array_equal(np.concatenate([original_state[:50], original_state[100:]]), ablated_state)

    expected_remaining = np.maximum(0, original_env.cap - original_env.used[:, 0]) / np.maximum(1, original_env.cap)
    assert np.array_equal(original_state[50:100], expected_remaining.astype(np.float32))
    assert np.array_equal(original_env.legal_mask(0, 0), ablated_env.legal_mask(0, 0))

    visible_satellite = int(np.flatnonzero(original_env.V[:, 0])[0])
    capacity = int(original_env.cap[visible_satellite])
    for user in range(capacity):
        assert original_env.legal_mask(user, 0)[visible_satellite] == 1
        assert ablated_env.legal_mask(user, 0)[visible_satellite] == 1
        original_env.step_user(user, 0, visible_satellite)
        ablated_env.step_user(user, 0, visible_satellite)
        assert np.array_equal(original_env.occupancy, ablated_env.occupancy)

    next_user = capacity
    original_mask = original_env.legal_mask(next_user, 0)
    ablated_mask = ablated_env.legal_mask(next_user, 0)
    assert np.array_equal(original_mask, ablated_mask)
    assert original_mask[visible_satellite] == 0
    assert original_mask[original_env.block_action] == 1
    assert np.array_equal(
        original.topk_mask(
            original_mask,
            original_env.used,
            original_env.cap,
            0,
            int(original_env.last_sat[next_user]),
            original_env.block_action,
            3,
        ),
        ablated.topk_mask(
            ablated_mask,
            ablated_env.used,
            ablated_env.cap,
            0,
            int(ablated_env.last_sat[next_user]),
            ablated_env.block_action,
            3,
        ),
    )
    _, original_blocked = original_env.step_user(next_user, 0, visible_satellite)
    _, ablated_blocked = ablated_env.step_user(next_user, 0, visible_satellite)
    assert original_blocked and ablated_blocked
    assert original_env.occupancy[visible_satellite] == capacity
    assert ablated_env.occupancy[visible_satellite] == capacity
    assert np.all(ablated_env.occupancy <= ablated_env.cap)

    post_capacity_original_state = original_env.make_state(next_user, 0)
    post_capacity_ablated_state = ablated_env.make_state(next_user, 0)
    assert post_capacity_original_state[50 + visible_satellite] == 0.0
    assert np.array_equal(
        np.concatenate([post_capacity_original_state[:50], post_capacity_original_state[100:]]),
        post_capacity_ablated_state,
    )

    q = ablated.QNet(ablated_env.state_dim, ablated_env.n_actions)
    target_q = ablated.QNet(ablated_env.state_dim, ablated_env.n_actions)
    target_q.load_state_dict(q.state_dict())
    assert q(torch.tensor(post_capacity_ablated_state)).shape == (51,)

    expert_transitions, expert_stats = ablated_env.demo_rollouts_from_csv(
        str(EXPERT), topk=3, return_stats=True
    )
    assert expert_stats["expert_transitions_constructed"] == 89975
    assert expert_stats["expert_transitions_accepted"] == 72756
    assert expert_stats["expert_transitions_discarded"] == 17219
    assert expert_stats["total_illegal_satellite_actions"] == 17219
    assert all(transition.state.shape == (201,) for transition in expert_transitions)
    assert all(transition.next_state.shape == (201,) for transition in expert_transitions)

    replay = ablated.Replay(1024)
    for transition in expert_transitions[:512]:
        replay.push(*transition)
    sampled = replay.sample(32)
    assert np.stack(sampled.state).shape == (32, 201)
    assert np.stack(sampled.next_state).shape == (32, 201)
    optimizer = torch.optim.Adam(q.parameters(), lr=1e-3)
    assert ablated.ddqn_update(q, target_q, optimizer, replay, 32, 0.99, torch.device("cpu"))

    synthetic_dir = OUTPUT / "synthetic_data"
    synthetic_input, synthetic_rate, synthetic_expert = write_synthetic_inputs(synthetic_dir)
    train_output = OUTPUT / "train_run"
    args = SimpleNamespace(
        input=str(synthetic_input),
        input_angle=str(synthetic_rate),
        mshbo_csv=str(synthetic_expert),
        episodes=1,
        device="cpu",
        lr=1e-3,
        gamma=0.99,
        batch_size=2,
        replay_capacity=100,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=30000,
        target_update=2000,
        lambda_handover=3.0,
        lambda_block=12.0,
        lambda_throughput=2.0,
        stay_bonus=0.05,
        lookahead=5,
        lookahead_penalty=0.1,
        topk=3,
        sticky_prob=0.2,
        pretrain_updates=1,
        seed=42,
        training_order_seed=42,
        demo_order_seed=42,
        eval_order_seed=42,
        output_dir=str(train_output),
        checkpoint_out=None,
        policy_out=None,
    )
    metadata = ablated.train(args)
    assert metadata["state_dimension"] == 201
    assert metadata["model_variant"] == "Proposed Dueling DDQN + MSHBO w/o Remaining-Capacity State"
    assert metadata["pretraining_updates_actual"] == 1
    assert Path(metadata["checkpoint_path"]).name == "dqn_imi_no_capacity_state.pt"
    assert Path(metadata["checkpoint_path"]).is_file()
    assert Path(metadata["policy_export_path"]).name == "policy_actions_dqn_no_capacity_state_201.csv"
    with open(metadata["policy_export_path"], "r", newline="", encoding="utf-8") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 6

    hashes_after = {str(path): sha256(path) for path in tracked}
    assert hashes_before == hashes_after

    report = {
        "baseline_source": str(PROJECT / "Dueling_ddqn.py"),
        "baseline_source_sha256": hashes_before[str(PROJECT / "Dueling_ddqn.py")],
        "original_state": {
            "dimension": 251,
            "fields": [
                {"name": "visibility", "indices": "0..49", "length": 50},
                {"name": "remaining_capacity_ratio", "indices": "50..99", "length": 50},
                {"name": "last_satellite_one_hot", "indices": "100..149", "length": 50},
                {"name": "next_slot_visibility", "indices": "150..199", "length": 50},
                {"name": "normalized_rate", "indices": "200..249", "length": 50},
                {"name": "time_ratio", "indices": "250", "length": 1},
            ],
        },
        "remaining_capacity_formula": "max(0, capacity[s] - used[s,t]) / max(1, capacity[s])",
        "ablated_state": {
            "dimension": 201,
            "fields": [
                {"name": "visibility", "indices": "0..49", "length": 50},
                {"name": "last_satellite_one_hot", "indices": "50..99", "length": 50},
                {"name": "next_slot_visibility", "indices": "100..149", "length": 50},
                {"name": "normalized_rate", "indices": "150..199", "length": 50},
                {"name": "time_ratio", "indices": "200", "length": 1},
            ],
        },
        "checks": {
            "only_remaining_capacity_removed": True,
            "state_and_next_state_are_201": True,
            "network_forward_pass": True,
            "replay_sample_and_ddqn_update": True,
            "actual_mshbo_expert_loader": True,
            "expert_pretraining_update": True,
            "action_masks_identical": True,
            "full_satellite_excluded": True,
            "capacity_never_exceeded": True,
            "training_and_evaluation_state_format": True,
            "tracked_original_files_unchanged": True,
        },
        "actual_expert_replay": {
            "constructed": expert_stats["expert_transitions_constructed"],
            "accepted": expert_stats["expert_transitions_accepted"],
            "discarded": expert_stats["expert_transitions_discarded"],
            "state_dimension": 201,
        },
        "smoke_training": {
            "synthetic_episodes": 1,
            "formal_300_episode_training_started": False,
            "checkpoint": metadata["checkpoint_path"],
            "policy": metadata["policy_export_path"],
        },
        "formal_experiment_prepared": {
            "ue_count": 25,
            "satellite_capacity": 5,
            "episodes": 300,
            "training_seed": 42,
            "evaluation_order_seeds": [42, 123, 2024, 2025, 2026],
        },
        "tracked_original_sha256_before_and_after": {
            path: {"before": hashes_before[path], "after": hashes_after[path]}
            for path in hashes_before
        },
    }
    report_path = OUTPUT / "validation_report.json"
    with open(report_path, "x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
