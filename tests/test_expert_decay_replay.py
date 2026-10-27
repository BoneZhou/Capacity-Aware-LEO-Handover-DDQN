import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import Dueling_ddqn as base
from Dueling_ddqn_expert_decay import (
    ScheduledExpertReplay,
    _write_artifacts_with_decay,
    build_parser,
    expert_ratio_at_episode,
    stochastic_sample_count,
)


def transition(value):
    state = np.array([value], dtype=np.float32)
    return base.Transition(
        state,
        0,
        state.copy(),
        float(value),
        0.0,
        np.ones(1, dtype=np.float32),
    )


def ready_replay(*, episode, batch_size=20, online_capacity=200):
    replay = ScheduledExpertReplay(
        online_capacity,
        pretraining_updates=1,
        transitions_per_episode=1,
    )
    for value in range(100):
        replay.push(*transition(value))
    replay.sample(1)
    for value in range(max(episode, batch_size)):
        replay.push(*transition(1000 + value))
    if replay.current_episode != max(episode, batch_size):
        raise AssertionError("test replay episode setup failed")
    return replay


class ExpertDecayReplayTests(unittest.TestCase):
    def test_ratio_schedule(self):
        self.assertAlmostEqual(expert_ratio_at_episode(0), 0.10)
        self.assertAlmostEqual(expert_ratio_at_episode(50), 0.05)
        self.assertEqual(expert_ratio_at_episode(100), 0.0)
        self.assertEqual(expert_ratio_at_episode(300), 0.0)

    def test_stochastic_rounding_tracks_expected_ratio(self):
        np.random.seed(4)
        counts = [stochastic_sample_count(256, 0.10) for _ in range(1000)]
        self.assertTrue(set(counts).issubset({25, 26}))
        self.assertAlmostEqual(sum(counts) / (1000 * 256), 0.10, places=3)

    def test_mixed_batch_size_and_ratio(self):
        np.random.seed(1)
        replay = ready_replay(episode=50, batch_size=20)
        batch = replay.sample(20)
        self.assertEqual(len(batch.action), 20)
        self.assertEqual(replay.last_sample_counts, {"expert": 1, "online": 19})

    def test_episode_zero_samples_ten_percent(self):
        np.random.seed(2)
        replay = ScheduledExpertReplay(
            50, pretraining_updates=1, transitions_per_episode=100
        )
        for value in range(20):
            replay.push(*transition(value))
        replay.sample(1)
        for value in range(10):
            replay.push(*transition(100 + value))
        replay.sample(10)
        self.assertEqual(replay.current_episode, 0)
        self.assertEqual(replay.last_sample_counts, {"expert": 1, "online": 9})

    def test_episode_100_uses_no_expert(self):
        np.random.seed(3)
        replay = ready_replay(episode=100, batch_size=20)
        replay.sample(20)
        self.assertEqual(replay.last_sample_counts, {"expert": 0, "online": 20})

    def test_expert_buffer_survives_online_fifo(self):
        replay = ScheduledExpertReplay(
            5, pretraining_updates=1, transitions_per_episode=100
        )
        for value in range(10):
            replay.push(*transition(value))
        replay.sample(1)
        expert_ids = [id(item) for item in replay.expert]
        for value in range(20):
            replay.push(*transition(100 + value))
        self.assertEqual(len(replay.online), 5)
        self.assertEqual([id(item) for item in replay.expert], expert_ids)

    def test_online_warmup_is_not_filled_by_expert(self):
        replay = ScheduledExpertReplay(
            10, pretraining_updates=1, transitions_per_episode=100
        )
        for value in range(10):
            replay.push(*transition(value))
        replay.sample(1)
        for value in range(3):
            replay.push(*transition(100 + value))
        self.assertEqual(len(replay), 3)
        self.assertLess(len(replay), 4)

    def test_cli_defaults_match_formal_training(self):
        args = build_parser().parse_args(
            ["--input", "input.txt", "--input_angle", "rate.txt", "--mshbo_csv", "expert.csv"]
        )
        self.assertEqual(args.episodes, 300)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.lambda_handover, 3.0)
        self.assertEqual(args.topk, 3)
        self.assertEqual(args.replay_capacity, 200000)
        self.assertEqual(args.pretrain_updates, 2000)

    def test_artifacts_record_schedule_and_sample_counts(self):
        replay = ScheduledExpertReplay(
            10, pretraining_updates=1, transitions_per_episode=100
        )
        replay.expert.append(transition(1))
        replay.episode_sample_counts[0] = {"expert": 2, "online": 18}
        metadata = {"expert_replay_stats": {}, "counters": {}}
        episode_logs = [
            {
                "episode_index": 1,
                "episode_return": 0.0,
                "best_return_so_far": 0.0,
                "best_checkpoint_updated": True,
                "epsilon_first_slot": 1.0,
                "epsilon_last_slot": 1.0,
                "decisions": 1,
                "transitions_completed": 1,
                "blocks": 0,
                "handovers": 0,
                "gradient_updates": 1,
            }
        ]
        project_tmp = Path(__file__).resolve().parent.parent / "tmp"
        with tempfile.TemporaryDirectory(dir=project_tmp) as directory:
            paths = {
                "metadata_path": str(Path(directory) / "run_metadata.json"),
                "training_log_path": str(Path(directory) / "training_log.csv"),
            }
            _write_artifacts_with_decay(metadata, episode_logs, paths, replay)
            saved_metadata = json.loads(Path(paths["metadata_path"]).read_text(encoding="utf-8"))
            with open(paths["training_log_path"], newline="", encoding="utf-8") as file:
                saved_log = next(csv.DictReader(file))
        self.assertEqual(saved_metadata["expert_replay_schedule"]["decay_end_episode_index"], 100)
        self.assertEqual(saved_metadata["counters"]["main_training_expert_samples"], 2)
        self.assertEqual(saved_log["expert_samples"], "2")
        self.assertEqual(saved_log["online_samples"], "18")
        self.assertEqual(float(saved_log["actual_expert_ratio"]), 0.10)


if __name__ == "__main__":
    unittest.main()
