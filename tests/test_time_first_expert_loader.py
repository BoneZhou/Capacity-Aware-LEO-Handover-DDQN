import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from Dueling_ddqn import HandoverEnv, train


HERE = Path(__file__).resolve().parent
PROJECT_DATA_DIR = HERE.parent
TEST_TMP_DIR = HERE.parent / "tmp"
TEST_TMP_DIR.mkdir(exist_ok=True)


def synthetic_env(n, m, slots, capacities, visibility=None, seed=42):
    env = HandoverEnv.__new__(HandoverEnv)
    env.n = n
    env.m = m
    env.Tb = 0
    env.Te = slots
    env.T = slots
    env.cap = np.asarray(capacities, dtype=int)
    env.V = (
        np.ones((n, slots), dtype=np.uint8)
        if visibility is None
        else np.asarray(visibility, dtype=np.uint8)
    )
    env.rate_norm = np.ones((n, slots), dtype=np.float32)
    env.block_action = n
    env.n_actions = n + 1
    env.state_dim = n * 5 + 1
    env.lambda_h = 3.0
    env.lambda_b = 12.0
    env.lambda_t = 2.0
    env.stay_bonus = 0.05
    env.lookahead = 5
    env.lookahead_penalty = 0.1
    env.training_order_seed = seed
    env.demonstration_order_seed = seed
    env.evaluation_order_seed = seed
    env.training_order_rng = np.random.default_rng(seed)
    env.demo_order_rng = np.random.default_rng(seed)
    env.evaluation_order_rng = np.random.default_rng(seed)
    env.order_rng = env.training_order_rng
    env.reset_multi_user_episode()
    return env


def write_policy(path, rows, header=("t", "user", "sat")):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class ExpertCSVFailFastTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP_DIR)
        self.root = Path(self.temp.name)
        self.env = synthetic_env(2, 2, 2, [2, 2])
        self.complete_rows = [
            (t, user, -1)
            for t in range(self.env.T)
            for user in range(self.env.m)
        ]

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_row_is_rejected(self):
        path = self.root / "missing.csv"
        write_policy(path, self.complete_rows[:-1])
        with self.assertRaisesRegex(ValueError, "missing_rows=1"):
            self.env.demo_rollouts_from_csv(str(path))

    def test_duplicate_row_is_rejected(self):
        path = self.root / "duplicate.csv"
        write_policy(path, self.complete_rows + [self.complete_rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate_rows=1"):
            self.env.demo_rollouts_from_csv(str(path))

    def test_invalid_satellite_id_is_counted_and_rejected(self):
        path = self.root / "invalid_sat.csv"
        rows = list(self.complete_rows)
        rows[0] = (0, 0, self.env.n + 7)
        write_policy(path, rows)
        with self.assertRaisesRegex(ValueError, "invalid_satellite_ids=1"):
            self.env.demo_rollouts_from_csv(str(path))

    def test_expected_header_is_required(self):
        path = self.root / "bad_header.csv"
        write_policy(path, self.complete_rows, header=("time", "ue", "action"))
        with self.assertRaisesRegex(ValueError, "columns t,user,sat"):
            self.env.demo_rollouts_from_csv(str(path))

    def test_malformed_integer_row_is_rejected(self):
        path = self.root / "malformed.csv"
        rows = list(self.complete_rows)
        rows[0] = ("not-a-time", 0, -1)
        write_policy(path, rows)
        with self.assertRaisesRegex(ValueError, "Malformed MSHBO expert row"):
            self.env.demo_rollouts_from_csv(str(path))

    def test_zero_accepted_transitions_is_rejected(self):
        env = synthetic_env(1, 1, 1, [1], visibility=[[0]])
        path = self.root / "all_illegal.csv"
        write_policy(path, [(0, 0, 0)])
        with self.assertRaisesRegex(ValueError, "zero accepted transitions"):
            env.demo_rollouts_from_csv(str(path))

    def test_missing_or_non_file_path_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self.env.demo_rollouts_from_csv(str(self.root / "absent.csv"))
        with self.assertRaisesRegex(OSError, "regular file"):
            self.env.demo_rollouts_from_csv(str(self.root))

    def test_proposed_training_requires_expert_csv_before_setup(self):
        with self.assertRaisesRegex(ValueError, "requires --mshbo_csv"):
            train(SimpleNamespace(seed=42, mshbo_csv=None))
        with self.assertRaises(FileNotFoundError):
            train(SimpleNamespace(seed=42, mshbo_csv=str(self.root / "absent.csv")))


class ExpertLegalityAndTransitionTests(unittest.TestCase):
    def test_legality_filtering_preserves_time_first_pending_semantics(self):
        env = synthetic_env(
            n=2,
            m=4,
            slots=2,
            capacities=[1, 4],
            visibility=[[1, 1], [0, 1]],
            seed=42,
        )
        rows = [
            (0, 0, 0),
            (0, 1, 0),
            (0, 2, 1),
            (0, 3, -1),
        ] + [(1, user, 1) for user in range(4)]

        with tempfile.TemporaryDirectory(dir=TEST_TMP_DIR) as tmp:
            path = Path(tmp) / "expert.csv"
            write_policy(path, rows)
            transitions, stats = env.demo_rollouts_from_csv(
                str(path), topk=1, return_stats=True
            )

        self.assertTrue(stats["expert_data_complete"])
        self.assertEqual(stats["source_csv_rows"], 8)
        self.assertEqual(stats["in_window_rows"], 8)
        self.assertEqual(stats["missing_rows"], 0)
        self.assertEqual(stats["duplicate_rows"], 0)
        self.assertEqual(stats["explicit_block_actions"], 1)
        self.assertEqual(stats["legal_satellite_actions"], 5)
        self.assertEqual(stats["invisible_satellite_actions"], 1)
        self.assertEqual(stats["capacity_full_satellite_actions"], 1)
        self.assertEqual(stats["total_illegal_satellite_actions"], 2)
        self.assertEqual(stats["expert_transitions_constructed"], 8)
        self.assertEqual(stats["expert_transitions_accepted"], 6)
        self.assertEqual(stats["expert_transitions_discarded"], 2)
        self.assertEqual(len(transitions), 6)

        # The two illegal t=0 satellite decisions were stepped as blocking, but
        # their -12 transitions were not retained.  The explicit BLOCK remains.
        self.assertEqual(sum(tr.reward == -12.0 for tr in transitions), 1)
        self.assertEqual(sum(tr.done == 0.0 for tr in transitions), 2)
        self.assertEqual(sum(tr.done == 1.0 for tr in transitions), 4)
        self.assertEqual(int(env.used[0, 0]), 1)
        self.assertEqual(int(env.used[1, 1]), 4)

        terminal_t1 = [tr for tr in transitions if tr.done == 1.0]
        last_slice = slice(2 * env.n, 3 * env.n)
        histories = [int(np.count_nonzero(tr.state[last_slice])) for tr in terminal_t1]
        # One UE was legally connected at t=0.  Explicit BLOCK and both illegal
        # satellite actions correctly cleared history before their t=1 states.
        self.assertEqual(sorted(histories), [0, 0, 0, 1])

    def test_legal_action_outside_topk_is_retained(self):
        env = synthetic_env(4, 1, 1, [1, 1, 1, 1], seed=11)
        with tempfile.TemporaryDirectory(dir=TEST_TMP_DIR) as tmp:
            path = Path(tmp) / "expert.csv"
            write_policy(path, [(0, 0, 3)])
            transitions, stats = env.demo_rollouts_from_csv(
                str(path), topk=1, return_stats=True
            )
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].action, 3)
        self.assertEqual(stats["legal_satellite_actions"], 1)
        self.assertEqual(stats["legal_outside_topk_actions"], 1)

    def test_demo_rollout_does_not_advance_training_order_rng(self):
        env_with_demo = synthetic_env(1, 3, 2, [3], seed=73)
        untouched = synthetic_env(1, 3, 2, [3], seed=73)
        with tempfile.TemporaryDirectory(dir=TEST_TMP_DIR) as tmp:
            path = Path(tmp) / "expert.csv"
            write_policy(
                path,
                [(t, user, -1) for t in range(2) for user in range(3)],
            )
            env_with_demo.demo_rollouts_from_csv(str(path))

        after_demo = [
            env_with_demo.next_training_user_order().tolist() for _ in range(5)
        ]
        expected = [untouched.next_training_user_order().tolist() for _ in range(5)]
        self.assertEqual(after_demo, expected)


class RealExpertRegressionTests(unittest.TestCase):
    def test_current_m25_c5_expert_statistics(self):
        csv_path = (
            PROJECT_DATA_DIR
            / "throughput"
            / "MSHBO"
            / "MSHBO_m25_c5_policy.csv"
        )
        if not csv_path.is_file():
            self.skipTest("Public repository does not include generated MSHBO expert policy")
        env = HandoverEnv(
            str(PROJECT_DATA_DIR / "input.txt"),
            rate_path=str(PROJECT_DATA_DIR / "input_angle.txt"),
            seed=42,
            demonstration_order_seed=42,
        )
        transitions, stats = env.demo_rollouts_from_csv(
            str(csv_path), topk=3, return_stats=True
        )

        immutable_source_counts = {
            "source_csv_rows": 90000,
            "in_window_rows": 89975,
            "out_of_window_rows": 25,
            "missing_rows": 0,
            "duplicate_rows": 0,
            "explicit_block_actions": 6670,
            "invalid_satellite_ids": 0,
            "expert_transitions_constructed": 89975,
        }
        for key, value in immutable_source_counts.items():
            self.assertEqual(stats[key], value, key)
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest().upper()
        self.assertEqual(
            digest,
            "90B86B53EB6778C4F5FCABAB5D23C21E10DDD9D8E09C8CA588463E54A213B4D8",
        )
        self.assertTrue(stats["expert_data_complete"])
        self.assertEqual(
            stats["total_illegal_satellite_actions"],
            stats["invisible_satellite_actions"]
            + stats["capacity_full_satellite_actions"],
        )
        self.assertEqual(
            stats["expert_transitions_accepted"]
            + stats["expert_transitions_discarded"],
            stats["expert_transitions_constructed"],
        )
        self.assertEqual(len(transitions), stats["expert_transitions_accepted"])
        self.assertLessEqual(int(env.used.max()), 5)
        for transition in transitions:
            if transition.action < env.n:
                self.assertGreater(float(transition.state[transition.action]), 0.0)
                remaining = float(transition.state[env.n + transition.action])
                previous = float(transition.state[2 * env.n + transition.action])
                self.assertTrue(
                    remaining > 0.0 or previous == 1.0,
                    "Expert satellite action must be a free target or a legal stay",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
