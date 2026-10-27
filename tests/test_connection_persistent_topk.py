import csv
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import Dueling_ddqn as proposed
import Dueling_ddqn_no_mshbo as dueling
import Standard_ddqn as standard


MODULES = (proposed, dueling, standard)


def synthetic_env(module, n, m, slots, capacity, visibility=None, seed=42):
    env = module.HandoverEnv.__new__(module.HandoverEnv)
    env.n = n
    env.m = m
    env.Tb = 0
    env.Te = slots
    env.T = slots
    env.cap = np.asarray(capacity, dtype=int)
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
    env.lookahead = 0
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


def assert_consistent(testcase, env):
    expected = np.bincount(
        env.last_sat[env.last_sat >= 0], minlength=env.n
    )
    np.testing.assert_array_equal(env.occupancy, expected)
    testcase.assertTrue(np.all(env.occupancy >= 0))
    testcase.assertTrue(np.all(env.occupancy <= env.cap))
    np.testing.assert_array_equal(env.used[:, env._active_slot], env.occupancy)


class ConnectionPersistentCapacityTests(unittest.TestCase):
    def test_stay_is_not_double_counted_and_full_owner_is_protected(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                env = synthetic_env(module, 1, 2, 2, [1])
                env.step_user(0, 0, 0)
                env.make_state(1, 1)
                self.assertEqual(int(env.occupancy[0]), 1)
                self.assertEqual(float(env.legal_mask(0, 1)[0]), 1.0)
                self.assertEqual(float(env.legal_mask(1, 1)[0]), 0.0)

                _, rejected = env.step_user(1, 1, 0)
                self.assertTrue(rejected)
                self.assertEqual(env.last_sat.tolist(), [0, -1])
                env.step_user(0, 1, 0)
                self.assertEqual(int(env.occupancy[0]), 1)
                assert_consistent(self, env)

    def test_handover_block_and_failed_handover_update_capacity_correctly(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                env = synthetic_env(module, 2, 2, 3, [1, 1])
                env.step_user(0, 0, 0)
                env.step_user(1, 0, 1)
                env.make_state(0, 1)

                _, rejected = env.step_user(0, 1, 1)
                self.assertTrue(rejected)
                self.assertEqual(env.last_sat.tolist(), [0, 1])
                np.testing.assert_array_equal(env.occupancy, [1, 1])

                env.step_user(1, 1, env.block_action)
                np.testing.assert_array_equal(env.occupancy, [1, 0])
                env.step_user(0, 1, 1)
                self.assertEqual(env.last_sat.tolist(), [1, -1])
                np.testing.assert_array_equal(env.occupancy, [0, 1])
                self.assertLess(env.last_reward_components[0]["handover"], 0.0)

                env.step_user(0, 2, env.block_action)
                np.testing.assert_array_equal(env.occupancy, [0, 0])
                assert_consistent(self, env)

    def test_visibility_loss_releases_before_first_decision(self):
        visibility = [[1, 0], [1, 1]]
        for module in MODULES:
            with self.subTest(module=module.__name__):
                env = synthetic_env(module, 2, 1, 2, [1, 1], visibility)
                env.step_user(0, 0, 0)
                state = env.make_state(0, 1)
                self.assertEqual(int(env.last_sat[0]), -1)
                np.testing.assert_array_equal(env.occupancy, [0, 0])
                self.assertEqual(int(np.count_nonzero(state[4:6])), 0)
                assert_consistent(self, env)

    def test_shuffle_does_not_remove_valid_reservations(self):
        orders = ([0, 1], [1, 0], [0, 1])
        for module in MODULES:
            with self.subTest(module=module.__name__):
                env = synthetic_env(module, 2, 2, 3, [1, 1])
                initial = [0, 1]
                for t, order in enumerate(orders):
                    for user in order:
                        env.make_state(user, t)
                        env.step_user(user, t, initial[user])
                self.assertEqual(env.last_sat.tolist(), initial)
                np.testing.assert_array_equal(env.occupancy, [1, 1])
                assert_consistent(self, env)


class NextStateTopKTests(unittest.TestCase):
    def test_next_state_and_mask_use_same_ue_actual_next_turn(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                env = synthetic_env(module, 4, 2, 2, [1, 1, 1, 1])
                first_state = env.make_state(0, 0)
                env.step_user(0, 0, 3)
                env.step_user(1, 0, 0)
                pending = module.Pending(first_state.copy(), 3, 2.0)

                env.make_state(1, 1)
                env.step_user(1, 1, 0)
                next_state = env.make_state(0, 1)
                full_mask = env.legal_mask(0, 1)
                next_mask = module.topk_mask(
                    full_mask,
                    env.used,
                    env.cap,
                    1,
                    int(env.last_sat[0]),
                    env.block_action,
                    1,
                )
                transition = module.Transition(
                    pending.state,
                    pending.action,
                    next_state.copy(),
                    pending.reward,
                    0.0,
                    next_mask.copy(),
                )

                self.assertEqual(float(transition.next_state[2 * env.n + 3]), 1.0)
                self.assertEqual(float(transition.next_state[env.n + 0]), 0.0)
                self.assertEqual(float(transition.mask[3]), 1.0)
                self.assertEqual(float(transition.mask[env.block_action]), 1.0)
                self.assertEqual(int(np.count_nonzero(transition.mask[: env.n])), 1)

                q_values = torch.tensor([[100.0, 20.0, 10.0, 5.0, 1.0]])
                selected = module.masked_next_argmax(
                    q_values, torch.tensor(transition.mask).unsqueeze(0)
                )
                self.assertEqual(int(selected.item()), 3)

    def test_online_pending_push_stores_selection_mask(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module.train)
                push_start = source.index("buf.push(\n                        prev.state")
                push_end = source.index("completed_transitions += 1", push_start)
                self.assertIn("selection_mask.copy()", source[push_start:push_end])
                self.assertLess(
                    source.index("selection_mask = topk_mask"),
                    source.index("if pending[user] is not None"),
                )

    def test_three_variants_share_environment_and_topk_rules(self):
        method_names = (
            "reset_multi_user_episode",
            "_assert_capacity_consistency",
            "_prepare_slot",
            "make_state",
            "legal_mask",
            "step_user",
        )
        for name in method_names:
            sources = [inspect.getsource(getattr(module.HandoverEnv, name)) for module in MODULES]
            self.assertEqual(sources[0], sources[1], name)
            self.assertEqual(sources[0], sources[2], name)
        topk_sources = [inspect.getsource(module.topk_mask) for module in MODULES]
        self.assertEqual(topk_sources[0], topk_sources[1])
        self.assertEqual(topk_sources[0], topk_sources[2])


class ExpertReplayPersistentTests(unittest.TestCase):
    def test_illegal_expert_actions_are_filtered_and_targets_use_topk(self):
        env = synthetic_env(proposed, 2, 2, 2, [1, 1])
        rows = [(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / "expert.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["t", "user", "sat"])
                writer.writerows(rows)
            transitions, stats = env.demo_rollouts_from_csv(
                str(path), topk=1, return_stats=True
            )

        self.assertGreater(stats["expert_transitions_discarded"], 0)
        self.assertEqual(
            stats["expert_transitions_accepted"]
            + stats["expert_transitions_discarded"],
            env.m * env.T,
        )
        for transition in transitions:
            action = int(transition.action)
            if action < env.n:
                self.assertGreater(float(transition.state[action]), 0.0)
                remaining = float(transition.state[env.n + action])
                previous = float(transition.state[2 * env.n + action])
                self.assertTrue(remaining > 0.0 or previous == 1.0)
            if not transition.done:
                self.assertEqual(float(transition.mask[env.block_action]), 1.0)
                self.assertLessEqual(int(np.count_nonzero(transition.mask[: env.n])), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
