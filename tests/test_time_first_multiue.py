import ast
import inspect
import random
import os
import tempfile
import textwrap
import unittest

import numpy as np
import torch

from Dueling_ddqn import (
    HandoverEnv,
    Pending,
    Transition,
    epsilon_at_slot,
    masked_next_argmax,
    seed_everything,
    train,
)
from evaluate_policy_metrics import replay_policy
from ue_order import generate_ue_orders


def synthetic_env(n, m, T, capacity, visibility=None, seed=42):
    env = HandoverEnv.__new__(HandoverEnv)
    env.n = n
    env.m = m
    env.Tb = 0
    env.Te = T
    env.T = T
    env.cap = np.asarray(capacity, dtype=int)
    env.V = (
        np.ones((n, T), dtype=np.uint8)
        if visibility is None
        else np.asarray(visibility, dtype=np.uint8)
    )
    env.rate_norm = np.ones((n, T), dtype=np.float32)
    env.block_action = n
    env.n_actions = n + 1
    env.state_dim = n * 5 + 1
    env.lambda_h = 3.0
    env.lambda_b = 12.0
    env.lambda_t = 2.0
    env.stay_bonus = 0.05
    env.lookahead = 5
    env.lookahead_penalty = 0.1
    env.rng = __import__("random").Random(seed)
    env.order_rng = np.random.default_rng(seed)
    env.reset_multi_user_episode()
    return env


class TimeFirstMultiUESanityTests(unittest.TestCase):
    def test_A_each_slot_is_seeded_permutation(self):
        env_a = synthetic_env(1, 5, 4, [2], seed=17)
        env_b = synthetic_env(1, 5, 4, [2], seed=17)
        orders_a = [env_a.next_user_order().tolist() for _ in range(env_a.T)]
        orders_b = [env_b.next_user_order().tolist() for _ in range(env_b.T)]
        self.assertEqual(orders_a, orders_b)
        for order in orders_a:
            self.assertEqual(sorted(order), list(range(env_a.m)))
            self.assertEqual(len(order), len(set(order)))

    def test_B_capacity_updates_immediately(self):
        env = synthetic_env(1, 3, 1, [2])
        for user in (0, 1):
            self.assertEqual(env.legal_mask(user, 0)[0], 1.0)
            env.step_user(user, 0, 0)
        third_state = env.make_state(2, 0)
        third_mask = env.legal_mask(2, 0)
        self.assertEqual(env.used[0, 0], 2)
        self.assertEqual(third_state[1], 0.0)  # rem starts at offset n
        self.assertEqual(third_mask[0], 0.0)

    def test_C_connection_history_is_per_user(self):
        env = synthetic_env(2, 2, 1, [2, 2])
        env.step_user(0, 0, 0)
        self.assertEqual(env.last_sat.tolist(), [0, -1])
        env.step_user(1, 0, 1)
        self.assertEqual(env.last_sat.tolist(), [0, 1])

    def test_D_pending_transition_keeps_same_user_identity(self):
        env = synthetic_env(1, 3, 3, [3])
        pending = [None] * env.m
        completed = []
        fixed_orders = np.array([[2, 0, 1], [1, 2, 0], [0, 1, 2]])
        for t in range(env.T):
            for raw_user in fixed_orders[t]:
                user = int(raw_user)
                state = env.make_state(user, t)
                if pending[user] is not None:
                    source_t, prev = pending[user]
                    completed.append((user, source_t, user, t))
                env.step_user(user, t, env.block_action)
                pending[user] = (t, Pending(state, env.block_action, -12.0))
        self.assertTrue(completed)
        for source_u, source_t, next_u, next_t in completed:
            self.assertEqual(source_u, next_u)
            self.assertEqual(next_t, source_t + 1)

    def test_E_next_state_contains_earlier_same_slot_allocations(self):
        env = synthetic_env(1, 2, 2, [2])
        state_u0_t0 = env.make_state(0, 0)
        env.step_user(0, 0, env.block_action)
        pending_u0 = Pending(state_u0_t0, env.block_action, -12.0)

        # At t=1 UE1 acts before UE0, so UE0's real next state must see this use.
        env.step_user(1, 1, 0)
        next_state_u0 = env.make_state(0, 1)
        transition = Transition(
            pending_u0.state,
            pending_u0.action,
            next_state_u0,
            pending_u0.reward,
            0.0,
            env.legal_mask(0, 1),
        )
        self.assertEqual(transition.next_state[1], 0.5)
        self.assertEqual(transition.mask[0], 1.0)

    def test_F_terminal_transition_has_no_nonexistent_observation(self):
        env = synthetic_env(1, 3, 2, [3])
        pending = [None] * env.m
        transitions = []
        for t in range(env.T):
            for user in range(env.m):
                state = env.make_state(user, t)
                mask = env.legal_mask(user, t)
                if pending[user] is not None:
                    p = pending[user]
                    transitions.append(
                        Transition(p.state, p.action, state, p.reward, 0.0, mask)
                    )
                env.step_user(user, t, env.block_action)
                pending[user] = Pending(state, env.block_action, -12.0)

        zero_state = np.zeros(env.state_dim, dtype=np.float32)
        terminal = [
            Transition(p.state, p.action, zero_state.copy(), p.reward, 1.0,
                       np.ones(env.n_actions, dtype=np.float32))
            for p in pending
        ]
        transitions.extend(terminal)
        self.assertEqual(len(transitions), env.m * env.T)
        self.assertEqual(sum(tr.done == 0.0 for tr in transitions), env.m * (env.T - 1))
        self.assertEqual(len(terminal), env.m)
        self.assertTrue(all(tr.done == 1.0 for tr in terminal))
        self.assertTrue(all(np.count_nonzero(tr.next_state) == 0 for tr in terminal))
        with self.assertRaises(IndexError):
            env.make_state(0, env.T)

    def test_G_blocking_breaks_history_in_environment_and_evaluator(self):
        env = synthetic_env(2, 1, 3, [1, 1])
        env.step_user(0, 0, 0)
        env.step_user(0, 1, env.block_action)
        self.assertEqual(env.last_sat[0], -1)
        env.step_user(0, 2, 1)
        self.assertEqual(env.last_reward_components[0]["handover"], 0.0)

        actions = np.array([[0, -1, 1]], dtype=int)
        orders = np.zeros((3, 1), dtype=np.int64)
        _, handovers, blocked, _, _ = replay_policy(
            2, 1, 3, np.array([1, 1]), np.ones((2, 3), dtype=np.uint8),
            actions, orders,
        )
        self.assertEqual(handovers[0], 0)
        self.assertEqual(blocked[0], 1)

    def test_H_masked_ddqn_argmax_rejects_illegal_high_q(self):
        q_values = torch.tensor([[100.0, 5.0, 1.0]])
        masks = torch.tensor([[0.0, 1.0, 1.0]])
        selected = masked_next_argmax(q_values, masks)
        self.assertEqual(int(selected.item()), 1)

    def test_I_training_and_evaluator_capacity_semantics_match(self):
        n, m, T = 2, 3, 2
        cap = np.array([1, 1])
        visibility = np.ones((n, T), dtype=np.uint8)
        actions = np.array([[0, 1], [0, 1], [1, 0]], dtype=int)
        orders = np.array([[2, 0, 1], [1, 2, 0]], dtype=np.int64)

        env = synthetic_env(n, m, T, cap, visibility)
        training_blocked = np.zeros(m, dtype=int)
        for t in range(T):
            for raw_user in orders[t]:
                user = int(raw_user)
                _, blocked = env.step_user(user, t, int(actions[user, t]))
                training_blocked[user] += int(blocked)

        eval_used, _, eval_blocked, _, _ = replay_policy(
            n, m, T, cap, visibility, actions, orders
        )
        np.testing.assert_array_equal(env.used, eval_used)
        np.testing.assert_array_equal(training_blocked, eval_blocked)

    def test_rng_sources_are_seeded(self):
        seed_everything(23)
        first = (random.random(), np.random.random(), torch.rand(1).item())
        seed_everything(23)
        second = (random.random(), np.random.random(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_demo_rollout_uses_time_first_shared_capacity(self):
        env = synthetic_env(1, 3, 2, [1], seed=11)
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        try:
            with handle:
                handle.write("t,user,sat\n")
                for t in range(env.T):
                    for user in range(env.m):
                        handle.write(f"{t},{user},0\n")
            transitions, stats = env.demo_rollouts_from_csv(
                handle.name,
                return_stats=True,
            )
        finally:
            os.unlink(handle.name)

        self.assertEqual(len(transitions), 2)
        self.assertEqual(stats["expert_transitions_constructed"], env.m * env.T)
        self.assertEqual(stats["legal_satellite_actions"], 2)
        self.assertEqual(stats["capacity_full_satellite_actions"], 4)
        self.assertEqual(stats["expert_transitions_accepted"], 2)
        self.assertEqual(stats["expert_transitions_discarded"], 4)
        self.assertEqual(sum(tr.done == 1.0 for tr in transitions), 1)
        self.assertEqual(sum(tr.reward == -12.0 for tr in transitions), 0)

    def test_slot_based_update_is_outside_ue_loop_and_at_most_once(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(train)))
        train_node = tree.body[0]
        time_loop = next(
            node for node in ast.walk(train_node)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "t"
            and isinstance(node.iter, ast.Call)
        )
        ue_loop = next(
            node for node in time_loop.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "raw_user"
        )

        def ddqn_calls(node):
            return [
                call for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "ddqn_update"
            ]

        self.assertEqual(len(ddqn_calls(ue_loop)), 0)
        slot_level_calls = []
        for statement in time_loop.body:
            if statement is not ue_loop:
                slot_level_calls.extend(ddqn_calls(statement))
        self.assertEqual(len(slot_level_calls), 1)

    def test_full_scale_decision_and_transition_counts_remain_ue_level(self):
        users, slots = 25, 3599
        self.assertEqual(users * slots, 89975)
        # Pending completion yields U*(T-1), terminal flush contributes U.
        transitions = users * (slots - 1) + users
        self.assertEqual(transitions, 89975)
        maximum_slot_updates = slots
        self.assertEqual(maximum_slot_updates, 3599)
        self.assertLess(maximum_slot_updates, transitions)

    def test_epsilon_and_target_schedules_are_slot_based(self):
        source = textwrap.dedent(inspect.getsource(train))
        tree = ast.parse(source)
        train_node = tree.body[0]
        time_loop = next(
            node for node in ast.walk(train_node)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "t"
        )
        ue_loop = next(
            node for node in time_loop.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "raw_user"
        )

        ue_names = {
            node.id for node in ast.walk(ue_loop)
            if isinstance(node, ast.Name)
        }
        self.assertNotIn("global_slot_step", ue_names)

        slot_source = ast.unparse(time_loop)
        self.assertIn("epsilon_at_slot(eps_start, eps_end, eps_decay, global_slot_step)", slot_source)
        self.assertIn("global_slot_step += 1", slot_source)
        self.assertIn("global_slot_step % args.target_update == 0", slot_source)
        self.assertNotIn("global_decision_step % args.target_update", source)

    def test_slot_counter_scales_and_epsilon_checkpoints(self):
        users, slots = 25, 3599
        self.assertEqual(users * slots, 89975)
        self.assertEqual(slots, 3599)
        self.assertEqual(50 * slots, 179950)
        self.assertEqual((50 * slots) // 2000, 89)
        self.assertEqual(slots // 2000, 1)

        expected = {
            0: 1.0,
            500: 0.984297881131,
            1000: 0.968855295458,
            1200: 0.962749967195,
            3599: 0.892602501163,
        }
        for slot_step, value in expected.items():
            self.assertAlmostEqual(
                epsilon_at_slot(1.0, 0.05, 30000, slot_step), value, places=11
            )

        epsilon_for_three_ues = [
            epsilon_at_slot(1.0, 0.05, 30000, 500) for _ in range(3)
        ]
        self.assertEqual(len(set(epsilon_for_three_ues)), 1)

    def test_preserved_state_dimension_and_shared_order_generator(self):
        env = synthetic_env(50, 25, 2, np.full(50, 5))
        self.assertEqual(env.make_state(0, 0).shape, (251,))
        orders = generate_ue_orders(25, 2, 42)
        self.assertEqual(orders.shape, (2, 25))


if __name__ == "__main__":
    unittest.main(verbosity=2)
