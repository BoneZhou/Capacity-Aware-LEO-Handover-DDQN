import inspect
import random
import unittest
from collections import deque

import numpy as np
import torch
import torch.optim as optim

import Dueling_ddqn as expert_dueling
import Dueling_ddqn_no_mshbo as dueling
import Standard_ddqn as standard


class ReferenceReplay:
    """The pre-optimization deque implementation used as the oracle."""

    def __init__(self, transition_type, cap):
        self.transition_type = transition_type
        self.buf = deque(maxlen=cap)

    def push(self, *args):
        self.buf.append(self.transition_type(*args))

    def __getitem__(self, index):
        return self.buf[index]

    def sample(self, batch):
        indexes = np.random.choice(len(self.buf), batch, replace=False)
        items = [self.buf[index] for index in indexes]
        return self.transition_type(*zip(*items))

    def __len__(self):
        return len(self.buf)


def transition_payload(index, state_dim=6, action_dim=4):
    state = np.arange(state_dim, dtype=np.float32) + np.float32(index * 10)
    next_state = state + np.float32(0.5)
    mask = np.zeros(action_dim, dtype=np.float32)
    mask[index % action_dim] = 1.0
    action = index % action_dim
    reward = float(index) / 7.0
    done = float(index % 5 == 0)
    return state, action, next_state, reward, done, mask


def assert_transition_equal(testcase, actual, expected):
    np.testing.assert_array_equal(actual.state, expected.state)
    testcase.assertEqual(actual.action, expected.action)
    np.testing.assert_array_equal(actual.next_state, expected.next_state)
    testcase.assertEqual(actual.reward, expected.reward)
    testcase.assertEqual(actual.done, expected.done)
    np.testing.assert_array_equal(actual.mask, expected.mask)


def assert_batch_equal(testcase, actual, expected):
    testcase.assertEqual(len(actual.action), len(expected.action))
    for index in range(len(actual.action)):
        np.testing.assert_array_equal(actual.state[index], expected.state[index])
        testcase.assertEqual(actual.action[index], expected.action[index])
        np.testing.assert_array_equal(actual.next_state[index], expected.next_state[index])
        testcase.assertEqual(actual.reward[index], expected.reward[index])
        testcase.assertEqual(actual.done[index], expected.done[index])
        np.testing.assert_array_equal(actual.mask[index], expected.mask[index])


def rng_states_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def synthetic_env(seed=42):
    env = dueling.HandoverEnv.__new__(dueling.HandoverEnv)
    env.n = 3
    env.m = 4
    env.Tb = 0
    env.Te = 20
    env.T = 20
    env.cap = np.array([2, 2, 2], dtype=int)
    env.V = np.ones((env.n, env.T), dtype=np.uint8)
    env.V[2, 7:10] = 0
    env.rate_norm = np.vstack(
        [
            np.linspace(0.2, 0.8, env.T, dtype=np.float32),
            np.linspace(0.8, 0.2, env.T, dtype=np.float32),
            np.full(env.T, 0.5, dtype=np.float32),
        ]
    )
    env.block_action = env.n
    env.n_actions = env.n + 1
    env.state_dim = env.n * 5 + 1
    env.lambda_h = 3.0
    env.lambda_b = 12.0
    env.lambda_t = 2.0
    env.stay_bonus = 0.05
    env.lookahead = 5
    env.lookahead_penalty = 0.1
    env.rng = random.Random(seed)
    env.order_rng = np.random.default_rng(seed)
    env.reset_multi_user_episode()
    return env


def encode_transition(transition):
    return (
        transition.state.tobytes(),
        int(transition.action),
        transition.next_state.tobytes(),
        float(transition.reward),
        float(transition.done),
        transition.mask.tobytes(),
    )


def run_training_smoke(replay_factory, seed=31):
    dueling.seed_everything(seed)
    env = synthetic_env(seed)
    q = dueling.QNet(env.state_dim, env.n_actions)
    target = dueling.QNet(env.state_dim, env.n_actions)
    target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=1e-3)
    replay = replay_factory(64)
    pending = [None for _ in range(env.m)]

    batch_size = 8
    gamma = 0.99
    target_update = 5
    global_slot_step = 0
    decisions = 0
    transitions = 0
    updates = 0
    target_syncs = 0
    trace = []
    choice_trace = []

    original_choice = np.random.choice

    def tracked_choice(*args, **kwargs):
        result = original_choice(*args, **kwargs)
        choice_trace.append(np.asarray(result).copy().tolist())
        return result

    np.random.choice = tracked_choice
    try:
        for t in range(env.T):
            epsilon = dueling.epsilon_at_slot(1.0, 0.05, 30000, global_slot_step)
            order = env.next_user_order()
            for raw_user in order:
                user = int(raw_user)
                state = env.make_state(user, t)
                legal_mask = env.legal_mask(user, t)
                selection_mask = dueling.topk_mask(
                    legal_mask,
                    env.used,
                    env.cap,
                    t,
                    int(env.last_sat[user]),
                    env.block_action,
                    3,
                )

                if pending[user] is not None:
                    previous = pending[user]
                    replay.push(
                        previous.state,
                        previous.action,
                        state.copy(),
                        previous.reward,
                        0.0,
                        selection_mask.copy(),
                    )
                    transitions += 1

                if random.random() < epsilon:
                    action = int(np.random.choice(np.where(selection_mask > 0)[0]))
                else:
                    with torch.no_grad():
                        q_values = q(torch.tensor(state, dtype=torch.float32))
                    action = dueling.masked_argmax(q_values, selection_mask)

                sticky_sat = int(env.last_effective_sat[user])
                if (
                    sticky_sat != -1
                    and random.random() < 0.2
                    and legal_mask[sticky_sat] > 0
                ):
                    action = sticky_sat

                reward, blocked = env.step_user(user, t, action)
                pending[user] = dueling.Pending(state.copy(), action, reward)
                decisions += 1
                trace.append(
                    (
                        t,
                        tuple(int(value) for value in order),
                        user,
                        state.tobytes(),
                        legal_mask.tobytes(),
                        selection_mask.tobytes(),
                        action,
                        reward,
                        blocked,
                        env.used[:, t].tobytes(),
                        tuple(int(value) for value in env.last_sat),
                        tuple(item is not None for item in pending),
                    )
                )

            if dueling.ddqn_update(
                q, target, optimizer, replay, batch_size, gamma, torch.device("cpu")
            ):
                updates += 1
            global_slot_step += 1
            if global_slot_step % target_update == 0:
                target.load_state_dict(q.state_dict())
                target_syncs += 1

        zero_state = np.zeros(env.state_dim, dtype=np.float32)
        terminal_mask = np.ones(env.n_actions, dtype=np.float32)
        for user in range(env.m):
            previous = pending[user]
            replay.push(
                previous.state,
                previous.action,
                zero_state.copy(),
                previous.reward,
                1.0,
                terminal_mask.copy(),
            )
            transitions += 1
    finally:
        np.random.choice = original_choice

    with torch.no_grad():
        probe_q = q(torch.tensor(env.make_state(0, env.T - 1), dtype=torch.float32))

    return {
        "trace": trace,
        "choice_trace": choice_trace,
        "replay": [encode_transition(replay[index]) for index in range(len(replay))],
        "weights": {
            name: value.detach().cpu().numpy().tobytes()
            for name, value in q.state_dict().items()
        },
        "probe_q": probe_q.detach().cpu().numpy().tobytes(),
        "decisions": decisions,
        "transitions": transitions,
        "updates": updates,
        "target_syncs": target_syncs,
        "global_slots": global_slot_step,
        "next_epsilon": dueling.epsilon_at_slot(1.0, 0.05, 30000, global_slot_step),
    }


class ReplayRingBufferTests(unittest.TestCase):
    def make_pair(self, capacity):
        return (
            ReferenceReplay(dueling.Transition, capacity),
            dueling.Replay(capacity),
        )

    def assert_logical_contents_equal(self, reference, ring):
        self.assertEqual(len(ring), len(reference))
        for index in range(len(reference)):
            assert_transition_equal(self, ring[index], reference[index])
        if len(reference):
            assert_transition_equal(self, ring[0], reference[0])
            assert_transition_equal(self, ring[-1], reference[-1])

    def test_A_before_wraparound_matches_deque(self):
        reference, ring = self.make_pair(7)
        for index in range(5):
            payload = transition_payload(index)
            reference.push(*payload)
            ring.push(*payload)
            self.assert_logical_contents_equal(reference, ring)
        with self.assertRaises(IndexError):
            _ = ring[len(ring)]

    def test_B_capacity_and_repeated_wraparound_match_deque_after_every_push(self):
        reference, ring = self.make_pair(7)
        for index in range(50):
            payload = transition_payload(index)
            reference.push(*payload)
            ring.push(*payload)
            self.assert_logical_contents_equal(reference, ring)
        self.assertEqual(ring[0].action, reference[0].action)
        self.assertEqual(ring[-1].reward, reference[-1].reward)

    def test_C_deterministic_sample_and_rng_state_equivalence(self):
        for pushes in (5, 7, 50):
            with self.subTest(pushes=pushes):
                reference, ring = self.make_pair(7)
                for index in range(pushes):
                    payload = transition_payload(index)
                    reference.push(*payload)
                    ring.push(*payload)

                batch_size = min(4, len(reference))
                np.random.seed(12345)
                initial_state = np.random.get_state()

                expected_indexes = np.random.choice(
                    len(reference), batch_size, replace=False
                )
                expected_items = [reference[index] for index in expected_indexes]
                expected_batch = dueling.Transition(*zip(*expected_items))
                expected_final_state = np.random.get_state()

                np.random.set_state(initial_state)
                actual_indexes = []
                original_choice = np.random.choice

                def recording_choice(*args, **kwargs):
                    result = original_choice(*args, **kwargs)
                    actual_indexes.append(np.asarray(result).copy())
                    return result

                np.random.choice = recording_choice
                try:
                    actual_batch = ring.sample(batch_size)
                finally:
                    np.random.choice = original_choice
                actual_final_state = np.random.get_state()

                self.assertEqual(len(actual_indexes), 1)
                np.testing.assert_array_equal(actual_indexes[0], expected_indexes)
                assert_batch_equal(self, actual_batch, expected_batch)
                self.assertEqual(
                    list(actual_batch.reward),
                    [reference[index].reward for index in expected_indexes],
                )
                self.assertTrue(
                    rng_states_equal(actual_final_state, expected_final_state)
                )

    def test_D_payload_integrity_before_and_after_wrap(self):
        reference, ring = self.make_pair(9)
        for index in range(31):
            payload = transition_payload(index, state_dim=251, action_dim=51)
            reference.push(*payload)
            ring.push(*payload)
        self.assert_logical_contents_equal(reference, ring)

        np.random.seed(9)
        initial_state = np.random.get_state()
        expected = reference.sample(8)
        np.random.set_state(initial_state)
        actual = ring.sample(8)
        assert_batch_equal(self, actual, expected)

    def test_E_replay_warmup_and_update_opportunities_are_unchanged(self):
        batch_size = 256
        users = 25
        slots = 3599
        reference, ring = self.make_pair(200000)
        first_ready_reference = None
        first_ready_ring = None
        update_opportunities_reference = 0
        update_opportunities_ring = 0

        for t in range(slots):
            if t > 0:
                for user in range(users):
                    payload = transition_payload(t * users + user)
                    reference.push(*payload)
                    ring.push(*payload)
            if len(reference) >= batch_size:
                first_ready_reference = (
                    t if first_ready_reference is None else first_ready_reference
                )
                update_opportunities_reference += 1
            if len(ring) >= batch_size:
                first_ready_ring = t if first_ready_ring is None else first_ready_ring
                update_opportunities_ring += 1

        for user in range(users):
            payload = transition_payload(slots * users + user)
            reference.push(*payload)
            ring.push(*payload)

        self.assertEqual(first_ready_reference, 11)
        self.assertEqual(first_ready_ring, first_ready_reference)
        self.assertEqual(update_opportunities_ring, update_opportunities_reference)
        self.assertEqual(update_opportunities_ring, 3588)
        self.assertEqual(len(ring), users * slots)
        self.assertEqual(len(ring), len(reference))

    def test_F_short_fixed_seed_training_path_matches_deque(self):
        reference_result = run_training_smoke(
            lambda capacity: ReferenceReplay(dueling.Transition, capacity)
        )
        ring_result = run_training_smoke(dueling.Replay)
        self.assertEqual(ring_result, reference_result)
        self.assertEqual(ring_result["decisions"], 80)
        self.assertEqual(ring_result["transitions"], 80)
        self.assertEqual(ring_result["updates"], 18)
        self.assertEqual(ring_result["target_syncs"], 4)
        self.assertEqual(ring_result["global_slots"], 20)

    def test_G_cross_variant_replay_implementation_and_expert_preload_parity(self):
        self.assertEqual(
            inspect.getsource(expert_dueling.Replay),
            inspect.getsource(dueling.Replay),
        )
        self.assertEqual(
            inspect.getsource(standard.Replay),
            inspect.getsource(dueling.Replay),
        )

        reference = ReferenceReplay(expert_dueling.Transition, 11)
        expert_ring = expert_dueling.Replay(11)
        for index in range(37):
            payload = transition_payload(index)
            reference.push(*payload)
            expert_ring.push(*payload)
        self.assert_logical_contents_equal(reference, expert_ring)

    @unittest.skip(
        "Replay benchmark deferred while production CUDA training may be active"
    )
    def test_H_replay_sampling_benchmark(self):
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
