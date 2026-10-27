import ast
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import Dueling_ddqn as proposed
import Dueling_ddqn_no_mshbo as dueling
import Standard_ddqn as standard
from run_artifacts import build_time_first_metadata
from test_connection_persistent_topk import synthetic_env


MODULES = (proposed, dueling, standard)
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def cli_option(module, option):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and node.args[0].value == option:
            matches.append(node)
    if len(matches) != 1:
        raise AssertionError(f"Expected one {option} declaration in {module.__name__}")
    return matches[0]


class HandoverPenaltyCliTests(unittest.TestCase):
    def test_all_cli_declarations_are_float_with_default_three(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                node = cli_option(module, "--lambda_handover")
                keywords = {item.arg: item.value for item in node.keywords}
                self.assertEqual(ast.literal_eval(keywords["default"]), 3.0)
                self.assertIsInstance(keywords["type"], ast.Name)
                self.assertEqual(keywords["type"].id, "float")

    def test_default_and_override_change_only_handover_component(self):
        for module in MODULES:
            for penalty in (3.0, 1.0):
                with self.subTest(module=module.__name__, penalty=penalty):
                    env = synthetic_env(module, 2, 1, 4, [1, 1])
                    env.lambda_h = penalty

                    env.step_user(0, 0, 0)
                    env.step_user(0, 1, 0)
                    stay = env.last_reward_components[0]
                    self.assertEqual(stay["handover"], 0.0)
                    self.assertEqual(stay["stay"], 0.05)
                    self.assertEqual(stay["throughput"], 2.0)

                    env.step_user(0, 2, 1)
                    handover = env.last_reward_components[0]
                    self.assertEqual(handover["handover"], -penalty)
                    self.assertEqual(handover["stay"], 0.0)
                    self.assertEqual(handover["throughput"], 2.0)

                    env.step_user(0, 3, env.block_action)
                    blocked = env.last_reward_components[0]
                    self.assertEqual(blocked["blocking"], -12.0)
                    self.assertEqual(blocked["throughput"], 0.0)

    def test_training_and_proposed_expert_use_environment_weight(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                training_source = inspect.getsource(module.train)
                self.assertIn("lambda_h=args.lambda_handover", training_source)
        expert_source = inspect.getsource(proposed.HandoverEnv.demo_rollouts_from_csv)
        self.assertIn("self.step_user", expert_source)

    def test_metadata_records_actual_override(self):
        args = SimpleNamespace(
            input=str(PROJECT_ROOT / "input.txt"),
            input_angle=str(PROJECT_ROOT / "input_angle.txt"),
            mshbo_csv=None,
            seed=42,
            episodes=1,
            lr=1e-3,
            batch_size=2,
            gamma=0.99,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay=10,
            target_update=2,
            replay_capacity=100,
            pretrain_updates=0,
            topk=3,
            sticky_prob=0.2,
            lookahead=5,
            lambda_handover=1.0,
            lambda_block=12.0,
            lambda_throughput=2.0,
            stay_bonus=0.05,
            lookahead_penalty=0.1,
        )
        env = synthetic_env(proposed, 2, 1, 2, [1, 1])
        metadata = build_time_first_metadata(
            model_variant="test",
            args=args,
            env=env,
            paths={
                "output_dir": "unused",
                "checkpoint_path": "unused.pt",
                "policy_path": "unused.csv",
                "metadata_path": "unused.json",
                "training_log_path": "unused-log.csv",
            },
            training_order_seed=42,
            demonstration_order_seed=42,
            evaluation_order_seed=42,
            demonstration_enabled=False,
            actual_pretraining_updates=0,
            best_episode=1,
            best_episode_return=0.0,
            counters={},
            source_path=str(PROJECT_ROOT / "rl" / "Dueling_ddqn.py"),
            rate_normalization_max_mbps=proposed.RATE_NORMALIZATION_MAX_MBPS,
            action_to_rate_id=np.arange(env.n),
            started_at_utc="test",
            device="cpu",
        )
        self.assertEqual(metadata["reward_weights"]["handover"], 1.0)
        self.assertEqual(metadata["reward_weights"]["blocking"], 12.0)
        self.assertEqual(metadata["reward_weights"]["throughput"], 2.0)
        self.assertEqual(metadata["reward_weights"]["stay_bonus"], 0.05)
        self.assertEqual(metadata["cli_arguments"]["lambda_handover"], 1.0)


if __name__ == "__main__":
    unittest.main()
