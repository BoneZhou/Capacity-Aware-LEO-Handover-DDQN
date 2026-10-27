import ast
import csv
import inspect
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import Dueling_ddqn as proposed
import Dueling_ddqn_no_mshbo as no_mshbo
import Standard_ddqn as standard
from run_artifacts import prepare_run_paths
from ue_order import generate_ue_orders


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parents[1] / "DDQN_Training_Delivery_Clean" / "Dueling_DDQN"


def order_only_env(env_class, training_seed=17, demo_seed=19, evaluation_seed=23):
    env = env_class.__new__(env_class)
    env.m = 7
    env.training_order_seed = training_seed
    env.demonstration_order_seed = demo_seed
    env.evaluation_order_seed = evaluation_seed
    env.training_order_rng = np.random.default_rng(training_seed)
    env.demo_order_rng = np.random.default_rng(demo_seed)
    env.evaluation_order_rng = np.random.default_rng(evaluation_seed)
    env.order_rng = env.training_order_rng
    return env


def ast_node(path, node_type, name, parent_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = tree.body
    if parent_name is not None:
        parent = next(
            node for node in roots
            if isinstance(node, ast.ClassDef) and node.name == parent_name
        )
        roots = parent.body
    return next(node for node in roots if isinstance(node, node_type) and node.name == name)


class RNGIsolationTests(unittest.TestCase):
    def test_fixed_training_seed_is_reproducible_and_per_slot_is_permutation(self):
        first = order_only_env(proposed.HandoverEnv, training_seed=31)
        second = order_only_env(proposed.HandoverEnv, training_seed=31)
        orders_first = [first.next_training_user_order().tolist() for _ in range(8)]
        orders_second = [second.next_training_user_order().tolist() for _ in range(8)]
        self.assertEqual(orders_first, orders_second)
        for order in orders_first:
            self.assertEqual(sorted(order), list(range(first.m)))

    def test_three_variants_align_for_same_training_order_seed(self):
        environments = [
            order_only_env(module.HandoverEnv, training_seed=43)
            for module in (proposed, no_mshbo, standard)
        ]
        sequences = [
            [env.next_training_user_order().tolist() for _ in range(10)]
            for env in environments
        ]
        self.assertEqual(sequences[0], sequences[1])
        self.assertEqual(sequences[0], sequences[2])

    def test_demo_consumption_does_not_advance_training_stream(self):
        with_demo = order_only_env(proposed.HandoverEnv, training_seed=47, demo_seed=53)
        without_demo = order_only_env(proposed.HandoverEnv, training_seed=47, demo_seed=53)
        for _ in range(25):
            with_demo.next_demo_user_order()
        training_after_demo = [
            with_demo.next_training_user_order().tolist() for _ in range(12)
        ]
        untouched_training = [
            without_demo.next_training_user_order().tolist() for _ in range(12)
        ]
        self.assertEqual(training_after_demo, untouched_training)

    def test_evaluation_stream_is_independent_and_seeded(self):
        env = order_only_env(proposed.HandoverEnv)
        env.reset_evaluation_order_rng(61)
        first = [env.next_evaluation_user_order().tolist() for _ in range(6)]
        self.assertEqual(first, generate_ue_orders(env.m, 6, 61).tolist())
        env.reset_evaluation_order_rng(61)
        repeated = [env.next_evaluation_user_order().tolist() for _ in range(6)]
        env.reset_evaluation_order_rng(62)
        different = [env.next_evaluation_user_order().tolist() for _ in range(6)]
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        for order in different:
            self.assertEqual(sorted(order), list(range(env.m)))


class AlgorithmRegressionTests(unittest.TestCase):
    def test_three_variants_share_capacity_and_mask_rules(self):
        paths = [Path(module.__file__) for module in (proposed, no_mshbo, standard)]
        for parent, functions in {
            "HandoverEnv": (
                "reset_multi_user_episode",
                "_assert_capacity_consistency",
                "_prepare_slot",
                "make_state",
                "legal_mask",
                "step_user",
            ),
            None: ("masked_next_argmax", "topk_mask"),
        }.items():
            for function_name in functions:
                dumps = [
                    ast.dump(
                        ast_node(path, ast.FunctionDef, function_name, parent_name=parent),
                        include_attributes=False,
                    )
                    for path in paths
                ]
                self.assertEqual(dumps[0], dumps[1], function_name)
                self.assertEqual(dumps[0], dumps[2], function_name)
        for module in (proposed, no_mshbo, standard):
            update_source = inspect.getsource(module.ddqn_update)
            self.assertIn("a_star = masked_next_argmax(q(ns_b), m_b)", update_source)
            self.assertIn("tq_ns = tq(ns_b).gather(1, a_star).squeeze(1)", update_source)
            self.assertIn("target = r_b + (1.0 - d_b) * gamma * tq_ns", update_source)

    def test_export_uses_topk_after_full_legal_mask(self):
        for module in (proposed, no_mshbo, standard):
            source = ast.unparse(ast_node(Path(module.__file__), ast.FunctionDef, "export_policy"))
            self.assertIn("legal_mask = env.legal_mask(user, t)", source)
            self.assertIn("mask = topk_mask", source)

    def test_training_still_has_one_slot_level_update(self):
        for module in (proposed, no_mshbo, standard):
            train_node = ast_node(Path(module.__file__), ast.FunctionDef, "train")
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
            calls_in_ue = [
                node for node in ast.walk(ue_loop)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ddqn_update"
            ]
            calls_in_slot = [
                node for statement in time_loop.body if statement is not ue_loop
                for node in ast.walk(statement)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ddqn_update"
            ]
            self.assertEqual(calls_in_ue, [])
            self.assertEqual(len(calls_in_slot), 1)


class MetadataSmokeTests(unittest.TestCase):
    def _write_synthetic_inputs(self, root):
        input_path = root / "input.txt"
        rate_path = root / "input_angle.txt"
        input_path.write_text(
            "50 3 0 2\n" + "".join("0 2 2\n" for _ in range(50)),
            encoding="utf-8",
        )
        rate_rows = []
        for satellite in range(50):
            for t in range(2):
                rate = proposed.RATE_NORMALIZATION_MAX_MBPS if (satellite, t) == (0, 0) else 1000.0
                rate_rows.append(f"{satellite} {t} {rate:.6f}\n")
        rate_path.write_text("".join(rate_rows), encoding="utf-8")
        demo_path = root / "demo.csv"
        demo_path.write_text(
            "t,user,sat\n" + "".join(
                f"{t},{user},0\n" for t in range(2) for user in range(3)
            ),
            encoding="utf-8",
        )
        return input_path, rate_path, demo_path

    def _args(self, input_path, rate_path, output_dir, demo_path=None):
        return SimpleNamespace(
            input=str(input_path), input_angle=str(rate_path),
            mshbo_csv=str(demo_path) if demo_path else None,
            episodes=1, device="cpu", lr=1e-3, gamma=0.99,
            batch_size=2, replay_capacity=100,
            eps_start=0.2, eps_end=0.1, eps_decay=10,
            target_update=2, lambda_handover=3.0, lambda_block=12.0,
            lambda_throughput=2.0, stay_bonus=0.05,
            lookahead=1, lookahead_penalty=0.1,
            topk=3, sticky_prob=0.2, pretrain_updates=1,
            seed=71, training_order_seed=73, demo_order_seed=79,
            eval_order_seed=83, output_dir=str(output_dir),
            checkpoint_out=None, policy_out=None,
        )

    def test_all_variants_write_loadable_checkpoint_metadata_and_log(self):
        workspace_tmp = HERE.parents[1] / "tmp"
        workspace_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_tmp) as tmp:
            root = Path(tmp)
            input_path, rate_path, demo_path = self._write_synthetic_inputs(root)
            variants = [
                (proposed, "proposed", demo_path, True),
                (no_mshbo, "no_mshbo", None, False),
                (standard, "standard", None, False),
            ]
            for module, name, expert, expected_demo in variants:
                with self.subTest(variant=name):
                    args = self._args(input_path, rate_path, root / name, expert)
                    metadata = module.train(args)
                    metadata_path = Path(metadata["metadata_path"])
                    log_path = Path(metadata["training_log_path"])
                    checkpoint_path = Path(metadata["checkpoint_path"])
                    policy_path = Path(metadata["policy_export_path"])
                    self.assertTrue(metadata_path.is_file())
                    self.assertTrue(log_path.is_file())
                    self.assertTrue(checkpoint_path.is_file())
                    self.assertTrue(policy_path.is_file())

                    parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
                    self.assertEqual(parsed["best_episode"], 1)
                    self.assertEqual(
                        parsed["best_episode_return"], metadata["best_episode_return"]
                    )
                    self.assertEqual(parsed["training_order_seed"], 73)
                    self.assertEqual(parsed["evaluation_order_seed"], 83)
                    self.assertEqual(
                        parsed["mshbo_demonstration_enabled"], expected_demo
                    )
                    self.assertEqual(parsed["td_target_mask_type"], "next_state_top_k")
                    self.assertEqual(parsed["policy_export_mask_type"], "top_k")
                    self.assertEqual(parsed["capacity_accounting"], "connection_persistent")
                    if expected_demo:
                        expert_stats = parsed["expert_replay_stats"]
                        self.assertTrue(expert_stats["expert_data_complete"])
                        self.assertEqual(expert_stats["source_csv_rows"], 6)
                        self.assertEqual(expert_stats["in_window_rows"], 6)
                        self.assertEqual(expert_stats["legal_satellite_actions"], 4)
                        self.assertEqual(
                            expert_stats["capacity_full_satellite_actions"], 2
                        )
                        self.assertEqual(
                            expert_stats["expert_transitions_accepted"], 4
                        )
                        self.assertEqual(
                            expert_stats["expert_transitions_discarded"], 2
                        )
                        self.assertEqual(
                            expert_stats["expert_transitions_loaded_into_replay"], 4
                        )
                        self.assertEqual(
                            expert_stats["requested_pretraining_updates"], 1
                        )
                        self.assertEqual(
                            expert_stats["actual_pretraining_updates"], 1
                        )
                    else:
                        self.assertIsNone(parsed["expert_replay_stats"])

                    with log_path.open(newline="", encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(int(rows[0]["episode_index"]), 1)
                    self.assertEqual(rows[0]["best_checkpoint_updated"], "True")
                    self.assertAlmostEqual(
                        float(rows[0]["best_return_so_far"]),
                        parsed["best_episode_return"],
                    )

                    state_dict = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=False
                    )
                    reloaded = module.QNet(parsed["state_dimension"], parsed["action_dimension"])
                    reloaded.load_state_dict(state_dict, strict=True)

    def test_existing_artifact_is_never_overwritten(self):
        workspace_tmp = HERE.parents[1] / "tmp"
        workspace_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_tmp) as tmp:
            output_dir = Path(tmp) / "protected_run"
            output_dir.mkdir()
            (output_dir / "run_metadata.json").write_text("historical", encoding="utf-8")
            args = SimpleNamespace(
                output_dir=str(output_dir), seed=42,
                checkpoint_out=None, policy_out=None,
            )
            with self.assertRaises(FileExistsError):
                prepare_run_paths(args, "test", "checkpoint.pt", "policy.csv")
            self.assertEqual(
                (output_dir / "run_metadata.json").read_text(encoding="utf-8"),
                "historical",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

