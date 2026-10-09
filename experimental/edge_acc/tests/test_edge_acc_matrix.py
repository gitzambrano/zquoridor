"""Verify resource limits and paired baseline order for the durable matrix."""
import unittest
from unittest.mock import patch
from tools import run_edge_acc_matrix as matrix


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.resource_policy = patch.dict(matrix.CONFIG, fixed_workers=False)
        self.resource_policy.start()
        self.addCleanup(self.resource_policy.stop)

    def test_user_fixed_concurrency_does_not_gate_on_ram(self):
        with patch.dict(matrix.CONFIG, fixed_workers=True):
            self.assertEqual(matrix.safe_workers({"mode": "h2h"}, 4096), 5)
            self.assertEqual(matrix.safe_workers({"mode": "external", "opponent": "claustrophobia"}, 4096), 6)
            self.assertEqual(matrix.safe_workers({"mode": "h2h"}, 0), 5)
            self.assertEqual(matrix.safe_workers({"mode": "external", "opponent": "claustrophobia"}, 0), 6)

    def test_cpu_and_worker_limits_reject_oversubscription(self):
        with patch.dict(matrix.CONFIG, h2h_workers=7):
            with self.assertRaises(RuntimeError):
                matrix.apply_limits()
        with patch.dict(matrix.CONFIG, claustrophobia_workers=7):
            with self.assertRaises(RuntimeError):
                matrix.apply_limits()

    def test_memory_reserve_reduces_parallel_games(self):
        self.assertEqual(matrix.safe_workers({"mode": "h2h"}, 9000), 1)
        self.assertEqual(matrix.safe_workers({"mode": "external", "opponent": "claustrophobia"}, 9000), 2)
        self.assertEqual(matrix.safe_workers({"mode": "h2h"}, 6000), 0)
        self.assertEqual(matrix.safe_workers({"mode": "h2h"}, 24000), 3)
        with patch.dict(matrix.CONFIG, h2h_workers=6):
            self.assertEqual(matrix.safe_workers({"mode": "h2h"}, 10000), 1)
            self.assertEqual(matrix.safe_workers({"mode": "h2h"}, 24000), 3)

    def test_explicit_modest_start_admits_one_worker_without_lowering_reserve(self):
        with patch.dict(matrix.CONFIG, modest_start=True):
            self.assertEqual(matrix.safe_workers({"mode": "h2h"}, 4096), 1)
            self.assertEqual(matrix.safe_workers({"mode": "h2h"}, 4095), 0)
            self.assertEqual(matrix.CONFIG["ram_reserve_mib"], 2048)
        self.assertEqual(matrix.safe_workers({"mode": "h2h"}, 4096), 0)

    def test_scheduler_advances_external_baselines_when_h2h_does_not_fit(self):
        tasks = matrix.task_plan()
        task = matrix.eligible_task(tasks, set(), 5000)
        self.assertEqual(task["name"], "titanium_504_off_0")
        self.assertEqual(matrix.ram_per_game(task), 2048)
        self.assertIsNone(matrix.eligible_task(tasks, set(), 4000))
        done = {task["name"]}
        self.assertEqual(matrix.eligible_task([t for t in tasks if t["name"] not in done], done, 5000)["name"], "titanium_504_delta_dense_only_0")

    def test_each_external_candidate_follows_its_paired_baseline(self):
        tasks = matrix.task_plan()
        names = [task["name"] for task in tasks]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 92)
        for task in tasks:
            if task["mode"] == "external" and task["variant"] != "off":
                baseline = f"{task['opponent']}_{task['network']}_off_{task['shard']}"
                self.assertLess(names.index(baseline), names.index(task["name"]))

    def test_child_forwards_shared_limits_and_native_shard(self):
        with patch.object(matrix, "apply_limits"), patch.object(matrix, "configure_campaign"), \
             patch.object(matrix.campaign, "run_h2h_3plus2", return_value={}) as run:
            matrix.run_child("h2h_858_v3_node_dense_bfs_2")
        run.assert_called_once_with("858", "v3_node_dense_bfs", 2, 4)


if __name__ == "__main__":
    unittest.main()
