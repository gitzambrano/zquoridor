"""Check sequential campaign screening and restart guards."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_edge_acc_queue as queue


def h2h(center=56, normal=58, upper_center=62, upper_normal=65):
    return {"score_pct": (center * 266 + normal * 134) / 400,
            "center": {"complete_pairs": 133, "failed_games": 0, "score_pct": center,
                       "paired_bootstrap_95": {"score_high_pct": upper_center}},
            "normal": {"complete_pairs": 67, "failed_games": 0, "score_pct": normal,
                       "paired_bootstrap_95": {"score_high_pct": upper_normal}}}


def external(center_upper=6, normal_upper=8):
    return {"paired_delta_candidate_minus_off": {
        "center": {"complete_paired_openings": 100, "paired_bootstrap_95": [-1, center_upper]},
        "normal": {"complete_paired_openings": 50, "paired_bootstrap_95": [-2, normal_upper]}}}


class QueueTests(unittest.TestCase):
    def test_campaign_wrapper_forwards_arguments_with_queue_concurrency(self):
        arguments = ["external", "--network", "504", "--variant", "off",
                     "--opponent", "claustrophobia", "--participants", "off"]
        command = queue.campaign_wrapper_command(arguments)
        self.assertEqual(command[-len(arguments):], arguments)
        self.assertIn("c.CONFIG.update(h2h_workers=12, gpu_workers=6)", command[3])
        help_result = subprocess.run([*command, "--help"], capture_output=True,
                                     text=True, timeout=15)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("external-3plus2", help_result.stdout)

    def test_claustrophobia_memory_gate_waits_for_reserve_plus_worker_budget(self):
        self.assertEqual(queue.CONFIG["resource_policy"]["campaign_h2h_workers"], 12)
        self.assertEqual(queue.CONFIG["resource_policy"]["campaign_claustrophobia_workers"], 6)
        self.assertEqual(queue.required_cla_gpu_mib(), 2048 + 6 * 256)
        self.assertEqual(queue.parse_gpu_free_mib("3584\n"), 3584)
        fake_queue = object.__new__(queue.Queue)
        fake_queue.status = {"current": {"resource_wait": {"free_mib": 3000}}}
        with patch.object(queue, "query_gpu_free_mib", side_effect=[3000, 3584]), \
                patch.object(fake_queue, "save") as save, \
                patch.object(queue.time, "sleep") as sleep:
            fake_queue.wait_for_cla_gpu("cla-pre300")
        sleep.assert_called_once_with(30)
        self.assertNotIn("resource_wait", fake_queue.status["current"])
        self.assertEqual(save.call_count, 2)

    def test_gpu_gate_selects_external_cla_jobs_but_not_reports_or_titanium(self):
        self.assertTrue(queue.claustrophobia_campaign([
            "external", "--weight", "pre300", "--opponent", "claustrophobia",
            "--time-control", "200ms"]))
        self.assertTrue(queue.claustrophobia_campaign([
            "external-3plus2", "--network", "504", "--opponent", "claustrophobia"]))
        self.assertFalse(queue.claustrophobia_campaign([
            "report", "--mode", "external", "--opponent", "claustrophobia"]))
        self.assertFalse(queue.claustrophobia_campaign([
            "external", "--network", "504", "--opponent", "titanium"]))

    def test_native_clock_revalidation_plan_restarts_all_network_and_edge_stages(self):
        plan = queue.campaign_task_plan()
        summary = queue.campaign_plan_summary(plan)
        self.assertEqual(summary["task_count"], 156)
        self.assertEqual(summary["jobs_by_stage"], {
            "network_claustrophobia_200_report": 1,
            "network_h2h_200": 3, "edge_h2h_200": 6,
            "edge_external_200": 16, "network_external_200": 6,
            "network_h2h_3plus2": 12, "edge_external_3plus2": 64,
            "network_external_3plus2": 24, "edge_h2h_3plus2": 24,
        })
        self.assertEqual(summary["games_total_max"], 22600)
        self.assertEqual(summary["games_total_if_v301_off_reuse_is_exact"], 21200)
        self.assertEqual(summary["potential_v301_off_games_reused"], 1400)
        stages = [task["stage"] for task in plan]
        self.assertEqual([task["name"] for task in plan[:3]], [
            "network_200ms_pre300_claustrophobia",
            "network_200ms_v300_claustrophobia",
            "network_200ms_v301_claustrophobia",
        ])
        self.assertEqual(plan[3]["stage"], "network_claustrophobia_200_report")
        self.assertEqual(plan[3]["args"], ["report", "--mode", "external",
                                             "--opponent", "claustrophobia",
                                             "--time-control", "200ms"])
        self.assertTrue(all(task["args"][task["args"].index("--opponent") + 1]
                            == "claustrophobia" for task in plan[:3]))
        self.assertEqual(len({task["name"] for task in plan}), len(plan))
        self.assertEqual(stages[4:7], ["network_h2h_200"] * 3)
        self.assertEqual(stages[7:13], ["edge_h2h_200"] * 6)
        self.assertLess(stages.index("network_h2h_3plus2"),
                        stages.index("edge_external_3plus2"))
        self.assertLess(stages.index("network_external_3plus2"),
                        stages.index("edge_h2h_3plus2"))
        self.assertTrue(all("--shard-index" in task["args"]
                            for task in plan if task["stage"].endswith("3plus2")))

    def test_center_loss_cannot_hide_behind_normal_gain(self):
        self.assertFalse(queue.h2h_screen(h2h(center=48, normal=70))[0])
        self.assertFalse(queue.external_screen(external(center_upper=-0.5, normal_upper=20))[0])

    def test_normal_clear_regression_also_blocks(self):
        self.assertFalse(queue.h2h_screen(h2h(normal=40, upper_normal=49))[0])
        self.assertFalse(queue.external_screen(external(normal_upper=-0.5))[0])

    def test_incomplete_pairs_never_qualify(self):
        data = h2h()
        data["center"]["complete_pairs"] = 132
        with self.assertRaises(RuntimeError):
            queue.h2h_screen(data)
        data = external()
        data["paired_delta_candidate_minus_off"]["center"]["complete_paired_openings"] = 99
        with self.assertRaises(RuntimeError):
            queue.external_screen(data)

    def test_uncertain_external_loss_is_not_clear_loss(self):
        self.assertTrue(queue.h2h_screen(h2h())[0])
        self.assertTrue(queue.external_screen(external())[0])

    def test_changed_source_refuses_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(queue, "snapshot", return_value={"version": "a"}):
                queue.Queue(Path(directory))
            with patch.object(queue, "snapshot", return_value={"version": "b"}):
                with self.assertRaises(RuntimeError):
                    queue.Queue(Path(directory))

    def test_restart_preserves_live_child_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(queue, "snapshot", return_value={"version": "a"}):
                first = queue.Queue(Path(directory))
                first.status["current"] = {"child_pid": 123, "child_token": 456}
                first.save()
                second = queue.Queue(Path(directory))
                self.assertEqual(second.orphan, {"child_pid": 123, "child_token": 456})

    def test_waiting_status_preserves_live_child_identity(self):
        current = {"name": "network_200ms_pre300_claustrophobia",
                   "child_pid": 28128, "child_token": 98765,
                   "log": "pre300.log"}
        waiting = queue.waiting_current(current, 28128, 98765)
        self.assertEqual(waiting["child_pid"], 28128)
        self.assertEqual(waiting["child_token"], 98765)
        self.assertEqual(waiting["pid"], 28128)
        self.assertEqual(waiting["name"], current["name"])
        self.assertEqual(waiting["log"], current["log"])

    @unittest.skipUnless(os.name == "nt", "Windows process identity")
    def test_windows_process_token_is_stable(self):
        token = queue.process_token(os.getpid())
        self.assertIsInstance(token, int)
        self.assertEqual(token, queue.process_token(os.getpid()))


if __name__ == "__main__":
    unittest.main()
