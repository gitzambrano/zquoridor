import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import run_edge_acc_campaign as campaign
from tools import run_edge_acc_report as report
from tools.external import local_arena


class ExternalParticipantsTests(unittest.TestCase):
    def test_worker_counts_respect_current_machine_limits(self):
        self.assertEqual(campaign.CONFIG["h2h_workers"], 8)
        self.assertEqual(campaign.CONFIG["gpu_workers"], 2)

    def test_native_clock_campaign_uses_a_fresh_protocol_identity(self):
        self.assertEqual(campaign.OUT.name, "edge_acc_campaign_clock_v2")
        self.assertEqual(campaign.CONFIG["clock_protocol"], "edge-acc-native-clock-v2")
        self.assertIn("deadline", campaign.CONFIG["clock_policy"]["200ms"])
        self.assertIn("increment", campaign.CONFIG["clock_policy"]["3plus2"])
        self.assertEqual(campaign.CONFIG["claustrophobia_max_sims"], 0xFFFFFFFF)

    def test_live_clock_audit_is_required_and_source_bound(self):
        expected_sources = {"arena": "arena", "claustrophobia_builder": "claustro-builder",
                            "claustrophobia_source": "claustro-src", "titanium_builder": "ti-builder"}
        expected_executables = {"claustrophobia": "claustro-exe", "titanium": "ti-exe"}
        adapters = {"claustrophobia": {"executable_sha256": "claustro-exe"},
                    "titanium": {"executable_sha256": "ti-exe"}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clock_protocol_validation.json"
            path.write_text(json.dumps({"status": "PASS", "protocol": "native-clock-v2",
                                        "source_sha256": expected_sources,
                                        "executable_sha256": expected_executables}))
            hashes = [expected_sources[key] for key in (
                "arena", "claustrophobia_builder", "claustrophobia_source", "titanium_builder")]
            with patch.object(campaign, "sha256", side_effect=hashes):
                self.assertEqual(campaign.validate_clock_protocol_audit(adapters, path)["status"], "PASS")
            stale_sources = {**expected_sources, "arena": "stale"}
            path.write_text(json.dumps({"status": "PASS", "protocol": "native-clock-v2",
                                        "source_sha256": stale_sources,
                                        "executable_sha256": expected_executables}))
            with patch.object(campaign, "sha256", side_effect=hashes):
                with self.assertRaisesRegex(RuntimeError, "source hash mismatch: arena"):
                    campaign.validate_clock_protocol_audit(adapters, path)

    def test_cached_summary_preserves_order_and_ignores_only_irrelevant_metadata(self):
        rows = []
        for opening in range(12):
            # Deliberately interleave colors in a stable, non-sorted order.
            for side, result in ((1, opening % 2), (0, (opening + 1) % 2)):
                rows.append({"opponent": "claustrophobia", "opening_index": opening,
                    "zq_player": side, "status": "ok", "result": result,
                    "termination": "goal", "repeated_states": opening % 3,
                    "plies": 18 + opening, "run_id": "shared-off-run",
                    "move_times_ms": list(range(80))})
        with patch.dict(report.CONFIG, {"bootstrap": 1200, "seed": 1927}):
            report._summarize_cached.cache_clear()
            expected = local_arena.summarize_pairs(rows, bootstrap=1200, seed=1927)
            actual = report.summarize(rows)
            self.assertEqual(actual, expected)
            misses = report._summarize_cached.cache_info().misses
            metadata_only_change = [dict(row, move_times_ms=[999]) for row in rows]
            self.assertEqual(report.summarize(metadata_only_change), expected)
            self.assertEqual(report._summarize_cached.cache_info().misses, misses)
            identity_change = [dict(row, run_id="another-run") for row in rows]
            self.assertEqual(report.summarize(identity_change), expected)
            self.assertEqual(report._summarize_cached.cache_info().misses, misses + 1)

    def test_participant_selection_and_output_isolation(self):
        self.assertEqual(campaign.normalize_participants("off"), ("off",))
        self.assertEqual(campaign.normalize_participants("off,candidate"),
                         ("candidate", "off"))
        with self.assertRaises(ValueError):
            campaign.normalize_participants("off,other")
        with patch.object(campaign, "OUT", Path("Z:/campaign")):
            off = campaign.external_output_dir("504", "off", "titanium",
                three_plus_two=True, shard_index=2, shard_count=4,
                participants=("off",))
            candidate = campaign.external_output_dir("504", "v3", "titanium",
                three_plus_two=True, shard_index=2, shard_count=4,
                participants=("candidate",))
        self.assertEqual(str(off), "Z:\\campaign\\3plus2\\504\\off\\titanium\\shard_002_of_004")
        self.assertEqual(str(candidate), "Z:\\campaign\\3plus2\\504\\v3\\titanium\\shard_002_of_004")

    def test_shared_off_rows_require_matching_provenance_and_complete_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(campaign, "OUT", root), \
                 patch.object(campaign, "MAIN", root / "main"), \
                 patch.object(campaign, "source_hash", return_value={"src/a.hpp": "main-hash"}):
                baseline_dir = campaign.external_output_dir("504", "off", "titanium",
                    three_plus_two=False, shard_index=0, shard_count=1,
                    participants=("off",))
                baseline_dir.mkdir(parents=True)
                artifacts = {}
                for name in ("nnue", "center_book", "normal_book", "titanium_executable",
                             "titanium_clock_bridge_build", "referee"):
                    path = root / name
                    path.write_bytes(name.encode())
                    artifacts[name] = path
                main_exe = root / "main-off.exe"
                main_exe.write_bytes(b"main-off")
                contract = {
                    "protocol": campaign.CONFIG["clock_protocol"], "network": "504", "opponent": "titanium",
                    "time_control": "200ms", "move_time_ms": 200, "clock_initial_ms": 0,
                    "clock_increment_ms": 0, "clock_policy": campaign.CONFIG["clock_policy"]["200ms"],
                    "claustrophobia_max_sims": campaign.CONFIG["claustrophobia_max_sims"],
                    "claustrophobia_search_policy": campaign.CONFIG["claustrophobia_search_policy"],
                    "opponent_rng": "deterministic",
                    "seed": 7, "shard": [0, 1],
                    "selected_openings": {"center": [0], "normal": [0]},
                }
                baseline_config = {**contract, "variant": "off", "participants": ["off"],
                    "baseline_only": True,
                    "participant_source_sha256": {"off": {"src/a.hpp": "main-hash"}}}
                candidate_config = {**contract, "variant": "v3", "participants": ["candidate"],
                    "baseline_only": False,
                    "participant_source_sha256": {"candidate": {"src/a.hpp": "candidate-hash"}}}
                baseline_manifest = local_arena.make_manifest(
                    baseline_config, {**artifacts, "main_executable": main_exe})
                candidate_manifest = local_arena.make_manifest(candidate_config, artifacts)
                (baseline_dir / "manifest.json").write_text(json.dumps(baseline_manifest))
                rows = []
                for category, offset in (("center", 0), ("normal", 100000)):
                    for side in (0, 1):
                        rows.append({"run_id": baseline_manifest["run_id"], "participant": "off",
                            "category": category, "opening_index": offset, "zq_player": side,
                            "status": "ok"})
                (baseline_dir / "games.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                loaded = campaign.shared_off_rows("504", "titanium", three_plus_two=False,
                    shard_index=0, shard_count=1, candidate_manifest=candidate_manifest,
                    expected_categories={"center": [(0, [])], "normal": [(0, [])]},
                    expected_off_executable=main_exe)
                self.assertEqual(len(loaded), 4)
                baseline_config["participant_source_sha256"]["off"] = {"src/a.hpp": "stale"}
                (baseline_dir / "manifest.json").write_text(json.dumps(
                    local_arena.make_manifest(baseline_config, artifacts)))
                with self.assertRaisesRegex(RuntimeError, "source hash is stale"):
                    campaign.shared_off_rows("504", "titanium", three_plus_two=False,
                        shard_index=0, shard_count=1, candidate_manifest=candidate_manifest,
                        expected_categories={"center": [(0, [])], "normal": [(0, [])]},
                        expected_off_executable=main_exe)


if __name__ == "__main__":
    unittest.main()
