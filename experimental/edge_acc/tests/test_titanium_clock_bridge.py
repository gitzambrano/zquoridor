from pathlib import Path

from tools.build_titanium_clock_bridge import (
    DEFAULT_OUTPUT,
    PATCH_MARKER,
    _tree_hash,
    _patched_session,
)


UPSTREAM_SESSION = (
    Path(__file__).resolve().parents[2]
    / "Zquoridor/external_bots/titanium/repo/src/titanium/uci/session.rs"
)


def test_patch_adds_increment_protocol_and_native_allocator_limit():
    source = UPSTREAM_SESSION.read_text(encoding="utf-8")
    patched = _patched_session(source)

    assert PATCH_MARKER in patched
    assert '"nodes" | "rem" | "time" | "inc" | "opp"' in patched
    assert '"inc" => val.parse::<f64>().map(|v| inc_arg = Some(v)).is_ok()' in patched
    assert "!value.is_finite() || value < 0.0" in patched
    assert "allocate_move_budget_with_dists_and_walls(" in patched
    assert "remaining_ms.saturating_sub(budget.safety_ms)" in patched
    assert "crate::titanium::timeman::time_alloc::MAX_RATIO" in patched
    assert "info string clock remaining_ms={} inc_ms={} opp_ms={} allocated_ms={}" in patched


def test_zero_increment_and_fixed_movetime_keep_original_semantics():
    source = UPSTREAM_SESSION.read_text(encoding="utf-8")
    patched = _patched_session(source)

    assert "if inc_arg.unwrap_or(0.0) > 0.0" in patched
    assert "} else {\n                            budget.move_ms.max(1)\n                        };" in patched
    assert "if inc_arg.is_some()" not in patched
    assert "fixed_movetime: !use_rem" in patched
    assert "} else if let Some(secs) = time_sec {" in patched
    assert "(secs * 1000.0).max(1.0) as u64" in patched
    assert _patched_session(patched) == patched


def test_default_build_output_is_the_edge_workspace_cache():
    expected = Path(__file__).resolve().parents[1] / "results/benchmarks/titanium_clock_bridge"

    assert DEFAULT_OUTPUT == expected
    assert DEFAULT_OUTPUT.is_absolute()


def test_source_tree_digest_changes_when_patch_input_changes(tmp_path):
    source = tmp_path / "src" / "session.rs"
    source.parent.mkdir()
    source.write_bytes(b"upstream")
    original = _tree_hash(tmp_path)
    patched = _tree_hash(tmp_path, {"src/session.rs": b"patched"})

    assert original != patched
