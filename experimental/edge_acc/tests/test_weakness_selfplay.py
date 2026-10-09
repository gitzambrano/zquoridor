from tools.teacher.run_weakness_selfplay import choose_effective_target


def test_resume_activates_single_extra_block_while_training_is_active():
    target, extra_activated = choose_effective_target(
        current=510_000,
        primary=500_000,
        extra=100_000,
        stage_complete=False,
    )
    assert target == 600_000
    assert extra_activated


def test_resume_stops_at_primary_target_after_training_completes():
    target, extra_activated = choose_effective_target(
        current=510_000,
        primary=500_000,
        extra=100_000,
        stage_complete=True,
    )
    assert target == 500_000
    assert not extra_activated
