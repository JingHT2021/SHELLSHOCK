from shellshock.interaction.aim_state import AimState


def test_temporary_self_center_is_consumed_by_one_shift_only():
    state = AimState()
    state.mark_temporary_self((120.0, 240.0))

    assert state.source_for_shift((10.0, 20.0)) == (120.0, 240.0)
    assert state.source_for_shift((10.0, 20.0)) == (10.0, 20.0)
