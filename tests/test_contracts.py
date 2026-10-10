import pytest

from contracts import Argument, Position, argument_id


def make_position(**overrides) -> Position:
    fields = {
        "agent": "learning",
        "round": 0,
        "level": 2,
        "confidence": 0.7,
        "reason_tags": ["concept_gap"],
    }
    fields.update(overrides)
    return Position(**fields)


@pytest.mark.parametrize("level", [-1, 5])
def test_invalid_level_raises(level: int) -> None:
    with pytest.raises(ValueError):
        make_position(level=level)


def test_invalid_reason_tag_raises() -> None:
    with pytest.raises(ValueError):
        make_position(reason_tags=["not_a_tag"])


def test_position_round_trips() -> None:
    original = make_position(agent="integrity", round=1, reason_tags=["solution_leak_risk"], cap=3)
    assert Position.from_dict(original.to_dict()) == original


@pytest.mark.parametrize("cap", [None, 0, 1, 2, 3, 4])
def test_cap_accepts_none_and_valid_levels(cap: int | None) -> None:
    assert make_position(agent="integrity", cap=cap).cap == cap


@pytest.mark.parametrize("cap", [-1, 5])
def test_cap_rejects_out_of_range(cap: int) -> None:
    with pytest.raises(ValueError):
        make_position(agent="integrity", cap=cap)


def make_argument(**overrides) -> Argument:
    fields = {
        "id": argument_id("momentum", 0, "long_idle"),
        "agent": "momentum",
        "round": 0,
        "claim": "long_idle",
        "direction": 1,
        "evidence": {"seconds_since_last_edit": 240},
    }
    fields.update(overrides)
    return Argument(**fields)


@pytest.mark.parametrize(
    "overrides",
    [{"claim": "not_a_tag"}, {"direction": 2}, {"id": ""}, {"attacks": "momentum.r0.long_idle"}],
)
def test_invalid_argument_raises(overrides: dict) -> None:
    with pytest.raises(ValueError):
        make_argument(**overrides)


def test_position_with_arguments_round_trips() -> None:
    rebuttal = make_argument(
        id=argument_id("learning", 1, "active_progress"),
        agent="learning",
        round=1,
        claim="active_progress",
        direction=0,
        evidence={"recent_progress": True},
        attacks=argument_id("momentum", 0, "long_idle"),
    )
    original = make_position(agent="learning", round=1, arguments=[rebuttal])
    assert Position.from_dict(original.to_dict()) == original


def test_position_rejects_other_agents_arguments() -> None:
    with pytest.raises(ValueError):
        make_position(agent="learning", arguments=[make_argument()])
