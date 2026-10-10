import pytest

from contracts import Position


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
