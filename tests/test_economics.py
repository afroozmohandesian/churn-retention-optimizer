import pytest

from odin import add_expected_value
from odin.economics import break_even_success_rate
from tests.conftest import make


def test_formulas_match_worked_example():
    df = add_expected_value(make([("X", 1, 0.60, 1000, 80)]), 0.50)
    assert df.loc[0, "Expected_Retained_Value"] == pytest.approx(300.0)
    assert df.loc[0, "Expected_Net_Benefit"] == pytest.approx(220.0)


def test_input_is_not_mutated(raw):
    before = raw.copy()
    add_expected_value(raw, 0.5)
    assert raw.equals(before)


def test_break_even_rate():
    df = make([("X", 1, 0.5, 1000, 100)])
    assert break_even_success_rate(df).iloc[0] == pytest.approx(0.2)


def test_invalid_success_rate(raw):
    with pytest.raises(ValueError):
        add_expected_value(raw, 1.2)
