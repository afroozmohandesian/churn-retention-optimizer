"""Smoke tests for the Streamlit scenario explorer (skipped when Streamlit is absent)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit", minversion="1.36")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def run(**setup):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    for step in setup.get("steps", []):
        step(at)
        at.run()
    assert not at.exception, at.exception
    return at


def test_default_scenario_renders():
    at = run()
    assert at.title[0].value.startswith("Project Odin")
    assert len(at.metric) == 4


def test_pilot_dataset():
    at = run(steps=[lambda a: a.sidebar.radio[0].set_value("Pilot cohort (30)")])
    assert at.metric[0].value == "6"


def test_uplift_multi_offer_and_constraints():
    run(
        steps=[
            lambda a: a.sidebar.radio[1].set_value("Uplift model (per subscriber)"),
            lambda a: a.sidebar.checkbox[0].check(),
        ]
    )
    run(steps=[lambda a: a.sidebar.toggle[0].set_value(True)])
