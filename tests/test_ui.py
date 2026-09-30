"""Integration and end-to-end tests for the Streamlit UI (app.py)."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str((Path(__file__).parent.parent / "app.py").resolve())


def test_app_initial_load():
    """Verify that app.py loads cleanly on startup without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception, f"App raised an exception: {at.exception}"
    # Verify title and 7 tabs rendered
    assert at.title[0].value == "ALOHA vs CSMA Performance Analyzer"
    assert len(at.tabs) == 7


def test_app_run_single_sim_pure_aloha():
    """Verify running single simulation for Pure ALOHA via the UI."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value("Single Simulation")
    at.sidebar.selectbox[0].select("Pure ALOHA")
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception
    assert len(at.metric) >= 9


def test_app_run_single_sim_slotted_aloha():
    """Verify running single simulation for Slotted ALOHA via the UI."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value("Single Simulation")
    at.sidebar.selectbox[0].select("Slotted ALOHA")
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception
    assert len(at.metric) >= 9


def test_app_run_single_sim_csma():
    """Verify running single simulation for 1-Persistent CSMA via the UI."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value("Single Simulation")
    at.sidebar.selectbox[0].select("1-Persistent CSMA")
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception
    assert len(at.metric) >= 9


def test_app_comparison_mode():
    """Verify running comparison mode on identical traffic trace via UI."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception

    # Switch mode to "Compare All Protocols"
    at.sidebar.radio[0].set_value("Compare All Protocols")
    at.run()
    assert not at.exception

    # Click Run Comparison button in sidebar
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception

    # Verify comparison result stored in session state
    assert "comparison_result" in at.session_state
    comp_res = at.session_state["comparison_result"]
    assert "Pure ALOHA" in comp_res["metrics"]
    assert "Slotted ALOHA" in comp_res["metrics"]
    assert "1-Persistent CSMA" in comp_res["metrics"]


def test_app_load_sweep_execution():
    """Verify running an offered load sweep via the UI."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    # Find the button to run offered load sweep
    sweep_btn = None
    for b in at.button:
        if "Run Offered Load Sweep" in b.label:
            sweep_btn = b
            break

    assert sweep_btn is not None, "Run Offered Load Sweep button should be present"
    sweep_btn.click()
    at.run()

    assert not at.exception
    assert "load_sweep_df" in at.session_state
    df = at.session_state["load_sweep_df"]
    assert len(df) > 0
    assert "throughput_mean" in df.columns
    assert "collision_rate_mean" in df.columns
