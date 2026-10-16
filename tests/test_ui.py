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


def test_app_consistency_expander_single_simulation():
    """Verify that Consistency Checks expander renders in Single Simulation mode."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception

    # Expander for consistency checks should be rendered
    expander_labels = [e.label for e in at.expander]
    consistency_expanders = [l for l in expander_labels if "Consistency Checks" in l]
    assert len(consistency_expanders) >= 1
    assert "Passed" in consistency_expanders[0]


def test_app_consistency_expander_comparison_mode():
    """Verify that Consistency Checks expander renders in Comparison mode."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value("Compare All Protocols")
    at.run()
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception

    expander_labels = [e.label for e in at.expander]
    consistency_expanders = [l for l in expander_labels if "Consistency Checks" in l]
    assert len(consistency_expanders) >= 1
    assert any("Passed" in l for l in consistency_expanders)


def test_sweep_plots_error_bars_clipped_at_zero():
    """Verify that lower error bounds on sweep plots never drop below zero."""
    import numpy as np
    import pandas as pd
    from visualization.sweep_plots import plot_load_vs_collision_rate

    # Create dummy dataframe where standard deviation exceeds the mean (would go negative)
    dummy_df = pd.DataFrame({
        "load": [0.1, 0.5],
        "protocol": ["Pure ALOHA", "Pure ALOHA"],
        "collision_rate_mean": [0.02, 0.10],
        "collision_rate_std": [0.08, 0.05],  # 0.02 - 0.08 = -0.06 without clipping
    })

    fig = plot_load_vs_collision_rate(dummy_df)
    trace = fig.data[0]
    error_y = trace.error_y

    assert error_y.visible is True
    assert error_y.symmetric is False
    # arrayminus should be clamped such that y - arrayminus >= 0
    y_vals = trace.y  # scaled by 100 in plot_load_vs_collision_rate
    arrayminus = error_y.arrayminus
    for y, minus in zip(y_vals, arrayminus):
        assert y - minus >= -1e-9, f"Lower error bar dropped below zero: y={y}, minus={minus}"


def test_app_single_simulation_download_buttons():
    """Verify all CSV download buttons render and contain valid CSV data in Single Simulation mode."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception

    # Locate download buttons
    dl_labels = [b.label for b in at.download_button]
    assert any("Metrics Summary" in l for l in dl_labels)
    assert any("Consistency Results" in l for l in dl_labels)
    assert any("Packet-Level Table" in l for l in dl_labels)

    # Click each download button to verify the export pipeline executes cleanly
    for b in at.download_button:
        if not b.disabled:
            b.click()
            at.run()
            assert not at.exception


def test_app_comparison_mode_download_buttons():
    """Verify all CSV download buttons render and execute cleanly in Comparison mode."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value("Compare All Protocols")
    at.run()
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception

    dl_labels = [b.label for b in at.download_button]
    assert any("Comparison Metrics" in l for l in dl_labels)
    assert any("Comparison Consistency Results" in l for l in dl_labels)

    for b in at.download_button:
        if not b.disabled:
            b.click()
            at.run()
            assert not at.exception


def test_app_load_sweep_theoretical_overlay_and_downloads():
    """Verify theoretical overlay checkbox, note, peak table, and download buttons in Load Analysis."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    # Click run sweep
    sweep_btn = None
    for b in at.button:
        if "Run Offered Load Sweep" in b.label:
            sweep_btn = b
            break
    assert sweep_btn is not None
    sweep_btn.click()
    at.run()
    assert not at.exception

    # Verify theoretical overlay checkbox exists
    theory_checkbox = None
    for cb in at.checkbox:
        if "Overlay Theoretical Throughput Curves" in cb.label:
            theory_checkbox = cb
            break
    assert theory_checkbox is not None
    assert theory_checkbox.value is False

    # Toggle checkbox to True
    theory_checkbox.check()
    at.run()
    assert not at.exception

    # Verify under-plot explanatory note
    info_texts = [info.value for info in at.info]
    assert any("Simulated values can differ from theory" in text for text in info_texts)

    # Verify peak throughput comparison table
    df_values = [df.value for df in at.dataframe]
    peak_dfs = [df for df in df_values if "simulated_peak_throughput" in df.columns]
    assert len(peak_dfs) >= 1
    assert "Pure ALOHA" in list(peak_dfs[0]["protocol"])
    assert "Slotted ALOHA" in list(peak_dfs[0]["protocol"])

    # Verify download buttons for load sweep
    dl_labels = [b.label for b in at.download_button]
    assert any("Raw Load Sweep Results" in l for l in dl_labels)
    assert any("Aggregated Load Sweep Summary" in l for l in dl_labels)

    for b in at.download_button:
        if "Load Sweep" in b.label and not b.disabled:
            b.click()
            at.run()
            assert not at.exception


def test_app_node_sweep_execution_and_downloads():
    """Verify node count sweep execution and download buttons."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    # Click run node sweep
    node_btn = None
    for b in at.button:
        if "Run Node Count Sweep" in b.label:
            node_btn = b
            break
    assert node_btn is not None
    node_btn.click()
    at.run()
    assert not at.exception

    assert "node_sweep_df" in at.session_state
    dl_labels = [b.label for b in at.download_button]
    assert any("Raw Node Sweep Results" in l for l in dl_labels)
    assert any("Aggregated Node Sweep Summary" in l for l in dl_labels)

    for b in at.download_button:
        if "Node Sweep" in b.label and not b.disabled:
            b.click()
            at.run()
            assert not at.exception


