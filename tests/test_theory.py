"""Unit tests for analytical MAC throughput models and peak comparison."""

import math
import numpy as np
import pandas as pd
import pytest

from simulation.theory import (
    compute_throughput_peaks,
    pure_aloha_throughput,
    slotted_aloha_throughput,
)


def test_pure_aloha_theory_values():
    """Validates S = G * exp(-2G) returns 0 at G=0, 1/(2e) at G=0.5, and 0 for negative G."""
    assert pure_aloha_throughput(0.0) == 0.0
    assert pure_aloha_throughput(0) == 0.0
    assert pure_aloha_throughput(-0.5) == 0.0

    expected_peak = 1.0 / (2.0 * math.e)
    assert pure_aloha_throughput(0.5) == pytest.approx(expected_peak, rel=1e-7)

    # Array input check
    loads = np.array([0.0, 0.25, 0.5, 1.0])
    s_vals = pure_aloha_throughput(loads)
    assert isinstance(s_vals, np.ndarray)
    assert len(s_vals) == len(loads)
    assert s_vals[0] == 0.0
    assert s_vals[2] == pytest.approx(expected_peak, rel=1e-7)


def test_slotted_aloha_theory_values():
    """Validates S = G * exp(-G) returns 0 at G=0, 1/e at G=1.0, and 0 for negative G."""
    assert slotted_aloha_throughput(0.0) == 0.0
    assert slotted_aloha_throughput(0) == 0.0
    assert slotted_aloha_throughput(-1.0) == 0.0

    expected_peak = 1.0 / math.e
    assert slotted_aloha_throughput(1.0) == pytest.approx(expected_peak, rel=1e-7)

    # Array input check
    loads = np.array([0.0, 0.5, 1.0, 2.0])
    s_vals = slotted_aloha_throughput(loads)
    assert isinstance(s_vals, np.ndarray)
    assert len(s_vals) == len(loads)
    assert s_vals[0] == 0.0
    assert s_vals[2] == pytest.approx(expected_peak, rel=1e-7)


def test_compute_throughput_peaks_hand_built():
    """Validates peak-finding helper identifies empirical peaks accurately."""
    df_sample = pd.DataFrame([
        {"protocol": "Pure ALOHA", "load": 0.2, "throughput_mean": 0.134},
        {"protocol": "Pure ALOHA", "load": 0.5, "throughput_mean": 0.183},
        {"protocol": "Pure ALOHA", "load": 0.8, "throughput_mean": 0.161},
        {"protocol": "Slotted ALOHA", "load": 0.5, "throughput_mean": 0.303},
        {"protocol": "Slotted ALOHA", "load": 1.0, "throughput_mean": 0.367},
        {"protocol": "Slotted ALOHA", "load": 1.5, "throughput_mean": 0.334},
    ])

    peaks_df = compute_throughput_peaks(df_sample)
    assert len(peaks_df) == 2
    assert list(peaks_df["protocol"]) == ["Pure ALOHA", "Slotted ALOHA"]

    pure_row = peaks_df[peaks_df["protocol"] == "Pure ALOHA"].iloc[0]
    assert pure_row["simulated_peak_throughput"] == 0.183
    assert pure_row["simulated_peak_g"] == 0.5
    assert pure_row["theoretical_peak_throughput"] == 0.184
    assert pure_row["theoretical_peak_g"] == 0.5

    slotted_row = peaks_df[peaks_df["protocol"] == "Slotted ALOHA"].iloc[0]
    assert slotted_row["simulated_peak_throughput"] == 0.367
    assert slotted_row["simulated_peak_g"] == 1.0
    assert slotted_row["theoretical_peak_throughput"] == 0.368
    assert slotted_row["theoretical_peak_g"] == 1.0


def test_compute_throughput_peaks_empty():
    """Validates compute_throughput_peaks gracefully handles an empty DataFrame."""
    empty_df = pd.DataFrame()
    peaks_df = compute_throughput_peaks(empty_df)
    assert len(peaks_df) == 2
    assert list(peaks_df["protocol"]) == ["Pure ALOHA", "Slotted ALOHA"]
    assert peaks_df.iloc[0]["simulated_peak_throughput"] is None
    assert peaks_df.iloc[0]["theoretical_peak_throughput"] == 0.184
