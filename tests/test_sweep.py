"""Unit and integration tests for comparison mode and parameter sweep machinery."""

import math
from typing import Any, Dict
import numpy as np
import pytest

from simulation.base import Packet, PacketStatus
from simulation.csma import simulate_csma
from simulation.pure_aloha import simulate_pure_aloha
from simulation.slotted_aloha import simulate_slotted_aloha
from simulation.sweep import (
    clone_trace,
    compute_run_statistics,
    run_comparison_simulation,
    run_load_sweep,
    run_node_sweep,
)


# =====================================================================
# Test (a): Comparison Mode produces identical packet arrival times
# =====================================================================

def test_comparison_mode_identical_packet_arrivals():
    """Verify that comparison mode feeds an identical packet arrival trace across all 3 protocols."""
    seed = 42
    comp_result = run_comparison_simulation(
        num_nodes=6,
        sim_time=100.0,
        offered_load=0.25,
        packet_duration=1.0,
        propagation_delay=0.01,
        max_retries=10,
        seed=seed,
    )

    trace = comp_result["trace"]
    results = comp_result["results"]
    metrics = comp_result["metrics"]

    assert len(trace) > 0, "Shared packet trace should not be empty"

    pure_pkts = results["Pure ALOHA"]["packets"]
    slotted_pkts = results["Slotted ALOHA"]["packets"]
    csma_pkts = results["1-Persistent CSMA"]["packets"]

    assert len(pure_pkts) == len(trace)
    assert len(slotted_pkts) == len(trace)
    assert len(csma_pkts) == len(trace)

    for i in range(len(trace)):
        orig = trace[i]
        p_pure = pure_pkts[i]
        p_slotted = slotted_pkts[i]
        p_csma = csma_pkts[i]

        # 1. Packet ID matches
        assert p_pure.packet_id == orig.packet_id
        assert p_slotted.packet_id == orig.packet_id
        assert p_csma.packet_id == orig.packet_id

        # 2. Source Node matches
        assert p_pure.source_node == orig.source_node
        assert p_slotted.source_node == orig.source_node
        assert p_csma.source_node == orig.source_node

        # 3. Exact identical generation time across all 3 protocols
        assert math.isclose(p_pure.generation_time, orig.generation_time, abs_tol=1e-9)
        assert math.isclose(p_slotted.generation_time, orig.generation_time, abs_tol=1e-9)
        assert math.isclose(p_csma.generation_time, orig.generation_time, abs_tol=1e-9)

    # Confirm metrics dictionary populated for all 3 protocols
    assert "Pure ALOHA" in metrics
    assert "Slotted ALOHA" in metrics
    assert "1-Persistent CSMA" in metrics


# =====================================================================
# Test (b): Load sweep with small fixed config produces expected rows
# =====================================================================

def test_load_sweep_expected_rows():
    """Verify that load sweep with 2 loads x 2 runs produces the expected rows and structure."""
    loads = [0.1, 0.4]
    protocols = ["Pure ALOHA", "Slotted ALOHA", "1-Persistent CSMA"]
    num_runs = 2

    def mock_sim_runner(protocol, num_nodes, sim_time, offered_load, packet_duration, propagation_delay, max_retries, seed):
        if protocol == "Pure ALOHA":
            return simulate_pure_aloha(num_nodes, sim_time, offered_load, packet_duration, propagation_delay, max_retries, seed)
        elif protocol == "Slotted ALOHA":
            return simulate_slotted_aloha(num_nodes, sim_time, offered_load, packet_duration, propagation_delay, max_retries, seed)
        else:
            return simulate_csma(num_nodes, sim_time, offered_load, packet_duration, propagation_delay, max_retries, seed)

    sweep_results = run_load_sweep(
        load_points=loads,
        protocols=protocols,
        num_nodes=3,
        sim_time=20.0,
        packet_duration=1.0,
        propagation_delay=0.01,
        max_retries=5,
        base_seed=100,
        num_runs=num_runs,
        run_sim_cached_fn=mock_sim_runner,
    )

    # 2 loads * 3 protocols = 6 aggregated rows
    expected_rows = len(loads) * len(protocols)
    assert len(sweep_results) == expected_rows

    required_keys = [
        "load", "protocol", "runs",
        "throughput_mean", "throughput_std",
        "collision_rate_mean", "collision_rate_std",
        "avg_delay_mean", "avg_delay_std",
        "avg_retransmissions_mean", "avg_retransmissions_std",
    ]

    for row in sweep_results:
        assert row["runs"] == num_runs
        assert row["load"] in loads
        assert row["protocol"] in protocols
        for key in required_keys:
            assert key in row, f"Missing key {key} in sweep result row"
            assert isinstance(row[key], (int, float, str))


# =====================================================================
# Test (c): Mean and std computed correctly against hand-computed example
# =====================================================================

def test_mean_std_hand_computed_example():
    """Verify mean and sample standard deviation against exact hand-computed values.

    Reference calculation:
        Sample 1: throughput = 0.10, collision_rate = 0.05, avg_delay = 1.0, avg_retransmissions = 0.0
        Sample 2: throughput = 0.20, collision_rate = 0.15, avg_delay = 2.0, avg_retransmissions = 1.0
        Sample 3: throughput = 0.30, collision_rate = 0.25, avg_delay = 3.0, avg_retransmissions = 2.0

        Throughput:
            Mean = (0.10 + 0.20 + 0.30) / 3 = 0.20
            Variance (ddof=1) = ((0.1-0.2)^2 + (0.2-0.2)^2 + (0.3-0.2)^2) / (3 - 1)
                              = (0.01 + 0.0 + 0.01) / 2 = 0.01
            Std Dev = sqrt(0.01) = 0.10

        Collision Rate:
            Mean = (0.05 + 0.15 + 0.25) / 3 = 0.15
            Variance = (0.01 + 0.0 + 0.01) / 2 = 0.01
            Std Dev = sqrt(0.01) = 0.10

        Average Delay:
            Mean = (1.0 + 2.0 + 3.0) / 3 = 2.0
            Variance = (1.0 + 0.0 + 1.0) / 2 = 1.0
            Std Dev = 1.0
    """
    mock_runs = [
        {"throughput": 0.10, "collision_rate": 0.05, "avg_delay": 1.0, "avg_retransmissions": 0.0},
        {"throughput": 0.20, "collision_rate": 0.15, "avg_delay": 2.0, "avg_retransmissions": 1.0},
        {"throughput": 0.30, "collision_rate": 0.25, "avg_delay": 3.0, "avg_retransmissions": 2.0},
    ]

    stats = compute_run_statistics(mock_runs)

    assert math.isclose(stats["throughput"]["mean"], 0.20, abs_tol=1e-9)
    assert math.isclose(stats["throughput"]["std"], 0.10, abs_tol=1e-9)

    assert math.isclose(stats["collision_rate"]["mean"], 0.15, abs_tol=1e-9)
    assert math.isclose(stats["collision_rate"]["std"], 0.10, abs_tol=1e-9)

    assert math.isclose(stats["avg_delay"]["mean"], 2.0, abs_tol=1e-9)
    assert math.isclose(stats["avg_delay"]["std"], 1.0, abs_tol=1e-9)

    assert math.isclose(stats["avg_retransmissions"]["mean"], 1.0, abs_tol=1e-9)
    assert math.isclose(stats["avg_retransmissions"]["std"], 1.0, abs_tol=1e-9)


def test_single_run_statistics_zero_std():
    """Verify that a single run (num_runs=1) yields zero standard deviation."""
    single_run = [{"throughput": 0.35, "collision_rate": 0.12}]
    stats = compute_run_statistics(single_run)

    assert math.isclose(stats["throughput"]["mean"], 0.35, abs_tol=1e-9)
    assert stats["throughput"]["std"] == 0.0
    assert math.isclose(stats["collision_rate"]["mean"], 0.12, abs_tol=1e-9)
    assert stats["collision_rate"]["std"] == 0.0
