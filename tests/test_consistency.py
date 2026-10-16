"""Tests for simulation consistency checker and invariant validation."""

import math
from typing import Any, Dict
import pytest

from simulation.base import ChannelEventStatus, Packet, PacketStatus
from simulation.consistency import run_consistency_checks, validate_channel_log_physics
from simulation.csma import simulate_csma
from simulation.metrics import calculate_all_metrics
from simulation.pure_aloha import simulate_pure_aloha
from simulation.slotted_aloha import simulate_slotted_aloha
from simulation.sweep import clone_trace, run_comparison_simulation


def _sim_args(params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper returning simulator arguments without the protocol key."""
    return {k: v for k, v in params.items() if k != "protocol"}


@pytest.fixture
def base_params() -> Dict[str, Any]:
    return {
        "protocol": "Pure ALOHA",
        "num_nodes": 10,
        "sim_time": 1000.0,
        "offered_load": 0.1,
        "packet_duration": 1.0,
        "propagation_delay": 0.01,
        "max_retries": 10,
        "seed": 42,
    }


def test_consistency_checks_pure_aloha_passes(base_params):
    """Verify that a standard Pure ALOHA simulation passes all 10 consistency checks."""
    result = simulate_pure_aloha(**_sim_args(base_params))
    checks = run_consistency_checks(result, base_params)

    assert len(checks) == 10
    for name, passed, details in checks:
        assert passed, f"Consistency check failed: {name} - {details}"


def test_consistency_checks_slotted_aloha_passes(base_params):
    """Verify that a Slotted ALOHA simulation passes all 10 consistency checks."""
    params = dict(base_params)
    params["protocol"] = "Slotted ALOHA"
    result = simulate_slotted_aloha(**_sim_args(params))
    checks = run_consistency_checks(result, params)

    assert len(checks) == 10
    for name, passed, details in checks:
        assert passed, f"Consistency check failed: {name} - {details}"


def test_consistency_checks_csma_passes(base_params):
    """Verify that a 1-Persistent CSMA simulation passes all 10 consistency checks."""
    params = dict(base_params)
    params["protocol"] = "1-Persistent CSMA"
    result = simulate_csma(**_sim_args(params))
    checks = run_consistency_checks(result, params)

    assert len(checks) == 10
    for name, passed, details in checks:
        assert passed, f"Consistency check failed: {name} - {details}"


def test_consistency_checks_comparison_mode(base_params):
    """Verify that running checks on a Comparison Mode result validates all 3 protocols."""
    comp_result = run_comparison_simulation(**_sim_args(base_params))
    checks = run_consistency_checks(comp_result, base_params)

    # 3 protocols * 10 checks = 30 checks
    assert len(checks) == 30
    for name, passed, details in checks:
        assert passed, f"Comparison consistency check failed: {name} - {details}"


def test_packet_accounting_failure_detection(base_params):
    """Check 1: Negative test - tampering with packet status causes failure."""
    result = simulate_pure_aloha(**_sim_args(base_params))
    tampered_packets = [
        Packet(
            packet_id=p.packet_id,
            source_node=p.source_node,
            generation_time=p.generation_time,
            attempt_time=p.attempt_time,
            successful_time=p.successful_time,
            retry_count=p.retry_count,
            status=p.status,
        )
        for p in result["packets"]
    ]
    tampered_packets[0].status = "CORRUPTED"
    tampered_result = dict(result)
    tampered_result["packets"] = tampered_packets

    checks = run_consistency_checks(tampered_result, base_params)
    check1 = next(c for c in checks if "Packet Accounting" in c[0])
    assert not check1[1], "Check 1 should fail when a packet has non-accounted status"


def test_per_node_accounting_failure_detection(base_params):
    """Check 2: Negative test - tampering with node counts causes failure."""
    result = simulate_pure_aloha(**_sim_args(base_params))
    tampered_nodes = [
        type(n)(
            node_id=n.node_id,
            packets_generated=n.packets_generated,
            packets_successful=n.packets_successful,
            retransmissions=n.retransmissions,
        )
        for n in result["nodes"]
    ]
    tampered_nodes[0].packets_generated += 999
    tampered_result = dict(result)
    tampered_result["nodes"] = tampered_nodes

    checks = run_consistency_checks(tampered_result, base_params)
    check2 = next(c for c in checks if "Per-Node vs Packet-Level" in c[0])
    assert not check2[1], "Check 2 should fail when node generation count mismatches packet count"


def test_channel_log_success_failure_detection(base_params):
    """Check 3: Negative test - missing channel log SUCCESS entries causes failure."""
    result = simulate_pure_aloha(**_sim_args(base_params))
    tampered_log = [e for e in result["channel_log"]]
    for i, e in enumerate(tampered_log):
        if e[3] == "SUCCESS":
            tampered_log[i] = (e[0], e[1], e[2], "COLLISION")
            break
    tampered_result = dict(result)
    tampered_result["channel_log"] = tampered_log

    checks = run_consistency_checks(tampered_result, base_params)
    check3 = next(c for c in checks if "Channel Log SUCCESS" in c[0])
    assert not check3[1], "Check 3 should fail when log SUCCESS count differs from packet SUCCESS count"


def test_collision_count_failure_detection(base_params):
    """Check 4: Negative test - mismatch between collision attempts and channel log causes failure."""
    result = simulate_pure_aloha(**_sim_args(base_params))
    tampered_metrics = calculate_all_metrics(result["packets"], result["channel_log"], base_params["sim_time"])
    tampered_metrics["collision_attempts"] += 42
    tampered_result = dict(result)
    tampered_result["metrics"] = tampered_metrics

    checks = run_consistency_checks(tampered_result, base_params)
    check4 = next(c for c in checks if "Collision Count" in c[0])
    assert not check4[1], "Check 4 should fail when collision_attempts does not match log"


def test_retransmissions_count_failure_detection(base_params):
    """Check 5: Negative test - packet retry count mismatch causes failure."""
    result = simulate_pure_aloha(**_sim_args(base_params))
    tampered_packets = [
        Packet(
            packet_id=p.packet_id,
            source_node=p.source_node,
            generation_time=p.generation_time,
            attempt_time=p.attempt_time,
            successful_time=p.successful_time,
            retry_count=p.retry_count,
            status=p.status,
        )
        for p in result["packets"]
    ]
    tampered_packets[0].retry_count += 50
    tampered_result = dict(result)
    tampered_result["packets"] = tampered_packets

    checks = run_consistency_checks(tampered_result, base_params)
    check5 = next(c for c in checks if "Total Retransmissions" in c[0])
    assert not check5[1], "Check 5 should fail when packet retries diverge from node retransmissions"


def test_metric_bounds_failure_detection(base_params):
    """Check 6: Negative test - out of bounds metrics trigger failure."""
    result = simulate_pure_aloha(**_sim_args(base_params))
    tampered_metrics = calculate_all_metrics(result["packets"], result["channel_log"], base_params["sim_time"])
    tampered_metrics["collision_rate"] = 1.5  # Invalid rate > 1
    tampered_result = dict(result)
    tampered_result["metrics"] = tampered_metrics

    checks = run_consistency_checks(tampered_result, base_params)
    check6 = next(c for c in checks if "Metric Range and Validity" in c[0])
    assert not check6[1], "Check 6 should fail when a rate metric exceeds 1.0"


def test_temporal_causality_failure_detection(base_params):
    """Check 7: Negative test - successful_time < generation_time triggers failure."""
    result = simulate_pure_aloha(**_sim_args(base_params))
    tampered_packets = [
        Packet(
            packet_id=p.packet_id,
            source_node=p.source_node,
            generation_time=p.generation_time,
            attempt_time=p.attempt_time,
            successful_time=p.successful_time,
            retry_count=p.retry_count,
            status=p.status,
        )
        for p in result["packets"]
    ]
    for p in tampered_packets:
        if p.status == PacketStatus.SUCCESS:
            p.successful_time = p.generation_time - 5.0  # Time travel!
            break
    tampered_result = dict(result)
    tampered_result["packets"] = tampered_packets

    checks = run_consistency_checks(tampered_result, base_params)
    check7 = next(c for c in checks if "Temporal Causality" in c[0])
    assert not check7[1], "Check 7 should fail when successful_time precedes generation_time"


def test_deterministic_reproducibility(base_params):
    """Check 8: Same seed + params run twice gives bitwise identical results."""
    res1 = simulate_pure_aloha(**_sim_args(base_params))
    res2 = simulate_pure_aloha(**_sim_args(base_params))

    assert len(res1["packets"]) == len(res2["packets"])
    for p1, p2 in zip(res1["packets"], res2["packets"]):
        assert p1.packet_id == p2.packet_id
        assert p1.source_node == p2.source_node
        assert math.isclose(p1.generation_time, p2.generation_time, abs_tol=1e-9)
        assert p1.status == p2.status
        assert p1.retry_count == p2.retry_count

    assert res1["channel_log"] == res2["channel_log"]


def test_cross_path_equivalence_single_vs_comparison(base_params):
    """Check 9: Single-simulation result equals Comparison-mode result."""
    single_res = simulate_pure_aloha(**_sim_args(base_params))
    single_metrics = calculate_all_metrics(
        single_res["packets"],
        single_res["channel_log"],
        base_params["sim_time"],
        base_params["packet_duration"],
    )

    comp_res = run_comparison_simulation(**_sim_args(base_params))
    comp_pure_res = comp_res["results"]["Pure ALOHA"]
    comp_pure_metrics = comp_res["metrics"]["Pure ALOHA"]

    assert len(single_res["packets"]) == len(comp_pure_res["packets"])
    assert len(single_res["channel_log"]) == len(comp_pure_res["channel_log"])
    assert single_metrics == comp_pure_metrics


def test_channel_log_physics_pure_aloha_violations():
    """Verify that validate_channel_log_physics catches temporal overlap violations for Pure ALOHA."""
    # Positive case: non-overlapping successes
    log_valid = [
        (0.0, 1.0, 0, "SUCCESS"),
        (1.5, 2.5, 1, "SUCCESS"),
        (3.0, 4.0, 0, "COLLISION"),
        (3.5, 4.5, 1, "COLLISION"),
    ]
    passed, _ = validate_channel_log_physics(log_valid, "Pure ALOHA")
    assert passed

    # Violation 1: SUCCESS overlapping with another entry
    log_tampered_1 = [
        (0.0, 1.0, 0, "SUCCESS"),
        (0.5, 1.5, 1, "COLLISION"),
    ]
    passed, det = validate_channel_log_physics(log_tampered_1, "Pure ALOHA")
    assert not passed
    assert "SUCCESS entry overlaps" in det

    # Violation 2: Overlapping entries not both COLLISION
    log_tampered_2 = [
        (0.0, 1.0, 0, "SUCCESS"),
        (0.5, 1.5, 1, "SUCCESS"),
    ]
    passed, det = validate_channel_log_physics(log_tampered_2, "Pure ALOHA")
    assert not passed


def test_channel_log_physics_slotted_aloha_violations():
    """Verify that validate_channel_log_physics catches slot boundary and multi-occupancy violations for Slotted ALOHA."""
    # Positive case
    log_valid = [
        (0.0, 1.0, 0, "SUCCESS"),
        (1.0, 2.0, 1, "COLLISION"),
        (1.0, 2.0, 2, "COLLISION"),
    ]
    passed, _ = validate_channel_log_physics(log_valid, "Slotted ALOHA", packet_duration=1.0)
    assert passed

    # Violation 1: Non-slot-boundary start
    log_off_boundary = [(0.3, 1.3, 0, "SUCCESS")]
    passed, det = validate_channel_log_physics(log_off_boundary, "Slotted ALOHA", packet_duration=1.0)
    assert not passed
    assert "not on slot boundary" in det

    # Violation 2: 2+ entries in same slot where one is SUCCESS
    log_tampered_slot = [
        (1.0, 2.0, 0, "SUCCESS"),
        (1.0, 2.0, 1, "COLLISION"),
    ]
    passed, det = validate_channel_log_physics(log_tampered_slot, "Slotted ALOHA", packet_duration=1.0)
    assert not passed
    assert "expected COLLISION" in det


def test_channel_log_physics_csma_violations():
    """Verify that validate_channel_log_physics catches carrier-sensing overlap violations for 1-Persistent CSMA."""
    # Positive case: overlap within propagation_delay (0.01)
    log_valid = [
        (0.0, 1.0, 0, "SUCCESS"),
        (1.5, 2.5, 1, "SUCCESS"),
    ]
    passed, _ = validate_channel_log_physics(log_valid, "1-Persistent CSMA", propagation_delay=0.01)
    assert passed

    # Violation: SUCCESS overlaps another entry starting after propagation delay (e.g. at 0.05 > 0.01)
    log_tampered = [
        (0.0, 1.0, 0, "SUCCESS"),
        (0.05, 1.05, 1, "COLLISION"),
    ]
    passed, det = validate_channel_log_physics(log_tampered, "1-Persistent CSMA", propagation_delay=0.01)
    assert not passed
    assert "exceeding propagation_delay" in det


def test_channel_log_physics_across_multiple_seeds_and_loads(base_params):
    """Verify that simulator outputs physically valid logs across multiple seeds and offered loads."""
    for seed in [7, 42, 99]:
        for load in [0.1, 0.4, 0.8]:
            p = dict(base_params)
            p["seed"] = seed
            p["offered_load"] = load

            res_pure = simulate_pure_aloha(**_sim_args(p))
            passed, det = validate_channel_log_physics(res_pure["channel_log"], "Pure ALOHA")
            assert passed, f"Pure ALOHA failed physics: {det}"

            res_slotted = simulate_slotted_aloha(**_sim_args(p))
            passed, det = validate_channel_log_physics(res_slotted["channel_log"], "Slotted ALOHA")
            assert passed, f"Slotted ALOHA failed physics: {det}"

            res_csma = simulate_csma(**_sim_args(p))
            passed, det = validate_channel_log_physics(res_csma["channel_log"], "1-Persistent CSMA")
            assert passed, f"CSMA failed physics: {det}"

