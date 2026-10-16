"""Consistency checker and invariant verification for ALOHA vs CSMA simulations.

Validates simulation invariants, packet lifecycle accounting, metric boundaries,
temporal causality, deterministic reproducibility, and cross-path equivalence
between Single Simulation and Comparison Mode.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

from .base import ChannelEventStatus, Node, Packet, PacketStatus
from .csma import simulate_csma
from .metrics import calculate_all_metrics
from .pure_aloha import simulate_pure_aloha
from .slotted_aloha import simulate_slotted_aloha
from .sweep import run_comparison_simulation


def _get_simulator(protocol: str):
    """Returns the simulation entrypoint for a given protocol string."""
    if protocol == "Pure ALOHA":
        return simulate_pure_aloha
    elif protocol == "Slotted ALOHA":
        return simulate_slotted_aloha
    elif protocol == "1-Persistent CSMA":
        return simulate_csma
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


def _check_single_simulation(
    result: Dict[str, Any],
    params: Dict[str, Any],
) -> List[Tuple[str, bool, str]]:
    """Runs all 9 consistency checks on a single protocol simulation result."""
    checks: List[Tuple[str, bool, str]] = []

    packets: List[Packet] = result.get("packets", [])
    channel_log: List[Tuple[float, float, int, str]] = result.get("channel_log", [])
    nodes: List[Node] = result.get("nodes", [])

    sim_time = float(params.get("sim_time", 1000.0))
    packet_duration = float(params.get("packet_duration", 1.0))
    protocol = params.get("protocol", "Pure ALOHA")
    seed = int(params.get("seed", 42))

    # Pre-calculated or newly evaluated metrics
    metrics = result.get("metrics")
    if metrics is None:
        metrics = calculate_all_metrics(
            packets=packets,
            channel_log=channel_log,
            sim_time=sim_time,
            packet_duration=packet_duration,
        )

    # -----------------------------------------------------------------
    # Check 1: packets_generated == successful + dropped + pending
    # -----------------------------------------------------------------
    gen_count = len(packets)
    succ_count = sum(1 for p in packets if p.status == PacketStatus.SUCCESS)
    drop_count = sum(1 for p in packets if p.status == PacketStatus.DROPPED)
    pend_count = sum(1 for p in packets if p.status == PacketStatus.PENDING)

    passed_1 = (gen_count == (succ_count + drop_count + pend_count))
    details_1 = (
        f"Generated ({gen_count}) == Successful ({succ_count}) + "
        f"Dropped ({drop_count}) + Pending ({pend_count})"
    )
    if not passed_1:
        details_1 += f" [Mismatch: sum={succ_count + drop_count + pend_count} != generated={gen_count}]"
    checks.append(("Packet Accounting (generated == successful + dropped + pending)", passed_1, details_1))

    # -----------------------------------------------------------------
    # Check 2: sum of per-node packets_generated/successful equals packet-level counts
    # -----------------------------------------------------------------
    if nodes:
        node_gen = sum(n.packets_generated for n in nodes)
        node_succ = sum(n.packets_successful for n in nodes)
        passed_2 = (node_gen == gen_count) and (node_succ == succ_count)
        details_2 = (
            f"Generated: nodes={node_gen}, packets={gen_count}; "
            f"Successful: nodes={node_succ}, packets={succ_count}"
        )
    else:
        passed_2 = True
        details_2 = "Node list not present in result; skipped per-node check"
    checks.append(("Per-Node vs Packet-Level Counts", passed_2, details_2))

    # -----------------------------------------------------------------
    # Check 3: number of SUCCESS entries in channel_log equals successful packets
    # -----------------------------------------------------------------
    log_succ = sum(1 for e in channel_log if e[3] == "SUCCESS" or e[3] == ChannelEventStatus.SUCCESS)
    passed_3 = (log_succ == succ_count)
    details_3 = f"Channel log SUCCESS events ({log_succ}) == Packet SUCCESS count ({succ_count})"
    checks.append(("Channel Log SUCCESS Alignment", passed_3, details_3))

    # -----------------------------------------------------------------
    # Check 4: collision events in channel_log equal collision count used by collision_rate
    # -----------------------------------------------------------------
    log_cols = sum(1 for e in channel_log if e[3] == "COLLISION" or e[3] == ChannelEventStatus.COLLISION)
    col_attempts = metrics.get("collision_attempts", log_cols)
    expected_col_rate = (log_cols / len(channel_log)) if channel_log else 0.0
    rate_matches = math.isclose(metrics["collision_rate"], expected_col_rate, abs_tol=1e-9)
    passed_4 = (log_cols == col_attempts) and rate_matches
    details_4 = (
        f"Log collisions ({log_cols}) == collision_attempts ({col_attempts}); "
        f"collision_rate ({metrics['collision_rate']:.4f}) matches log fraction"
    )
    checks.append(("Collision Count and Metric Consistency", passed_4, details_4))

    # -----------------------------------------------------------------
    # Check 5: total retransmissions equals sum of packet retry counts
    # -----------------------------------------------------------------
    pkt_retries = sum(p.retry_count for p in packets)
    if nodes:
        node_retries = sum(n.retransmissions for n in nodes)
        retrans_metric_total = round(metrics["avg_retransmissions"] * gen_count) if gen_count else 0
        passed_5 = (pkt_retries == node_retries) and (pkt_retries == retrans_metric_total)
        details_5 = (
            f"Packet retry sum ({pkt_retries}) == Node retransmissions ({node_retries}) == "
            f"Metric total ({retrans_metric_total})"
        )
    else:
        retrans_metric_total = round(metrics["avg_retransmissions"] * gen_count) if gen_count else 0
        passed_5 = (pkt_retries == retrans_metric_total)
        details_5 = f"Packet retry sum ({pkt_retries}) == Metric total ({retrans_metric_total})"
    checks.append(("Total Retransmissions vs Packet Retries", passed_5, details_5))

    # -----------------------------------------------------------------
    # Check 6: every metric is in its valid range
    # -----------------------------------------------------------------
    invalid_metrics = []
    # Rates and fractions: [0, 1]
    for r_key in ["collision_rate", "success_rate", "channel_utilization", "channel_busy_fraction"]:
        if r_key in metrics:
            val = metrics[r_key]
            if not (-1e-9 <= val <= 1.0 + 1e-9):
                invalid_metrics.append(f"{r_key}={val} (expected in [0, 1])")

    # Non-negative metrics: >= 0
    for nn_key in ["throughput", "normalized_throughput", "avg_delay", "avg_retransmissions"]:
        if nn_key in metrics:
            val = metrics[nn_key]
            if val < -1e-9:
                invalid_metrics.append(f"{nn_key}={val} (expected >= 0)")

    # Non-negative counts: >= 0
    for c_key in ["dropped_packets", "pending_packets", "collision_attempts", "collision_affected_packets"]:
        if c_key in metrics:
            val = metrics[c_key]
            if val < 0:
                invalid_metrics.append(f"{c_key}={val} (expected >= 0)")

    # Relationship check: channel_busy_fraction >= channel_utilization
    if "channel_busy_fraction" in metrics and "channel_utilization" in metrics:
        if metrics["channel_busy_fraction"] < metrics["channel_utilization"] - 1e-9:
            invalid_metrics.append(
                f"channel_busy_fraction ({metrics['channel_busy_fraction']:.4f}) < "
                f"channel_utilization ({metrics['channel_utilization']:.4f})"
            )

    passed_6 = (len(invalid_metrics) == 0)
    details_6 = (
        "All metrics within valid physical and mathematical bounds (rates in [0, 1], delay >= 0, busy >= util)"
        if passed_6
        else f"Bound violations: {', '.join(invalid_metrics)}"
    )
    checks.append(("Metric Range and Validity Bounds", passed_6, details_6))

    # -----------------------------------------------------------------
    # Check 7: no packet has successful_time before generation_time
    # -----------------------------------------------------------------
    causality_errors = []
    for p in packets:
        if p.status == PacketStatus.SUCCESS:
            if p.successful_time is None:
                causality_errors.append(f"Pkt {p.packet_id} has SUCCESS status but successful_time is None")
            elif p.successful_time < p.generation_time - 1e-9:
                causality_errors.append(
                    f"Pkt {p.packet_id}: successful_time={p.successful_time:.4f} < generation_time={p.generation_time:.4f}"
                )

    passed_7 = (len(causality_errors) == 0)
    details_7 = (
        f"All {succ_count} successful packets satisfy successful_time >= generation_time"
        if passed_7
        else f"{len(causality_errors)} causality violations: {causality_errors[:3]}"
    )
    checks.append(("Temporal Causality (successful_time >= generation_time)", passed_7, details_7))

    # -----------------------------------------------------------------
    # Check 8: same seed + params run twice gives identical results (reproducibility)
    # -----------------------------------------------------------------
    simulator = _get_simulator(protocol)
    sim_kwargs = {
        "num_nodes": int(params.get("num_nodes", 10)),
        "sim_time": sim_time,
        "offered_load": float(params.get("offered_load", 0.1)),
        "packet_duration": packet_duration,
        "propagation_delay": float(params.get("propagation_delay", 0.01)),
        "max_retries": int(params.get("max_retries", 10)),
        "seed": seed,
    }
    repro_run = simulator(**sim_kwargs)
    repro_diffs = []

    if len(packets) != len(repro_run["packets"]):
        repro_diffs.append(f"Packet count mismatch ({len(packets)} vs {len(repro_run['packets'])})")
    else:
        for i, (p1, p2) in enumerate(zip(packets, repro_run["packets"])):
            if (
                p1.packet_id != p2.packet_id
                or p1.source_node != p2.source_node
                or not math.isclose(p1.generation_time, p2.generation_time, abs_tol=1e-9)
                or p1.status != p2.status
                or p1.retry_count != p2.retry_count
            ):
                repro_diffs.append(f"Packet {i} divergence")
                break

    if len(channel_log) != len(repro_run["channel_log"]):
        repro_diffs.append(f"Channel log count mismatch ({len(channel_log)} vs {len(repro_run['channel_log'])})")
    else:
        for i, (e1, e2) in enumerate(zip(channel_log, repro_run["channel_log"])):
            if (
                not math.isclose(e1[0], e2[0], abs_tol=1e-9)
                or not math.isclose(e1[1], e2[1], abs_tol=1e-9)
                or e1[2] != e2[2]
                or e1[3] != e2[3]
            ):
                repro_diffs.append(f"Channel log event {i} divergence")
                break

    passed_8 = (len(repro_diffs) == 0)
    details_8 = (
        f"Independent run with identical seed ({seed}) produced bitwise identical packets and channel events"
        if passed_8
        else f"Reproducibility divergence: {', '.join(repro_diffs)}"
    )
    checks.append(("Deterministic Reproducibility (Seed Invariance)", passed_8, details_8))

    # -----------------------------------------------------------------
    # Check 9: cross-path check: single-simulation equals comparison-mode
    # -----------------------------------------------------------------
    comp_kwargs = dict(sim_kwargs)
    comp_res = run_comparison_simulation(**comp_kwargs)
    comp_proto_res = comp_res["results"][protocol]
    comp_proto_metrics = comp_res["metrics"][protocol]

    cross_diffs = []
    if len(packets) != len(comp_proto_res["packets"]):
        cross_diffs.append(
            f"Packet count mismatch: single={len(packets)} vs comp={len(comp_proto_res['packets'])}"
        )
    else:
        for i, (p1, p2) in enumerate(zip(packets, comp_proto_res["packets"])):
            if (
                p1.packet_id != p2.packet_id
                or p1.source_node != p2.source_node
                or not math.isclose(p1.generation_time, p2.generation_time, abs_tol=1e-9)
                or p1.status != p2.status
                or p1.retry_count != p2.retry_count
            ):
                cross_diffs.append(f"Packet {i} attribute difference")
                break

    if len(channel_log) != len(comp_proto_res["channel_log"]):
        cross_diffs.append(
            f"Channel log length mismatch: single={len(channel_log)} vs comp={len(comp_proto_res['channel_log'])}"
        )
    else:
        for i, (e1, e2) in enumerate(zip(channel_log, comp_proto_res["channel_log"])):
            if (
                not math.isclose(e1[0], e2[0], abs_tol=1e-9)
                or not math.isclose(e1[1], e2[1], abs_tol=1e-9)
                or e1[2] != e2[2]
                or e1[3] != e2[3]
            ):
                cross_diffs.append(f"Channel log event {i} difference")
                break

    # Compare metrics between single run and comparison mode run
    for m_key in [
        "throughput",
        "normalized_throughput",
        "collision_rate",
        "success_rate",
        "avg_delay",
        "avg_retransmissions",
        "dropped_packets",
        "channel_utilization",
        "channel_busy_fraction",
    ]:
        if m_key in metrics and m_key in comp_proto_metrics:
            if not math.isclose(float(metrics[m_key]), float(comp_proto_metrics[m_key]), abs_tol=1e-9):
                cross_diffs.append(
                    f"Metric {m_key}: single={metrics[m_key]} vs comp={comp_proto_metrics[m_key]}"
                )

    passed_9 = (len(cross_diffs) == 0)
    details_9 = (
        f"Single simulation and comparison mode produce identical results for {protocol} "
        f"({len(packets)} pkts, {len(channel_log)} events, metrics matched)"
        if passed_9
        else f"Cross-path mismatch: {', '.join(cross_diffs)}"
    )
    checks.append(("Cross-Path Equivalence (Single vs Comparison Mode)", passed_9, details_9))

    # -----------------------------------------------------------------
    # Check 10: channel_log physical collision and transmission invariants
    # -----------------------------------------------------------------
    prop_delay = float(params.get("propagation_delay", 0.01))
    passed_10, details_10 = validate_channel_log_physics(
        channel_log=channel_log,
        protocol=protocol,
        packet_duration=packet_duration,
        propagation_delay=prop_delay,
    )
    checks.append(("Channel Log Physical Invariants", passed_10, details_10))

    return checks


def validate_channel_log_physics(
    channel_log: List[Tuple[float, float, int, str]],
    protocol: str,
    packet_duration: float = 1.0,
    propagation_delay: float = 0.01,
) -> Tuple[bool, str]:
    """Validates physical channel_log invariants independent of protocol implementation.

    Invariants checked:
    - Pure ALOHA: every SUCCESS entry overlaps in time with NO other entry in the log;
      any two overlapping entries must both be COLLISION.
    - Slotted ALOHA: every entry starts on a slot boundary; no two entries share a slot
      unless both are COLLISION; any slot with 2+ entries has all of them COLLISION.
    - 1-Persistent CSMA: a SUCCESS entry may not overlap another entry's start unless that
      start is within propagation_delay of its own start.

    Args:
        channel_log: List of (start_time, end_time, node_id, status) tuples.
        protocol: Name of protocol ('Pure ALOHA', 'Slotted ALOHA', '1-Persistent CSMA').
        packet_duration: Duration of packet transmission in seconds.
        propagation_delay: Medium propagation delay in seconds.

    Returns:
        (passed, details): Tuple of bool and human-readable diagnostic message.
    """
    if not channel_log:
        return True, "Channel log is empty; physical invariants trivially satisfied"

    violations: List[str] = []

    if protocol == "Pure ALOHA":
        for i, e1 in enumerate(channel_log):
            s1, end1, n1, st1 = e1
            for j in range(i + 1, len(channel_log)):
                e2 = channel_log[j]
                s2, end2, n2, st2 = e2
                # Overlap in continuous time: max(s1, s2) < min(end1, end2)
                if max(s1, s2) < min(end1, end2) - 1e-9:
                    if st1 == "SUCCESS" or st2 == "SUCCESS":
                        violations.append(
                            f"SUCCESS entry overlaps with another transmission: entry {i} [{s1:.4f}, {end1:.4f}] ({st1}) and entry {j} [{s2:.4f}, {end2:.4f}] ({st2})"
                        )
                    if st1 != "COLLISION" or st2 != "COLLISION":
                        violations.append(
                            f"Overlapping entries are not both COLLISION: entry {i} [{s1:.4f}, {end1:.4f}] ({st1}) and entry {j} [{s2:.4f}, {end2:.4f}] ({st2})"
                        )

    elif protocol == "Slotted ALOHA":
        from collections import defaultdict
        slot_map = defaultdict(list)
        for i, e in enumerate(channel_log):
            s, end, n, st = e
            slot_idx = round(s / packet_duration)
            expected_start = slot_idx * packet_duration
            if abs(s - expected_start) > 1e-7:
                violations.append(
                    f"Entry {i} (node {n}) starts at {s:.6f}, which is not on slot boundary ({expected_start:.6f})"
                )
            slot_map[slot_idx].append((i, e))

        for slot_idx, entries in slot_map.items():
            if len(entries) >= 2:
                for idx, entry in entries:
                    if entry[3] != "COLLISION":
                        violations.append(
                            f"Slot {slot_idx} contains {len(entries)} entries, but entry {idx} (node {entry[2]}) has status {entry[3]} (expected COLLISION)"
                        )

    elif protocol == "1-Persistent CSMA":
        for i, e1 in enumerate(channel_log):
            s1, end1, n1, st1 = e1
            if st1 == "SUCCESS":
                for j, e2 in enumerate(channel_log):
                    if i == j:
                        continue
                    s2, end2, n2, st2 = e2
                    # Does e1 overlap e2's start? (i.e. s1 <= s2 < end1)
                    if s1 <= s2 < end1 - 1e-9:
                        if (s2 - s1) > propagation_delay + 1e-9:
                            violations.append(
                                f"SUCCESS entry {i} (node {n1}, [{s1:.4f}, {end1:.4f}]) overlaps entry {j} start ({s2:.4f}), exceeding propagation_delay {propagation_delay:.4f} (diff={s2 - s1:.4f})"
                            )
                    # Does e2 overlap e1's start? (i.e. s2 <= s1 < end2)
                    if s2 <= s1 < end2 - 1e-9:
                        if (s1 - s2) > propagation_delay + 1e-9:
                            violations.append(
                                f"SUCCESS entry {i} (node {n1}, start={s1:.4f}) started during ongoing entry {j} (node {n2}, [{s2:.4f}, {end2:.4f}]), exceeding propagation_delay {propagation_delay:.4f} (diff={s1 - s2:.4f})"
                            )

    passed = len(violations) == 0
    if passed:
        details = f"All {len(channel_log)} channel_log events strictly satisfy {protocol} physical collision and transmission invariants"
    else:
        details = f"{len(violations)} physical invariant violation(s) found: {'; '.join(violations[:3])}"

    return passed, details


def run_consistency_checks(
    result: Dict[str, Any],
    params: Dict[str, Any],
) -> List[Tuple[str, bool, str]]:
    """Runs the suite of invariant and consistency checks on simulation output.

    Handles either a Single Simulation result dictionary or a Comparison Mode
    result dictionary containing multiple protocols.

    Args:
        result: Simulation output dictionary (either single-sim or comparison-mode).
        params: Configuration dictionary containing num_nodes, sim_time, offered_load,
                packet_duration, seed, etc.

    Returns:
        List of (check_name, passed, details) tuples.
    """
    if "results" in result and "trace" in result:
        # Comparison mode result: validate all evaluated protocols
        all_checks = []
        for proto in ["Pure ALOHA", "Slotted ALOHA", "1-Persistent CSMA"]:
            if proto in result["results"]:
                proto_params = dict(params)
                proto_params["protocol"] = proto
                proto_res = dict(result["results"][proto])
                if "metrics" in result and proto in result["metrics"]:
                    proto_res["metrics"] = result["metrics"][proto]
                sub_checks = _check_single_simulation(proto_res, proto_params)
                for name, passed, details in sub_checks:
                    all_checks.append((f"[{proto}] {name}", passed, details))
        return all_checks
    else:
        # Single simulation result
        return _check_single_simulation(result, params)
