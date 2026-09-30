"""Unit and behavioral tests for Pure ALOHA and Slotted ALOHA simulation engines."""

import math
import pytest
import numpy as np

from simulation.base import Packet, PacketStatus
from simulation.pure_aloha import simulate_pure_aloha
from simulation.slotted_aloha import simulate_slotted_aloha


# =====================================================================
# Pure ALOHA Tests
# =====================================================================

def test_pure_aloha_single_isolated_transmission_succeeds():
    """Verify that a single isolated transmission in Pure ALOHA completes successfully."""
    pkt = Packet(packet_id=0, source_node=0, generation_time=1.0)
    result = simulate_pure_aloha(
        num_nodes=1,
        sim_time=10.0,
        offered_load=0.0,
        packet_duration=1.0,
        propagation_delay=0.0,
        max_retries=5,
        seed=42,
        initial_packets=[pkt],
    )

    packets = result["packets"]
    channel_log = result["channel_log"]
    nodes = result["nodes"]

    assert len(packets) == 1
    assert packets[0].status == PacketStatus.SUCCESS
    assert math.isclose(packets[0].attempt_time, 1.0, abs_tol=1e-7)
    assert math.isclose(packets[0].successful_time, 2.0, abs_tol=1e-7)
    assert packets[0].retry_count == 0

    assert len(channel_log) == 1
    start, end, node_id, status = channel_log[0]
    assert math.isclose(start, 1.0, abs_tol=1e-7)
    assert math.isclose(end, 2.0, abs_tol=1e-7)
    assert node_id == 0
    assert status == "SUCCESS"

    assert nodes[0].packets_successful == 1
    assert nodes[0].packets_collided == 0
    assert nodes[0].retransmissions == 0
    assert math.isclose(nodes[0].total_delay, 1.0, abs_tol=1e-7)


def test_pure_aloha_two_forced_overlapping_transmissions_collide():
    """Verify that two transmissions overlapping in time in Pure ALOHA both collide."""
    # Node 0 transmits on [1.0, 2.0], Node 1 transmits on [1.5, 2.5] -> overlap on [1.5, 2.0]
    pkts = [
        Packet(packet_id=0, source_node=0, generation_time=1.0),
        Packet(packet_id=1, source_node=1, generation_time=1.5),
    ]

    result = simulate_pure_aloha(
        num_nodes=2,
        sim_time=10.0,
        offered_load=0.0,
        packet_duration=1.0,
        propagation_delay=0.0,
        max_retries=0,  # No retries to isolate the initial collision
        seed=42,
        initial_packets=pkts,
    )

    channel_log = result["channel_log"]
    nodes = result["nodes"]
    packets = result["packets"]

    # Both initial transmissions must be logged as COLLISION
    assert len(channel_log) == 2
    assert channel_log[0][3] == "COLLISION"
    assert channel_log[1][3] == "COLLISION"

    assert nodes[0].packets_collided == 1
    assert nodes[1].packets_collided == 1

    # With max_retries=0, both packets must be dropped after the first collision
    assert packets[0].status == PacketStatus.DROPPED
    assert packets[1].status == PacketStatus.DROPPED


def test_pure_aloha_collision_triggers_retransmission():
    """Verify that a collision in Pure ALOHA increments retry count and triggers retransmission."""
    pkts = [
        Packet(packet_id=0, source_node=0, generation_time=1.0),
        Packet(packet_id=1, source_node=1, generation_time=1.4),
    ]

    result = simulate_pure_aloha(
        num_nodes=2,
        sim_time=25.0,
        offered_load=0.0,
        packet_duration=1.0,
        propagation_delay=0.0,
        max_retries=5,
        seed=123,
        initial_packets=pkts,
    )

    channel_log = result["channel_log"]
    nodes = result["nodes"]

    # At least the 2 initial colliding transmissions plus subsequent retransmissions
    assert len(channel_log) >= 3
    assert nodes[0].retransmissions >= 1 or nodes[1].retransmissions >= 1
    assert nodes[0].packets_collided >= 1
    assert nodes[1].packets_collided >= 1


def test_pure_aloha_max_retries_exceeded_drops_packet():
    """Verify that Pure ALOHA drops a packet after exceeding max_retries."""
    # Force repeated collisions by keeping nodes colliding on identical initial packets with max_retries=1
    pkts = [
        Packet(packet_id=0, source_node=0, generation_time=1.0),
        Packet(packet_id=1, source_node=1, generation_time=1.0),
    ]

    result = simulate_pure_aloha(
        num_nodes=2,
        sim_time=5.0,
        offered_load=0.0,
        packet_duration=1.0,
        propagation_delay=0.0,
        max_retries=0,
        seed=42,
        initial_packets=pkts,
    )

    packets = result["packets"]
    assert packets[0].status == PacketStatus.DROPPED
    assert packets[1].status == PacketStatus.DROPPED


def test_pure_aloha_consecutive_non_overlapping_transmissions_succeed():
    """Verify that sequential transmissions that do not overlap succeed in Pure ALOHA."""
    pkts = [
        Packet(packet_id=0, source_node=0, generation_time=1.0),
        Packet(packet_id=1, source_node=1, generation_time=2.1),  # Starts after Node 0 finishes at 2.0
    ]

    result = simulate_pure_aloha(
        num_nodes=2,
        sim_time=10.0,
        offered_load=0.0,
        packet_duration=1.0,
        propagation_delay=0.0,
        max_retries=5,
        seed=42,
        initial_packets=pkts,
    )

    channel_log = result["channel_log"]
    assert len(channel_log) == 2
    assert channel_log[0][3] == "SUCCESS"
    assert channel_log[1][3] == "SUCCESS"
    assert result["packets"][0].status == PacketStatus.SUCCESS
    assert result["packets"][1].status == PacketStatus.SUCCESS


# =====================================================================
# Slotted ALOHA Tests
# =====================================================================

def test_slotted_aloha_single_transmission_in_slot_succeeds():
    """Verify that a single transmission in a slot boundary succeeds in Slotted ALOHA."""
    # Packet arrives at t=0.4; next slot boundary is t=1.0 for slot_duration=1.0
    pkt = Packet(packet_id=0, source_node=0, generation_time=0.4)
    result = simulate_slotted_aloha(
        num_nodes=1,
        sim_time=10.0,
        offered_load=0.0,
        packet_duration=1.0,
        propagation_delay=0.0,
        max_retries=5,
        seed=42,
        initial_packets=[pkt],
    )

    channel_log = result["channel_log"]
    packets = result["packets"]
    nodes = result["nodes"]

    assert len(channel_log) == 1
    start, end, node_id, status = channel_log[0]
    assert math.isclose(start, 1.0, abs_tol=1e-7)
    assert math.isclose(end, 2.0, abs_tol=1e-7)
    assert node_id == 0
    assert status == "SUCCESS"

    assert packets[0].status == PacketStatus.SUCCESS
    assert math.isclose(packets[0].attempt_time, 1.0, abs_tol=1e-7)
    assert math.isclose(packets[0].successful_time, 2.0, abs_tol=1e-7)
    assert nodes[0].packets_successful == 1


def test_slotted_aloha_two_transmissions_in_same_slot_collide():
    """Verify that two transmissions scheduled for the same slot collide in Slotted ALOHA."""
    # Both arrive before slot 1 (t=1.0); both transmit at slot 1
    pkts = [
        Packet(packet_id=0, source_node=0, generation_time=0.2),
        Packet(packet_id=1, source_node=1, generation_time=0.7),
    ]

    result = simulate_slotted_aloha(
        num_nodes=2,
        sim_time=10.0,
        offered_load=0.0,
        packet_duration=1.0,
        propagation_delay=0.0,
        max_retries=0,  # Drop after first collision
        seed=42,
        initial_packets=pkts,
    )

    channel_log = result["channel_log"]
    assert len(channel_log) == 2
    # Both transmitted in slot [1.0, 2.0]
    assert math.isclose(channel_log[0][0], 1.0, abs_tol=1e-7)
    assert math.isclose(channel_log[1][0], 1.0, abs_tol=1e-7)
    assert channel_log[0][3] == "COLLISION"
    assert channel_log[1][3] == "COLLISION"

    assert result["packets"][0].status == PacketStatus.DROPPED
    assert result["packets"][1].status == PacketStatus.DROPPED


def test_slotted_aloha_transmissions_never_start_off_slot_boundary():
    """Verify strictly that NO transmission in Slotted ALOHA ever starts off a slot boundary."""
    slot_size = 1.25
    result = simulate_slotted_aloha(
        num_nodes=8,
        sim_time=100.0,
        offered_load=1.2,
        packet_duration=slot_size,
        propagation_delay=0.0,
        max_retries=10,
        seed=999,
    )

    channel_log = result["channel_log"]
    assert len(channel_log) > 0, "Simulation should generate and log transmission events"

    for start_time, end_time, node_id, status in channel_log:
        # Check start_time is an exact multiple of slot_size
        slot_number = round(start_time / slot_size)
        expected_start = round(slot_number * slot_size, 9)
        expected_end = round((slot_number + 1) * slot_size, 9)

        assert math.isclose(start_time, expected_start, abs_tol=1e-6), (
            f"Transmission by node {node_id} started at {start_time}, which is off-slot (expected {expected_start})"
        )
        assert math.isclose(end_time, expected_end, abs_tol=1e-6), (
            f"Transmission duration {end_time - start_time} does not match slot_size {slot_size}"
        )


def test_seed_randomness_reproducibility():
    """Verify that running with the same seed produces identical simulation outcomes."""
    res1 = simulate_pure_aloha(
        num_nodes=4,
        sim_time=50.0,
        offered_load=0.8,
        packet_duration=1.0,
        propagation_delay=0.0,
        max_retries=10,
        seed=777,
    )
    res2 = simulate_pure_aloha(
        num_nodes=4,
        sim_time=50.0,
        offered_load=0.8,
        packet_duration=1.0,
        propagation_delay=0.0,
        max_retries=10,
        seed=777,
    )

    assert len(res1["channel_log"]) == len(res2["channel_log"])
    for e1, e2 in zip(res1["channel_log"], res2["channel_log"]):
        assert math.isclose(e1[0], e2[0], abs_tol=1e-9)
        assert math.isclose(e1[1], e2[1], abs_tol=1e-9)
        assert e1[2] == e2[2]
        assert e1[3] == e2[3]
