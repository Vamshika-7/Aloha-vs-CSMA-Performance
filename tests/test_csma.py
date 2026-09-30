"""Unit and behavioral tests for 1-Persistent CSMA simulation engine."""

import math
import pytest
import numpy as np

from simulation.base import Packet, PacketStatus
from simulation.csma import simulate_csma
from simulation.pure_aloha import simulate_pure_aloha
from simulation.slotted_aloha import simulate_slotted_aloha
from simulation.metrics import throughput


def test_csma_transmits_immediately_when_channel_idle():
    """Verify that a node in 1-Persistent CSMA transmits immediately if the channel is idle."""
    pkt = Packet(packet_id=0, source_node=0, generation_time=1.5)
    result = simulate_csma(
        num_nodes=1,
        sim_time=10.0,
        offered_load=0.0,
        packet_duration=1.0,
        propagation_delay=0.01,
        max_retries=5,
        seed=42,
        initial_packets=[pkt],
    )

    channel_log = result["channel_log"]
    packets = result["packets"]
    nodes = result["nodes"]

    assert len(channel_log) == 1
    start, end, node_id, status = channel_log[0]
    # Sensed idle at generation time 1.5 -> transmits immediately at 1.5
    assert math.isclose(start, 1.5, abs_tol=1e-7)
    assert math.isclose(end, 2.5, abs_tol=1e-7)
    assert node_id == 0
    assert status == "SUCCESS"

    assert packets[0].status == PacketStatus.SUCCESS
    assert math.isclose(packets[0].attempt_time, 1.5, abs_tol=1e-7)
    assert math.isclose(packets[0].successful_time, 2.5, abs_tol=1e-7)
    assert nodes[0].packets_successful == 1


def test_csma_waits_while_channel_busy():
    """Verify that a node in 1-Persistent CSMA defers transmission while the channel is busy."""
    # Node 0 transmits over [1.0, 2.0]. Node 1 arrives at 1.5, after signal reached Node 1 (1.0 + 0.01 = 1.01).
    pkts = [
        Packet(packet_id=0, source_node=0, generation_time=1.0),
        Packet(packet_id=1, source_node=1, generation_time=1.5),
    ]

    prop_delay = 0.01
    result = simulate_csma(
        num_nodes=2,
        sim_time=10.0,
        offered_load=0.0,
        packet_duration=1.0,
        propagation_delay=prop_delay,
        max_retries=5,
        seed=42,
        initial_packets=pkts,
    )

    channel_log = result["channel_log"]
    assert len(channel_log) == 2

    # Node 0 must transmit immediately at 1.0 and succeed
    assert math.isclose(channel_log[0][0], 1.0, abs_tol=1e-7)
    assert math.isclose(channel_log[0][1], 2.0, abs_tol=1e-7)
    assert channel_log[0][2] == 0
    assert channel_log[0][3] == "SUCCESS"

    # Node 1 must NOT have transmitted at 1.5; it waited until channel cleared at 2.0 + prop_delay
    expected_node1_start = 2.0 + prop_delay
    assert math.isclose(channel_log[1][0], expected_node1_start, abs_tol=1e-7)
    assert math.isclose(channel_log[1][1], expected_node1_start + 1.0, abs_tol=1e-7)
    assert channel_log[1][2] == 1
    assert channel_log[1][3] == "SUCCESS"

    # Zero collisions occurred because Node 1 sensed the carrier
    assert result["nodes"][0].packets_collided == 0
    assert result["nodes"][1].packets_collided == 0


def test_csma_propagation_delay_forces_collision_between_simultaneous_senses():
    """Verify that tuning propagation delay forces a collision vs avoiding a collision.

    Scenario:
        Node 0 arrives and starts at t = 1.00.
        Node 1 arrives and senses at t = 1.05.
        Packet duration = 1.0.

    1. Large propagation delay (tau = 0.10):
       Node 0's signal arrives at Node 1 at 1.10.
       At t = 1.05, Node 1 senses IDLE because signal hasn't arrived.
       Node 1 transmits at 1.05 -> Both collide!

    2. Small propagation delay (tau = 0.02):
       Node 0's signal arrives at Node 1 at 1.02.
       At t = 1.05, Node 1 senses BUSY.
       Node 1 waits -> Node 0 succeeds, Node 1 transmits after clear and succeeds!
    """
    # Case 1: Large propagation delay -> FORCED COLLISION
    pkts_large = [
        Packet(packet_id=0, source_node=0, generation_time=1.00),
        Packet(packet_id=1, source_node=1, generation_time=1.05),
    ]
    res_collision = simulate_csma(
        num_nodes=2,
        sim_time=10.0,
        offered_load=0.0,
        packet_duration=1.0,
        propagation_delay=0.10,  # 0.10 > 0.05
        max_retries=0,  # Isolate initial collision
        seed=42,
        initial_packets=pkts_large,
    )
    log_col = res_collision["channel_log"]
    assert len(log_col) == 2
    assert log_col[0][3] == "COLLISION"
    assert log_col[1][3] == "COLLISION"
    assert res_collision["nodes"][0].packets_collided == 1
    assert res_collision["nodes"][1].packets_collided == 1

    # Case 2: Small propagation delay -> COLLISION AVOIDED VIA CARRIER SENSING
    pkts_small = [
        Packet(packet_id=0, source_node=0, generation_time=1.00),
        Packet(packet_id=1, source_node=1, generation_time=1.05),
    ]
    res_success = simulate_csma(
        num_nodes=2,
        sim_time=10.0,
        offered_load=0.0,
        packet_duration=1.0,
        propagation_delay=0.02,  # 0.02 < 0.05
        max_retries=0,
        seed=42,
        initial_packets=pkts_small,
    )
    log_suc = res_success["channel_log"]
    assert len(log_suc) == 2
    assert log_suc[0][3] == "SUCCESS"
    assert log_suc[1][3] == "SUCCESS"
    assert res_success["nodes"][0].packets_collided == 0
    assert res_success["nodes"][1].packets_collided == 0


def test_csma_multiple_waiting_nodes_collide_upon_channel_clear():
    """Verify that if 2+ nodes accumulate while the channel is busy, both transmit at channel clear and collide."""
    pkts = [
        Packet(packet_id=0, source_node=0, generation_time=1.0),   # Transmits [1.0, 2.0]
        Packet(packet_id=1, source_node=1, generation_time=1.3),   # Arrives while busy -> waits
        Packet(packet_id=2, source_node=2, generation_time=1.6),   # Arrives while busy -> waits
    ]

    prop_delay = 0.01
    result = simulate_csma(
        num_nodes=3,
        sim_time=10.0,
        offered_load=0.0,
        packet_duration=1.0,
        propagation_delay=prop_delay,
        max_retries=0,  # Isolate initial collision
        seed=42,
        initial_packets=pkts,
    )

    channel_log = result["channel_log"]
    assert len(channel_log) == 3

    # Node 0 succeeded
    assert channel_log[0][2] == 0
    assert channel_log[0][3] == "SUCCESS"

    # When Node 0 cleared at 2.01, both Node 1 and Node 2 transmitted simultaneously and collided
    clear_time = 2.0 + prop_delay
    assert math.isclose(channel_log[1][0], clear_time, abs_tol=1e-7)
    assert math.isclose(channel_log[2][0], clear_time, abs_tol=1e-7)
    assert channel_log[1][3] == "COLLISION"
    assert channel_log[2][3] == "COLLISION"


def test_csma_outperforms_aloha_empirically():
    """Verify empirically that CSMA outperforms Pure ALOHA and Slotted ALOHA under moderate load."""
    sim_time = 150.0
    num_nodes = 8
    offered_load = 0.6  # High enough to cause contention in ALOHA
    packet_duration = 1.0
    prop_delay = 0.005  # Realistic small propagation delay ratio a = 0.005
    seed = 42

    res_pure = simulate_pure_aloha(
        num_nodes=num_nodes,
        sim_time=sim_time,
        offered_load=offered_load,
        packet_duration=packet_duration,
        propagation_delay=prop_delay,
        max_retries=16,
        seed=seed,
    )

    res_slotted = simulate_slotted_aloha(
        num_nodes=num_nodes,
        sim_time=sim_time,
        offered_load=offered_load,
        packet_duration=packet_duration,
        propagation_delay=prop_delay,
        max_retries=16,
        seed=seed,
    )

    res_csma = simulate_csma(
        num_nodes=num_nodes,
        sim_time=sim_time,
        offered_load=offered_load,
        packet_duration=packet_duration,
        propagation_delay=prop_delay,
        max_retries=16,
        seed=seed,
    )

    tp_pure = throughput(res_pure["packets"], res_pure["channel_log"], sim_time)
    tp_slotted = throughput(res_slotted["packets"], res_slotted["channel_log"], sim_time)
    tp_csma = throughput(res_csma["packets"], res_csma["channel_log"], sim_time)

    # Empirically verify performance ordering: CSMA > Slotted ALOHA > Pure ALOHA
    assert tp_csma > tp_pure, f"CSMA throughput ({tp_csma}) should exceed Pure ALOHA ({tp_pure})"
    assert tp_csma > tp_slotted, f"CSMA throughput ({tp_csma}) should exceed Slotted ALOHA ({tp_slotted})"
