"""Slotted ALOHA simulation module.

In Slotted ALOHA:
- Time is partitioned into discrete slots of size equal to packet_duration.
- Transmissions may only start at exact slot boundaries (m * packet_duration).
- If a packet arrives mid-slot, it must wait until the next slot boundary.
- A collision occurs if and only if 2 or more nodes transmit in the same slot.
- A single transmission in a slot is successful.
- On collision, colliding nodes select a random future slot using exponential backoff.
- Packets exceeding max_retries are dropped.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from .base import Channel, Node, Packet, PacketStatus
from .traffic import generate_poisson_traffic


def get_earliest_slot(t: float, slot_duration: float) -> int:
    """Returns the earliest slot boundary index at or after timestamp t."""
    if t <= 0.0:
        return 0
    # Use rounding to avoid precision artifacts (e.g. 1.0000000000000002)
    ratio = round(t / slot_duration, 9)
    return int(np.ceil(ratio))


def simulate_slotted_aloha(
    num_nodes: int,
    sim_time: float,
    offered_load: float,
    packet_duration: float,
    propagation_delay: float = 0.0,
    max_retries: int = 16,
    seed: Optional[Union[int, np.random.Generator]] = None,
    initial_packets: Optional[List[Packet]] = None,
    traffic_trace: Optional[List[Packet]] = None,
) -> Dict[str, Any]:
    """Runs a discrete-slot simulation of the Slotted ALOHA MAC protocol.

    Args:
        num_nodes: Number of competing nodes.
        sim_time: Maximum simulation time window (seconds).
        offered_load: Dimensionless offered load G.
        packet_duration: Slot size and packet transmission duration (seconds).
        propagation_delay: Propagation delay (retained for uniform interface).
        max_retries: Maximum retransmission attempts before dropping a packet.
        seed: Random seed or initialized numpy Generator.
        initial_packets: Optional pre-generated list of packets for deterministic testing.
        traffic_trace: Optional alias for pre-generated packet arrival trace.

    Returns:
        dict containing:
            - 'packets': list of all Packet objects in their final state.
            - 'channel_log': list of (start_time, end_time, node_id, status) tuples.
            - 'nodes': list of Node objects.
    """
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    nodes = [Node(node_id=i) for i in range(num_nodes)]

    # Generate or inject packet traffic
    target_trace = traffic_trace if traffic_trace is not None else initial_packets
    if target_trace is not None:
        all_packets = target_trace
    else:
        all_packets = generate_poisson_traffic(
            num_nodes=num_nodes,
            sim_time=sim_time,
            offered_load=offered_load,
            packet_duration=packet_duration,
            rng=rng,
        )

    # Distribute packets to node queues, sorted by arrival time
    packets_by_node: Dict[int, List[Packet]] = {i: [] for i in range(num_nodes)}
    for pkt in all_packets:
        if pkt.source_node < num_nodes:
            packets_by_node[pkt.source_node].append(pkt)
            nodes[pkt.source_node].packets_generated += 1

    for i in range(num_nodes):
        packets_by_node[i].sort(key=lambda p: p.generation_time)

    # Next attempt slot for each node (None if idle with no scheduled transmission)
    scheduled_slot: Dict[int, Optional[int]] = {i: None for i in range(num_nodes)}

    channel = Channel()

    # Total integer slots in sim_time
    total_slots = int(np.floor(round(sim_time / packet_duration, 9)))

    for slot_idx in range(total_slots):
        slot_start = round(slot_idx * packet_duration, 9)
        slot_end = round((slot_idx + 1) * packet_duration, 9)

        # 1. Admit newly arrived packets (generation_time <= slot_start) into node queues
        for node_id in range(num_nodes):
            node = nodes[node_id]
            incoming = packets_by_node[node_id]
            while incoming and incoming[0].generation_time <= slot_start:
                arrived_pkt = incoming.pop(0)
                node.packet_queue.append(arrived_pkt)

            # If node has packets queued and no slot scheduled yet, schedule for this slot
            if node.packet_queue and scheduled_slot[node_id] is None:
                scheduled_slot[node_id] = slot_idx

        # 2. Identify all nodes attempting to transmit in this slot
        transmitting_nodes = [
            node_id
            for node_id in range(num_nodes)
            if scheduled_slot[node_id] == slot_idx and nodes[node_id].packet_queue
        ]

        if not transmitting_nodes:
            # Idle slot
            continue

        if len(transmitting_nodes) == 1:
            # Single transmission in the slot -> SUCCESS
            node_id = transmitting_nodes[0]
            node = nodes[node_id]
            pkt = node.packet_queue[0]

            pkt.attempt_time = slot_start
            pkt.successful_time = slot_end
            pkt.status = PacketStatus.SUCCESS
            node.packets_successful += 1
            node.total_delay += (pkt.successful_time - pkt.generation_time)

            channel.log(slot_start, slot_end, node_id, "SUCCESS")

            # Remove completed packet from queue
            node.packet_queue.pop(0)
            scheduled_slot[node_id] = None

            # If more packets already queued, schedule for next slot
            if node.packet_queue:
                earliest = get_earliest_slot(node.packet_queue[0].generation_time, packet_duration)
                scheduled_slot[node_id] = max(slot_idx + 1, earliest)

        else:
            # 2+ transmissions in the same slot -> COLLISION
            for node_id in transmitting_nodes:
                node = nodes[node_id]
                pkt = node.packet_queue[0]

                pkt.attempt_time = slot_start
                node.packets_collided += 1
                pkt.retry_count += 1
                node.retransmissions += 1

                channel.log(slot_start, slot_end, node_id, "COLLISION")

                if pkt.retry_count > max_retries:
                    # Drop packet after exceeding maximum retries
                    pkt.status = PacketStatus.DROPPED
                    node.packet_queue.pop(0)
                    scheduled_slot[node_id] = None

                    if node.packet_queue:
                        earliest = get_earliest_slot(
                            node.packet_queue[0].generation_time, packet_duration
                        )
                        scheduled_slot[node_id] = max(slot_idx + 1, earliest)
                else:
                    # Pick a future slot via exponential backoff: R in [1, 2^min(k, 10)]
                    k = min(pkt.retry_count, 10)
                    w = 2 ** k
                    backoff_slots = int(rng.integers(1, w + 1))
                    scheduled_slot[node_id] = slot_idx + backoff_slots

    return {
        "packets": all_packets,
        "channel_log": channel.events,
        "nodes": nodes,
    }


# Standard interface alias
simulate = simulate_slotted_aloha
