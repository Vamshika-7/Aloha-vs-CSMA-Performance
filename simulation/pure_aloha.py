"""Pure ALOHA simulation module.

In Pure ALOHA:
- Nodes transmit immediately whenever they have data (no channel sensing).
- Time is continuous.
- Any temporal overlap between two transmissions on the shared channel causes
  a collision for all overlapping transmissions.
- Colliding nodes use exponential random backoff before retransmitting.
- Packets exceeding max_retries are dropped.
"""

import heapq
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .base import Channel, Node, Packet, PacketStatus
from .randomness import stream_rng
from .traffic import generate_poisson_traffic


# Event priorities to ensure deterministic execution ordering at identical timestamps
PRIO_PACKET_ARRIVAL = 1
PRIO_RETRY_ATTEMPT = 2
PRIO_TRANSMIT_END = 3


def simulate_pure_aloha(
    num_nodes: int,
    sim_time: float,
    offered_load: float,
    packet_duration: float,
    propagation_delay: float = 0.0,
    max_retries: int = 16,
    seed: Optional[Union[int, np.random.Generator]] = None,
    initial_packets: Optional[List[Packet]] = None,
    traffic_trace: Optional[List[Packet]] = None,
    traffic_rng: Optional[np.random.Generator] = None,
    backoff_rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Runs a discrete-event simulation of the Pure ALOHA MAC protocol.

    Args:
        num_nodes: Number of competing nodes.
        sim_time: Maximum simulation time window.
        offered_load: Dimensionless offered load G.
        packet_duration: Transmission duration of one packet (seconds).
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
    rng = stream_rng(seed, "Pure ALOHA", backoff_rng)

    # Initialize nodes
    nodes = [Node(node_id=i) for i in range(num_nodes)]

    # Generate or inject packet traffic
    target_trace = traffic_trace if traffic_trace is not None else initial_packets

    if target_trace is not None:
        all_packets = target_trace
    else:
        traffic_rng = stream_rng(seed, "traffic", traffic_rng)
        all_packets = generate_poisson_traffic(
            num_nodes=num_nodes,
            sim_time=sim_time,
            offered_load=offered_load,
            packet_duration=packet_duration,
            rng=traffic_rng,
        )
    # Track node state: 'IDLE', 'TRANSMITTING', or 'BACKING_OFF'
    node_state = {i: "IDLE" for i in range(num_nodes)}

    # Event queue: min-heap of (event_time, priority, event_id, event_type, data)
    event_queue: List[Tuple[float, int, int, str, Any]] = []
    event_counter = 0

    def push_event(event_time: float, priority: int, event_type: str, data: Any) -> None:
        nonlocal event_counter
        heapq.heappush(event_queue, (event_time, priority, event_counter, event_type, data))
        event_counter += 1

    # Schedule initial packet arrivals
    for pkt in all_packets:
        if pkt.source_node < num_nodes:
            push_event(pkt.generation_time, PRIO_PACKET_ARRIVAL, "ARRIVAL", pkt)
            nodes[pkt.source_node].packets_generated += 1

    channel = Channel()

    # Track active and historical transmissions to detect overlaps
    # Each entry: dict(trans_id, node_id, packet, start_time, end_time, collided)
    transmissions: List[Dict[str, Any]] = []
    trans_counter = 0

    while event_queue:
        event_time, priority, _, event_type, data = heapq.heappop(event_queue)

        if event_type == "ARRIVAL":
            pkt: Packet = data
            node = nodes[pkt.source_node]
            node.packet_queue.append(pkt)

            # If node is IDLE and arrival time is within simulation window, transmit immediately
            if node_state[node.node_id] == "IDLE" and event_time <= sim_time:
                # Start transmitting head packet
                node_state[node.node_id] = "TRANSMITTING"
                head_pkt = node.packet_queue[0]
                head_pkt.attempt_time = event_time

                t_start = event_time
                t_end = event_time + packet_duration
                trans_id = trans_counter
                trans_counter += 1

                trans_record = {
                    "trans_id": trans_id,
                    "node_id": node.node_id,
                    "packet": head_pkt,
                    "start_time": t_start,
                    "end_time": t_end,
                    "collided": False,
                }

                # Check overlap with any previous transmission that overlaps with [t_start, t_end]
                for prev in transmissions:
                    # Overlap condition: max(s1, s2) < min(e1, e2)
                    if max(prev["start_time"], t_start) < min(prev["end_time"], t_end):
                        prev["collided"] = True
                        trans_record["collided"] = True

                transmissions.append(trans_record)
                push_event(t_end, PRIO_TRANSMIT_END, "TRANSMIT_END", trans_record)

        elif event_type == "RETRY":
            node_id: int = data
            node = nodes[node_id]

            if not node.packet_queue or event_time > sim_time:
                node_state[node_id] = "IDLE"
                continue

            node_state[node_id] = "TRANSMITTING"
            head_pkt = node.packet_queue[0]
            head_pkt.attempt_time = event_time

            t_start = event_time
            t_end = event_time + packet_duration
            trans_id = trans_counter
            trans_counter += 1

            trans_record = {
                "trans_id": trans_id,
                "node_id": node.node_id,
                "packet": head_pkt,
                "start_time": t_start,
                "end_time": t_end,
                "collided": False,
            }

            # Check overlap with all known transmissions
            for prev in transmissions:
                if max(prev["start_time"], t_start) < min(prev["end_time"], t_end):
                    prev["collided"] = True
                    trans_record["collided"] = True

            transmissions.append(trans_record)
            push_event(t_end, PRIO_TRANSMIT_END, "TRANSMIT_END", trans_record)

        elif event_type == "TRANSMIT_END":
            record: Dict[str, Any] = data
            node = nodes[record["node_id"]]
            pkt = record["packet"]

            is_collision = record["collided"]

            if is_collision:
                # Log collision event on the channel
                channel.log(record["start_time"], record["end_time"], node.node_id, "COLLISION")
                node.packets_collided += 1
                pkt.retry_count += 1
                node.retransmissions += 1

                if pkt.retry_count > max_retries:
                    pkt.status = PacketStatus.DROPPED
                    node.packet_queue.pop(0)
                    node_state[node.node_id] = "IDLE"
                    # If queue has pending packets, transmit next
                    if node.packet_queue and event_time <= sim_time:
                        push_event(event_time, PRIO_RETRY_ATTEMPT, "RETRY", node.node_id)
                else:
                    # Exponential random backoff
                    k = min(pkt.retry_count, 10)
                    max_backoff = (2 ** k) * packet_duration
                    backoff_delay = rng.uniform(0.0, max_backoff)
                    retry_time = event_time + backoff_delay
                    node_state[node.node_id] = "BACKING_OFF"
                    if retry_time <= sim_time:
                        push_event(retry_time, PRIO_RETRY_ATTEMPT, "RETRY", node.node_id)
                    else:
                        # Beyond simulation time, node will not retry
                        node_state[node.node_id] = "IDLE"
            else:
                # Successful transmission
                channel.log(record["start_time"], record["end_time"], node.node_id, "SUCCESS")
                pkt.status = PacketStatus.SUCCESS
                pkt.successful_time = record["end_time"]
                node.packets_successful += 1
                node.total_delay += (pkt.successful_time - pkt.generation_time)

                node.packet_queue.pop(0)
                node_state[node.node_id] = "IDLE"
                # If queue has pending packets, transmit next immediately
                if node.packet_queue and event_time <= sim_time:
                    push_event(event_time, PRIO_RETRY_ATTEMPT, "RETRY", node.node_id)

    return {
        "packets": all_packets,
        "channel_log": channel.events,
        "nodes": nodes,
    }


# Standard interface alias
simulate = simulate_pure_aloha
