"""1-Persistent CSMA simulation module.

In 1-Persistent CSMA:
- A node with a packet to transmit senses the shared channel.
- If the channel is sensed IDLE: transmit immediately.
- If the channel is sensed BUSY: continuously sense until the channel becomes IDLE,
  then transmit immediately with probability 1.
- Propagation delay window:
  When a node transmits, its signal takes propagation_delay (tau) to reach
  other nodes. During [t_start, t_start + tau), another node sensing the channel
  perceives it as IDLE and may also transmit, producing a collision.
- Collisions also occur when multiple nodes accumulate in the waiting state
  during an ongoing transmission and all transmit simultaneously the instant
  the channel becomes idle.
- Colliding nodes use exponential random backoff before re-sensing the channel.
- Packets exceeding max_retries are dropped.
"""

import heapq
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from .base import Channel, Node, Packet, PacketStatus
from .traffic import generate_poisson_traffic


# Event priorities for deterministic execution
PRIO_PACKET_ARRIVAL = 1
PRIO_CHANNEL_CLEAR = 2
PRIO_RETRY_SENSE = 3
PRIO_TRANSMIT_END = 4


def are_transmissions_colliding(s1: float, s2: float, tau: float) -> bool:
    """Determines whether two transmissions collide given their start times and propagation delay tau.

    Two transmissions collide if either was initiated before the other's signal
    could propagate to it, i.e., |s1 - s2| < tau (or s1 == s2 when tau == 0).
    """
    diff = abs(s1 - s2)
    if tau <= 0.0:
        return diff < 1e-9
    return diff < (tau - 1e-9) or diff < 1e-9


def simulate_csma(
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
    """Runs a discrete-event simulation of the 1-Persistent CSMA MAC protocol.

    Args:
        num_nodes: Number of competing nodes.
        sim_time: Maximum simulation time window (seconds).
        offered_load: Dimensionless offered load G.
        packet_duration: Transmission duration of one packet (seconds).
        propagation_delay: One-way propagation delay tau across the channel (seconds).
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

    # Track node state: 'IDLE', 'TRANSMITTING', 'WAITING_BUSY', 'BACKING_OFF'
    node_state = {i: "IDLE" for i in range(num_nodes)}
    waiting_nodes: Set[int] = set()

    channel = Channel()

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

    # Active and completed transmission records
    # Each record: dict(trans_id, node_id, packet, start_time, end_time, clear_time, collided)
    transmissions: List[Dict[str, Any]] = []
    trans_counter = 0

    def is_channel_sensed_busy(sensing_node_id: int, current_time: float) -> bool:
        """Checks if any active transmission is currently sensed as busy by sensing_node_id."""
        for rec in transmissions:
            # If the transmission was by another node
            if rec["node_id"] != sensing_node_id:
                s = rec["start_time"]
                c = rec["clear_time"]
                # Sensed busy starting when leading edge arrives: s + tau
                # Sensed busy ending when trailing edge clears: e + tau (= clear_time)
                # Note: A transmission starting exactly at current_time has not propagated yet (if tau > 0)
                # and even if tau == 0, two simultaneous decisions to transmit do not sense each other.
                if s < current_time:
                    sensed_start = s + propagation_delay
                    sensed_end = c
                    if sensed_start <= current_time + 1e-9 and current_time < sensed_end - 1e-9:
                        return True
        return False

    def start_transmission(transmitting_node_id: int, start_time: float) -> Dict[str, Any]:
        nonlocal trans_counter
        node = nodes[transmitting_node_id]
        head_pkt = node.packet_queue[0]
        head_pkt.attempt_time = start_time

        t_start = start_time
        t_end = start_time + packet_duration
        t_clear = t_end + propagation_delay

        rec = {
            "trans_id": trans_counter,
            "node_id": transmitting_node_id,
            "packet": head_pkt,
            "start_time": t_start,
            "end_time": t_end,
            "clear_time": t_clear,
            "collided": False,
        }
        trans_counter += 1

        # Check collision with all other transmissions within propagation delay window
        for prev in transmissions:
            if prev["node_id"] != transmitting_node_id:
                if are_transmissions_colliding(prev["start_time"], t_start, propagation_delay):
                    prev["collided"] = True
                    rec["collided"] = True

        transmissions.append(rec)
        node_state[transmitting_node_id] = "TRANSMITTING"

        # Schedule transmission completion event
        push_event(t_end, PRIO_TRANSMIT_END, "TRANSMIT_END", rec)
        # Schedule channel clear check when trailing edge clears
        push_event(t_clear, PRIO_CHANNEL_CLEAR, "CHANNEL_CLEAR_CHECK", None)

        return rec

    while event_queue:
        event_time, priority, _, event_type, data = heapq.heappop(event_queue)

        if event_type == "ARRIVAL":
            pkt: Packet = data
            node = nodes[pkt.source_node]
            node.packet_queue.append(pkt)

            # If node is IDLE and arrival is within simulation time, sense channel immediately
            if node_state[node.node_id] == "IDLE" and event_time <= sim_time:
                busy = is_channel_sensed_busy(node.node_id, event_time)
                if not busy:
                    start_transmission(node.node_id, event_time)
                else:
                    # Channel busy -> 1-persistent: wait until channel becomes idle
                    node_state[node.node_id] = "WAITING_BUSY"
                    waiting_nodes.add(node.node_id)

        elif event_type == "SENSE_AND_TRANSMIT":
            node_id: int = data
            node = nodes[node_id]

            if not node.packet_queue or event_time > sim_time:
                node_state[node_id] = "IDLE"
                waiting_nodes.discard(node_id)
                continue

            busy = is_channel_sensed_busy(node_id, event_time)
            if not busy:
                waiting_nodes.discard(node_id)
                start_transmission(node_id, event_time)
            else:
                node_state[node_id] = "WAITING_BUSY"
                waiting_nodes.add(node_id)

        elif event_type == "CHANNEL_CLEAR_CHECK":
            # Check if channel is truly idle now (no ongoing transmissions sensed)
            # Find any transmission whose trailing edge has not yet cleared
            active_signals = [rec for rec in transmissions if event_time < rec["clear_time"] - 1e-9]

            if not active_signals and waiting_nodes:
                # Channel is now completely idle!
                # All 1-persistent waiting nodes sense idle and transmit simultaneously
                ready_to_transmit = sorted(list(waiting_nodes))
                waiting_nodes.clear()

                if event_time <= sim_time:
                    for w_node_id in ready_to_transmit:
                        if nodes[w_node_id].packet_queue:
                            start_transmission(w_node_id, event_time)
                        else:
                            node_state[w_node_id] = "IDLE"

        elif event_type == "TRANSMIT_END":
            record: Dict[str, Any] = data
            node = nodes[record["node_id"]]
            pkt = record["packet"]

            is_collision = record["collided"]

            if is_collision:
                channel.log(record["start_time"], record["end_time"], node.node_id, "COLLISION")
                node.packets_collided += 1
                pkt.retry_count += 1
                node.retransmissions += 1

                if pkt.retry_count > max_retries:
                    pkt.status = PacketStatus.DROPPED
                    node.packet_queue.pop(0)
                    node_state[node.node_id] = "IDLE"
                    # If queue has next packet, sense channel for it
                    if node.packet_queue and event_time <= sim_time:
                        push_event(event_time, PRIO_RETRY_SENSE, "SENSE_AND_TRANSMIT", node.node_id)
                else:
                    # Exponential random backoff
                    k = min(pkt.retry_count, 10)
                    w = 2 ** k
                    backoff_delay = rng.uniform(0.0, w * packet_duration)
                    retry_time = event_time + backoff_delay
                    node_state[node.node_id] = "BACKING_OFF"
                    if retry_time <= sim_time:
                        push_event(retry_time, PRIO_RETRY_SENSE, "SENSE_AND_TRANSMIT", node.node_id)
                    else:
                        node_state[node.node_id] = "IDLE"
            else:
                # SUCCESS
                channel.log(record["start_time"], record["end_time"], node.node_id, "SUCCESS")
                pkt.status = PacketStatus.SUCCESS
                pkt.successful_time = record["end_time"]
                node.packets_successful += 1
                node.total_delay += (pkt.successful_time - pkt.generation_time)

                node.packet_queue.pop(0)
                node_state[node.node_id] = "IDLE"
                # If queue has next packet, sense channel for it
                if node.packet_queue and event_time <= sim_time:
                    push_event(event_time, PRIO_RETRY_SENSE, "SENSE_AND_TRANSMIT", node.node_id)

    return {
        "packets": all_packets,
        "channel_log": channel.events,
        "nodes": nodes,
    }


# Standard interface alias
simulate = simulate_csma
