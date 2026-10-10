"""Performance evaluation metrics for ALOHA and CSMA simulations.

All metric functions are pure functions accepting:
    packets: List of Packet objects in their final simulation state.
    channel_log: List of (start_time, end_time, node_id, status) tuples.
    sim_time: Total simulation duration in seconds.

Each ratio metric explicitly documents the exact denominator used.
"""

from typing import Any, Dict, List, Optional, Tuple
from .base import Packet, PacketStatus


def throughput(
    packets: List[Packet],
    channel_log: List[Tuple[float, float, int, str]],
    sim_time: float,
) -> float:
    """Calculates raw throughput: successfully delivered packets per second.

    Formula:
        throughput = (number of successfully delivered packets) / sim_time

    Denominator:
        sim_time (total simulation duration in seconds).
        Returns 0.0 if sim_time <= 0.
    """
    if sim_time <= 0.0:
        return 0.0
    successful_count = sum(1 for p in packets if p.status == PacketStatus.SUCCESS)
    return successful_count / sim_time


def normalized_throughput(
    packets: List[Packet],
    channel_log: List[Tuple[float, float, int, str]],
    sim_time: float,
    packet_duration: Optional[float] = None,
) -> float:
    """Calculates dimensionless normalized throughput (channel efficiency S).

    Formula:
        normalized_throughput = (successful_packets * packet_duration) / sim_time
                              = throughput * packet_duration

    Denominator:
        sim_time / packet_duration (total number of packet transmission slots
        available over the simulation window).
        Returns 0.0 if sim_time <= 0 or packet_duration <= 0.
    """
    if sim_time <= 0.0:
        return 0.0

    if packet_duration is None:
        if channel_log:
            packet_duration = channel_log[0][1] - channel_log[0][0]
        else:
            packet_duration = 1.0

    if packet_duration <= 0.0:
        return 0.0

    raw_tp = throughput(packets, channel_log, sim_time)
    return raw_tp * packet_duration


def collision_rate(
    packets: List[Packet],
    channel_log: List[Tuple[float, float, int, str]],
    sim_time: float,
) -> float:
    """Calculates the transmission collision rate.

    Formula:
        collision_rate = (number of COLLISION events in channel_log) / (total attempts in channel_log)

    Denominator:
        len(channel_log) (total number of transmission attempts logged on the channel).
        Returns 0.0 if len(channel_log) == 0.
    """
    total_transmissions = len(channel_log)
    if total_transmissions == 0:
        return 0.0
    collision_count = sum(1 for entry in channel_log if entry[3] == "COLLISION")
    return collision_count / total_transmissions


def collision_attempts(
    packets: List[Packet],
    channel_log: List[Tuple[float, float, int, str]],
    sim_time: float,
) -> int:
    """Counts collided transmission attempts, including repeated attempts."""
    return sum(1 for entry in channel_log if entry[3] == "COLLISION")


def collision_affected_packets(
    packets: List[Packet],
    channel_log: List[Tuple[float, float, int, str]],
    sim_time: float,
) -> int:
    """Counts unique generated packets that experienced at least one collision."""
    return sum(1 for packet in packets if packet.retry_count > 0)


def pending_packets(
    packets: List[Packet],
    channel_log: List[Tuple[float, float, int, str]],
    sim_time: float,
) -> int:
    """Counts packets still pending when the simulation window closes."""
    return sum(1 for packet in packets if packet.status == PacketStatus.PENDING)


def success_rate(
    packets: List[Packet],
    channel_log: List[Tuple[float, float, int, str]],
    sim_time: float,
) -> float:
    """Calculates the packet delivery success rate.

    Formula:
        success_rate = (number of successfully delivered packets) / (total generated packets)

    Denominator:
        len(packets) (total number of unique packets generated across all nodes).
        Returns 0.0 if len(packets) == 0.
    """
    total_packets = len(packets)
    if total_packets == 0:
        return 0.0
    successful_count = sum(1 for p in packets if p.status == PacketStatus.SUCCESS)
    return successful_count / total_packets


def avg_delay(
    packets: List[Packet],
    channel_log: List[Tuple[float, float, int, str]],
    sim_time: float,
) -> float:
    """Calculates average packet latency (delay) for successfully delivered packets.

    Formula:
        avg_delay = sum(successful_time - generation_time) / (number of successful packets)

    Denominator:
        Number of successfully delivered packets (len([p for p in packets if p.status == SUCCESS])).
        Returns 0.0 if no packets were successfully delivered.
    """
    successful_delays = [
        (p.successful_time - p.generation_time)
        for p in packets
        if p.status == PacketStatus.SUCCESS and p.successful_time is not None
    ]
    if not successful_delays:
        return 0.0
    return sum(successful_delays) / len(successful_delays)


def avg_retransmissions(
    packets: List[Packet],
    channel_log: List[Tuple[float, float, int, str]],
    sim_time: float,
) -> float:
    """Calculates the average number of retransmissions per generated packet.

    Formula:
        avg_retransmissions = sum(packet.retry_count) / (total generated packets)

    Denominator:
        len(packets) (total number of unique packets generated).
        Returns 0.0 if len(packets) == 0.
    """
    total_packets = len(packets)
    if total_packets == 0:
        return 0.0
    total_retries = sum(p.retry_count for p in packets)
    return total_retries / total_packets


def dropped_packets(
    packets: List[Packet],
    channel_log: List[Tuple[float, float, int, str]],
    sim_time: float,
) -> int:
    """Counts the total number of packets dropped due to exceeding max_retries.

    Formula:
        dropped_packets = count(p for p in packets if p.status == DROPPED)
    """
    return sum(1 for p in packets if p.status == PacketStatus.DROPPED)


def channel_utilization(
    packets: List[Packet],
    channel_log: List[Tuple[float, float, int, str]],
    sim_time: float,
) -> float:
    """Calculates the fraction of simulation time the channel was busy carrying transmissions.

    Overlapping transmission intervals are merged to avoid double-counting channel occupancy.

    Formula:
        channel_utilization = (total busy duration of merged transmission intervals) / sim_time

    Denominator:
        sim_time (total simulation duration in seconds).
        Returns 0.0 if sim_time <= 0.
    """
    if sim_time <= 0.0 or not channel_log:
        return 0.0

    # Extract [start_time, end_time] intervals from SUCCESS entries only,
    # clamped to [0, sim_time]. COLLISION intervals are excluded so that
    # channel utilization reflects only time spent carrying useful data.
    raw_intervals = []
    for entry in channel_log:
        if entry[3] != "SUCCESS":
            continue
        s = max(0.0, entry[0])
        e = min(sim_time, entry[1])
        if e > s:
            raw_intervals.append((s, e))

    if not raw_intervals:
        return 0.0

    # Sort intervals by start time
    raw_intervals.sort(key=lambda iv: iv[0])

    # Merge overlapping intervals
    merged = [raw_intervals[0]]
    for current_start, current_end in raw_intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            # Overlapping or contiguous interval -> extend
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    total_busy_time = sum(e - s for s, e in merged)
    return total_busy_time / sim_time


def calculate_all_metrics(
    packets: List[Packet],
    channel_log: List[Tuple[float, float, int, str]],
    sim_time: float,
    packet_duration: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculates and returns a dictionary of all performance metrics.

    Returns:
        dict with keys:
            - 'throughput'
            - 'normalized_throughput'
            - 'collision_rate'
            - 'collision_attempts'
            - 'collision_affected_packets'
            - 'success_rate'
            - 'avg_delay'
            - 'avg_retransmissions'
            - 'dropped_packets'
            - 'pending_packets'
            - 'channel_utilization'
    """
    return {
        "throughput": throughput(packets, channel_log, sim_time),
        "normalized_throughput": normalized_throughput(
            packets, channel_log, sim_time, packet_duration=packet_duration
        ),
        "collision_rate": collision_rate(packets, channel_log, sim_time),
        "collision_attempts": collision_attempts(packets, channel_log, sim_time),
        "collision_affected_packets": collision_affected_packets(
            packets, channel_log, sim_time
        ),
        "success_rate": success_rate(packets, channel_log, sim_time),
        "avg_delay": avg_delay(packets, channel_log, sim_time),
        "avg_retransmissions": avg_retransmissions(packets, channel_log, sim_time),
        "dropped_packets": dropped_packets(packets, channel_log, sim_time),
        "pending_packets": pending_packets(packets, channel_log, sim_time),
        "channel_utilization": channel_utilization(packets, channel_log, sim_time),
    }
