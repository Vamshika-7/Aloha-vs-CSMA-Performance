"""Traffic generation module for ALOHA and CSMA simulations."""

from typing import List, Optional
import numpy as np
from .base import Packet, PacketStatus


def generate_poisson_traffic(
    num_nodes: int,
    sim_time: float,
    offered_load: float,
    packet_duration: float,
    rng: np.random.Generator,
) -> List[Packet]:
    """Generates synthetic packet traffic using independent Poisson processes per node.

    Offered load G is defined as the average number of packet arrivals per
    packet transmission duration across the entire shared channel.
    Total arrival rate lambda = G / packet_duration.
    Per-node arrival rate lambda_node = lambda / num_nodes = G / (num_nodes * packet_duration).

    Args:
        num_nodes: Total number of contending nodes sharing the channel.
        sim_time: Maximum simulation duration in seconds.
        offered_load: Dimensionless offered load G (expected arrivals per packet duration).
        packet_duration: Transmission duration of a single packet in seconds.
        rng: Seeded numpy random Generator instance.

    Returns:
        List of Packet objects sorted chronologically by generation_time.
    """
    if num_nodes <= 0 or sim_time <= 0 or offered_load <= 0 or packet_duration <= 0:
        return []

    # Total arrival rate (packets/sec) across all nodes
    total_rate = offered_load / packet_duration
    per_node_rate = total_rate / num_nodes

    all_packets: List[Packet] = []

    for node_id in range(num_nodes):
        current_time = 0.0
        while True:
            # Draw exponential inter-arrival time: delta_t ~ Exp(per_node_rate)
            inter_arrival = rng.exponential(1.0 / per_node_rate)
            current_time += inter_arrival
            if current_time > sim_time:
                break

            all_packets.append(
                Packet(
                    packet_id=-1,  # Assigned sequentially after sorting
                    source_node=node_id,
                    generation_time=current_time,
                    status=PacketStatus.PENDING,
                )
            )

    # Sort all generated packets chronologically
    all_packets.sort(key=lambda p: (p.generation_time, p.source_node))

    # Assign unique 0-indexed packet IDs
    for idx, pkt in enumerate(all_packets):
        pkt.packet_id = idx

    return all_packets
