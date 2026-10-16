"""ALOHA vs CSMA Performance Analyzer simulation package."""

from .base import Channel, ChannelEventStatus, Node, Packet, PacketStatus
from .csma import simulate as simulate_csma
from .csma import simulate_csma as csma_sim
from .consistency import run_consistency_checks, validate_channel_log_physics
from .metrics import (
    avg_delay,
    avg_retransmissions,
    calculate_all_metrics,
    channel_busy_fraction,
    channel_utilization,
    collision_rate,
    dropped_packets,
    normalized_throughput,
    success_rate,
    throughput,
)
from .pure_aloha import simulate as simulate_pure_aloha
from .pure_aloha import simulate_pure_aloha as pure_aloha_sim
from .slotted_aloha import simulate as simulate_slotted_aloha
from .slotted_aloha import simulate_slotted_aloha as slotted_aloha_sim
from .sweep import (
    clone_trace,
    compute_run_statistics,
    run_comparison_simulation,
    run_load_sweep,
    run_node_sweep,
)
from .traffic import generate_poisson_traffic

from .theory import (
    compute_throughput_peaks,
    pure_aloha_throughput,
    slotted_aloha_throughput,
)
from .export import (
    dataframe_to_csv_bytes,
    export_comparison_dataframe,
    export_consistency_dataframe,
    export_metrics_dataframe,
    export_packets_dataframe,
    export_sweep_aggregated_dataframe,
    export_sweep_raw_dataframe,
)

__all__ = [
    "Channel",
    "ChannelEventStatus",
    "Node",
    "Packet",
    "PacketStatus",
    "generate_poisson_traffic",
    "simulate_pure_aloha",
    "pure_aloha_sim",
    "simulate_slotted_aloha",
    "slotted_aloha_sim",
    "simulate_csma",
    "csma_sim",
    "throughput",
    "normalized_throughput",
    "collision_rate",
    "success_rate",
    "avg_delay",
    "avg_retransmissions",
    "dropped_packets",
    "channel_utilization",
    "channel_busy_fraction",
    "calculate_all_metrics",
    "run_consistency_checks",
    "clone_trace",
    "run_comparison_simulation",
    "compute_run_statistics",
    "run_load_sweep",
    "run_node_sweep",
    "pure_aloha_throughput",
    "slotted_aloha_throughput",
    "compute_throughput_peaks",
    "export_packets_dataframe",
    "export_metrics_dataframe",
    "export_comparison_dataframe",
    "export_sweep_raw_dataframe",
    "export_sweep_aggregated_dataframe",
    "export_consistency_dataframe",
    "dataframe_to_csv_bytes",
]

