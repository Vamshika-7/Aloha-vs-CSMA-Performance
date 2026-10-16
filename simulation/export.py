"""Export module providing self-describing DataFrames for CSV downloads.

Produces clean, structured, and reproducible tabular data with full parameter
provenance and standardized column names with SI/engineering units.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from .base import Packet, PacketStatus


def _extract_params_dict(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts standardized parameter metadata columns with explicit units."""
    return {
        "protocol": str(params.get("protocol", "N/A")),
        "num_nodes": int(params.get("num_nodes", 0)),
        "sim_time_s": float(params.get("sim_time", 0.0)),
        "offered_load_g": float(params.get("offered_load", 0.0)),
        "packet_duration_s": float(params.get("packet_duration", 1.0)),
        "propagation_delay_s": float(params.get("propagation_delay", 0.0)),
        "max_retries": int(params.get("max_retries", 0)),
        "seed": int(params.get("seed", 0)),
    }


def export_packets_dataframe(packets: List[Packet], params: Dict[str, Any]) -> pd.DataFrame:
    """Exports packet lifecycle records as a self-describing DataFrame.

    Args:
        packets: List of Packet objects from simulation output.
        params: Simulation parameter configuration.

    Returns:
        pd.DataFrame containing parameter columns and packet records.
    """
    param_meta = _extract_params_dict(params)

    rows: List[Dict[str, Any]] = []
    for pkt in packets:
        row = dict(param_meta)
        row.update({
            "packet_id": int(pkt.packet_id),
            "source_node": int(pkt.source_node),
            "generation_time_s": float(pkt.generation_time),
            "attempt_time_s": float(pkt.attempt_time) if pkt.attempt_time is not None else -1.0,
            "successful_time_s": float(pkt.successful_time) if pkt.successful_time is not None else -1.0,
            "retry_count": int(pkt.retry_count),
            "status": str(pkt.status.value if hasattr(pkt.status, "value") else pkt.status),
        })
        rows.append(row)

    if not rows:
        columns = list(param_meta.keys()) + [
            "packet_id",
            "source_node",
            "generation_time_s",
            "attempt_time_s",
            "successful_time_s",
            "retry_count",
            "status",
        ]
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows)


def export_metrics_dataframe(metrics: Dict[str, Any], params: Dict[str, Any]) -> pd.DataFrame:
    """Exports summary performance metrics as a self-describing DataFrame.

    Args:
        metrics: Dictionary of evaluated metrics from calculate_all_metrics.
        params: Simulation parameter configuration.

    Returns:
        pd.DataFrame with 1 row containing parameters and all metric fields with units.
    """
    row = _extract_params_dict(params)
    row.update({
        "throughput_pkts_per_s": float(metrics.get("throughput", 0.0)),
        "normalized_throughput_s": float(metrics.get("normalized_throughput", 0.0)),
        "collision_rate": float(metrics.get("collision_rate", 0.0)),
        "collision_attempts": int(metrics.get("collision_attempts", 0)),
        "collision_affected_packets": int(metrics.get("collision_affected_packets", 0)),
        "success_rate": float(metrics.get("success_rate", 0.0)),
        "avg_delay_s": float(metrics.get("avg_delay", 0.0)),
        "avg_retransmissions": float(metrics.get("avg_retransmissions", 0.0)),
        "dropped_packets": int(metrics.get("dropped_packets", 0)),
        "pending_packets": int(metrics.get("pending_packets", 0)),
        "channel_utilization": float(metrics.get("channel_utilization", 0.0)),
        "channel_busy_fraction": float(metrics.get("channel_busy_fraction", 0.0)),
    })
    return pd.DataFrame([row])


def export_comparison_dataframe(
    comparison_result: Dict[str, Any],
    params: Dict[str, Any],
) -> pd.DataFrame:
    """Exports side-by-side multi-protocol comparison metrics as a self-describing DataFrame.

    Layout: One row per metric, one column per protocol, plus shared parameter columns.

    Args:
        comparison_result: Dictionary returned by run_comparison_simulation.
        params: Simulation parameter configuration.

    Returns:
        pd.DataFrame formatted with metric rows and protocol columns.
    """
    param_meta = _extract_params_dict(params)
    # Exclude protocol from shared metadata since each column is a protocol
    shared_params = {k: v for k, v in param_meta.items() if k != "protocol"}

    m_pure = comparison_result["metrics"]["Pure ALOHA"]
    m_slotted = comparison_result["metrics"]["Slotted ALOHA"]
    m_csma = comparison_result["metrics"]["1-Persistent CSMA"]

    metric_definitions = [
        ("Throughput [pkts/s]", "throughput"),
        ("Normalized Throughput (S)", "normalized_throughput"),
        ("Collision Rate", "collision_rate"),
        ("Collision Attempts", "collision_attempts"),
        ("Collision Affected Packets", "collision_affected_packets"),
        ("Success Rate", "success_rate"),
        ("Average Delay [s]", "avg_delay"),
        ("Average Retransmissions", "avg_retransmissions"),
        ("Dropped Packets", "dropped_packets"),
        ("Pending Packets", "pending_packets"),
        ("Channel Utilization (Useful)", "channel_utilization"),
        ("Channel Busy Fraction (Total)", "channel_busy_fraction"),
    ]

    rows = []
    for label, key in metric_definitions:
        row = {
            "metric": label,
            "pure_aloha": float(m_pure.get(key, 0.0)),
            "slotted_aloha": float(m_slotted.get(key, 0.0)),
            "csma_1_persistent": float(m_csma.get(key, 0.0)),
        }
        row.update(shared_params)
        rows.append(row)

    return pd.DataFrame(rows)


def export_sweep_raw_dataframe(raw_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Exports raw per-run results from parameter sweeps.

    Args:
        raw_records: List of dictionaries, each containing single-run results.

    Returns:
        pd.DataFrame with one row per protocol, sweep point (load or node count), and seed.
    """
    if not raw_records:
        return pd.DataFrame()

    df = pd.DataFrame(raw_records)
    # Ensure consistent ordering of columns if present
    preferred_order = [
        "protocol",
        "offered_load_g",
        "num_nodes",
        "seed",
        "run_index",
        "sim_time_s",
        "packet_duration_s",
        "propagation_delay_s",
        "max_retries",
        "throughput_pkts_per_s",
        "normalized_throughput_s",
        "collision_rate",
        "collision_attempts",
        "collision_affected_packets",
        "success_rate",
        "avg_delay_s",
        "avg_retransmissions",
        "dropped_packets",
        "pending_packets",
        "channel_utilization",
        "channel_busy_fraction",
    ]
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    return df[cols]


def export_sweep_aggregated_dataframe(
    sweep_data: Union[pd.DataFrame, List[Dict[str, Any]]],
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Exports aggregated mean and std table with parameter metadata columns.

    Args:
        sweep_data: Aggregated sweep DataFrame or list of dicts.
        params: Optional base parameter configuration.

    Returns:
        pd.DataFrame containing parameters and statistical summary columns.
    """
    df = pd.DataFrame(sweep_data) if not isinstance(sweep_data, pd.DataFrame) else sweep_data.copy()
    if df.empty:
        return df

    if params:
        param_meta = _extract_params_dict(params)
        # Add parameter columns if not already present
        for k, v in param_meta.items():
            if k not in df.columns and k not in ["load", "offered_load_g"]:
                df[k] = v

    return df


def export_consistency_dataframe(
    consistency_results: List[Tuple[str, bool, str]],
    params: Dict[str, Any],
) -> pd.DataFrame:
    """Exports consistency check outcomes as a self-describing pass/fail DataFrame.

    Args:
        consistency_results: List of (check_name, passed, details) tuples.
        params: Simulation parameter configuration.

    Returns:
        pd.DataFrame containing parameter columns and check results.
    """
    param_meta = _extract_params_dict(params)

    rows: List[Dict[str, Any]] = []
    for name, passed, details in consistency_results:
        row = dict(param_meta)
        row.update({
            "check_name": str(name),
            "passed": bool(passed),
            "status": "PASS" if passed else "FAIL",
            "details": str(details),
        })
        rows.append(row)

    if not rows:
        columns = list(param_meta.keys()) + ["check_name", "passed", "status", "details"]
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encodes a DataFrame into UTF-8 CSV bytes ready for st.download_button."""
    return df.to_csv(index=False).encode("utf-8")
