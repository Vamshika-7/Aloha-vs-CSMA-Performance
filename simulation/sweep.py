"""Comparison mode, load sweep, and multi-run statistical evaluation module."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .base import Packet, PacketStatus
from .csma import simulate_csma
from .metrics import calculate_all_metrics
from .randomness import stream_rng
from .pure_aloha import simulate_pure_aloha
from .slotted_aloha import simulate_slotted_aloha
from .traffic import generate_poisson_traffic


def clone_trace(trace: List[Packet]) -> List[Packet]:
    """Creates a deep, independent copy of a packet trace with reset lifecycle state."""
    return [
        Packet(
            packet_id=p.packet_id,
            source_node=p.source_node,
            generation_time=p.generation_time,
            attempt_time=None,
            successful_time=None,
            retry_count=0,
            status=PacketStatus.PENDING,
        )
        for p in trace
    ]


def run_comparison_simulation(
    num_nodes: int,
    sim_time: float,
    offered_load: float,
    packet_duration: float,
    propagation_delay: float,
    max_retries: int,
    seed: int,
    run_sim_cached_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Executes all three protocols against an IDENTICAL generated packet arrival trace.

    Args:
        num_nodes: Total nodes sharing the channel.
        sim_time: Maximum simulation duration (seconds).
        offered_load: Aggregate offered load G.
        packet_duration: Packet duration (seconds).
        propagation_delay: Propagation delay tau (seconds).
        max_retries: Maximum collision retries.
        seed: Random seed used to generate the shared traffic trace.
        run_sim_cached_fn: Optional cached simulator wrapper.

    Returns:
        dict containing:
            - 'trace': Original generated packet list.
            - 'results': {protocol_name: sim_result_dict}.
            - 'metrics': {protocol_name: metrics_dict}.
    """
    # 1. Generate one single shared traffic trace from traffic.py using the seed
    rng = stream_rng(seed, "traffic")
    shared_trace = generate_poisson_traffic(
        num_nodes=num_nodes,
        sim_time=sim_time,
        offered_load=offered_load,
        packet_duration=packet_duration,
        rng=rng,
    )

    protocols = ["Pure ALOHA", "Slotted ALOHA", "1-Persistent CSMA"]
    results = {}
    metrics = {}

    for proto in protocols:
        trace_copy = clone_trace(shared_trace)
        if run_sim_cached_fn is not None:
            res = run_sim_cached_fn(
                protocol=proto,
                num_nodes=num_nodes,
                sim_time=sim_time,
                offered_load=offered_load,
                packet_duration=packet_duration,
                propagation_delay=propagation_delay,
                max_retries=max_retries,
                seed=seed,
                traffic_trace=trace_copy,
                backoff_rng=stream_rng(seed, proto),
            )
        else:
            if proto == "Pure ALOHA":
                res = simulate_pure_aloha(
                    num_nodes=num_nodes,
                    sim_time=sim_time,
                    offered_load=offered_load,
                    packet_duration=packet_duration,
                    propagation_delay=propagation_delay,
                    max_retries=max_retries,
                    seed=seed,
                    traffic_trace=trace_copy,
                    backoff_rng=stream_rng(seed, proto),
                )
            elif proto == "Slotted ALOHA":
                res = simulate_slotted_aloha(
                    num_nodes=num_nodes,
                    sim_time=sim_time,
                    offered_load=offered_load,
                    packet_duration=packet_duration,
                    propagation_delay=propagation_delay,
                    max_retries=max_retries,
                    seed=seed,
                    traffic_trace=trace_copy,
                    backoff_rng=stream_rng(seed, proto),
                )
            else:
                res = simulate_csma(
                    num_nodes=num_nodes,
                    sim_time=sim_time,
                    offered_load=offered_load,
                    packet_duration=packet_duration,
                    propagation_delay=propagation_delay,
                    max_retries=max_retries,
                    seed=seed,
                    traffic_trace=trace_copy,
                    backoff_rng=stream_rng(seed, proto),
                )

        results[proto] = res
        metrics[proto] = calculate_all_metrics(
            packets=res["packets"],
            channel_log=res["channel_log"],
            sim_time=sim_time,
            packet_duration=packet_duration,
        )

    return {
        "trace": shared_trace,
        "results": results,
        "metrics": metrics,
    }


def compute_run_statistics(runs_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Calculates mean and sample standard deviation across multiple independent runs.

    Args:
        runs_metrics: List of metric dictionaries, one per run.

    Returns:
        dict mapping metric_name -> {"mean": float, "std": float}.
    """
    if not runs_metrics:
        return {}

    metric_names = list(runs_metrics[0].keys())
    stats: Dict[str, Dict[str, float]] = {}

    n = len(runs_metrics)
    ddof = 1 if n > 1 else 0

    for m in metric_names:
        values = [float(run[m]) for run in runs_metrics]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=ddof)) if n > 1 else 0.0
        stats[m] = {"mean": mean_val, "std": std_val}

    return stats


def run_load_sweep(
    load_points: List[float],
    protocols: List[str],
    num_nodes: int,
    sim_time: float,
    packet_duration: float,
    propagation_delay: float,
    max_retries: int,
    base_seed: int,
    num_runs: int,
    run_sim_cached_fn: Callable[..., Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    return_raw: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Executes a multi-run parameter sweep across offered loads for specified protocols.

    Args:
        load_points: List of offered load values G to evaluate.
        protocols: List of protocol names to simulate.
        num_nodes: Total nodes.
        sim_time: Simulation time per run.
        packet_duration: Packet duration.
        propagation_delay: Propagation delay.
        max_retries: Maximum retries.
        base_seed: Base random seed; run r uses (base_seed + r).
        num_runs: Number of independent simulation runs per (load, protocol).
        run_sim_cached_fn: Cached simulation executor.
        progress_callback: Optional callable(completed, total) to report progress.
        return_raw: If True, returns (aggregated_results, raw_results).

    Returns:
        List of aggregated record dictionaries with mean and std for all metrics,
        or tuple of (aggregated_results, raw_results) if return_raw=True.
    """
    load_points = [round(float(g), 4) for g in load_points]
    total_steps = len(load_points) * len(protocols) * num_runs
    completed_steps = 0

    aggregated_results: List[Dict[str, Any]] = []
    raw_results: List[Dict[str, Any]] = []

    for g in load_points:
        for proto in protocols:
            runs_metrics: List[Dict[str, float]] = []
            for r in range(num_runs):
                run_seed = base_seed + r
                res = run_sim_cached_fn(
                    protocol=proto,
                    num_nodes=num_nodes,
                    sim_time=sim_time,
                    offered_load=g,
                    packet_duration=packet_duration,
                    propagation_delay=propagation_delay,
                    max_retries=max_retries,
                    seed=run_seed,
                )
                m = calculate_all_metrics(
                    packets=res["packets"],
                    channel_log=res["channel_log"],
                    sim_time=sim_time,
                    packet_duration=packet_duration,
                )
                runs_metrics.append(m)
                raw_results.append({
                    "protocol": proto,
                    "offered_load_g": round(float(g), 4),
                    "load": round(float(g), 4),
                    "num_nodes": int(num_nodes),
                    "seed": run_seed,
                    "run_index": r,
                    "sim_time_s": float(sim_time),
                    "packet_duration_s": float(packet_duration),
                    "propagation_delay_s": float(propagation_delay),
                    "max_retries": int(max_retries),
                    "throughput_pkts_per_s": m["throughput"],
                    "normalized_throughput_s": m["normalized_throughput"],
                    "collision_rate": m["collision_rate"],
                    "collision_attempts": m["collision_attempts"],
                    "collision_affected_packets": m["collision_affected_packets"],
                    "success_rate": m["success_rate"],
                    "avg_delay_s": m["avg_delay"],
                    "avg_retransmissions": m["avg_retransmissions"],
                    "dropped_packets": m["dropped_packets"],
                    "pending_packets": m["pending_packets"],
                    "channel_utilization": m["channel_utilization"],
                    "channel_busy_fraction": m["channel_busy_fraction"],
                })

                completed_steps += 1
                if progress_callback:
                    progress_callback(completed_steps, total_steps)

            stats = compute_run_statistics(runs_metrics)
            record: Dict[str, Any] = {
                "load": round(float(g), 4),
                "protocol": proto,
                "runs": num_runs,
            }
            for metric_name, val_dict in stats.items():
                record[f"{metric_name}_mean"] = val_dict["mean"]
                record[f"{metric_name}_std"] = val_dict["std"]

            aggregated_results.append(record)

    if return_raw:
        return aggregated_results, raw_results
    return aggregated_results


def run_node_sweep(
    node_points: List[int],
    protocols: List[str],
    offered_load: float,
    sim_time: float,
    packet_duration: float,
    propagation_delay: float,
    max_retries: int,
    base_seed: int,
    num_runs: int,
    run_sim_cached_fn: Callable[..., Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    return_raw: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Executes a multi-run parameter sweep across node counts for specified protocols.

    Args:
        node_points: List of node counts to evaluate.
        protocols: List of protocol names to simulate.
        offered_load: Fixed offered load G.
        sim_time: Simulation time per run.
        packet_duration: Packet duration.
        propagation_delay: Propagation delay.
        max_retries: Maximum retries.
        base_seed: Base random seed; run r uses (base_seed + r).
        num_runs: Number of independent simulation runs per (nodes, protocol).
        run_sim_cached_fn: Cached simulation executor.
        progress_callback: Optional callable(completed, total) to report progress.
        return_raw: If True, returns (aggregated_results, raw_results).

    Returns:
        List of aggregated record dictionaries with mean and std for all metrics,
        or tuple of (aggregated_results, raw_results) if return_raw=True.
    """
    total_steps = len(node_points) * len(protocols) * num_runs
    completed_steps = 0

    aggregated_results: List[Dict[str, Any]] = []
    raw_results: List[Dict[str, Any]] = []

    for n_nodes in node_points:
        for proto in protocols:
            runs_metrics: List[Dict[str, float]] = []
            for r in range(num_runs):
                run_seed = base_seed + r
                res = run_sim_cached_fn(
                    protocol=proto,
                    num_nodes=n_nodes,
                    sim_time=sim_time,
                    offered_load=offered_load,
                    packet_duration=packet_duration,
                    propagation_delay=propagation_delay,
                    max_retries=max_retries,
                    seed=run_seed,
                )
                m = calculate_all_metrics(
                    packets=res["packets"],
                    channel_log=res["channel_log"],
                    sim_time=sim_time,
                    packet_duration=packet_duration,
                )
                runs_metrics.append(m)
                raw_results.append({
                    "protocol": proto,
                    "num_nodes": int(n_nodes),
                    "offered_load_g": round(float(offered_load), 4),
                    "seed": run_seed,
                    "run_index": r,
                    "sim_time_s": float(sim_time),
                    "packet_duration_s": float(packet_duration),
                    "propagation_delay_s": float(propagation_delay),
                    "max_retries": int(max_retries),
                    "throughput_pkts_per_s": m["throughput"],
                    "normalized_throughput_s": m["normalized_throughput"],
                    "collision_rate": m["collision_rate"],
                    "collision_attempts": m["collision_attempts"],
                    "collision_affected_packets": m["collision_affected_packets"],
                    "success_rate": m["success_rate"],
                    "avg_delay_s": m["avg_delay"],
                    "avg_retransmissions": m["avg_retransmissions"],
                    "dropped_packets": m["dropped_packets"],
                    "pending_packets": m["pending_packets"],
                    "channel_utilization": m["channel_utilization"],
                    "channel_busy_fraction": m["channel_busy_fraction"],
                })

                completed_steps += 1
                if progress_callback:
                    progress_callback(completed_steps, total_steps)

            stats = compute_run_statistics(runs_metrics)
            record: Dict[str, Any] = {
                "num_nodes": int(n_nodes),
                "protocol": proto,
                "runs": num_runs,
            }
            for metric_name, val_dict in stats.items():
                record[f"{metric_name}_mean"] = val_dict["mean"]
                record[f"{metric_name}_std"] = val_dict["std"]

            aggregated_results.append(record)

    if return_raw:
        return aggregated_results, raw_results
    return aggregated_results
