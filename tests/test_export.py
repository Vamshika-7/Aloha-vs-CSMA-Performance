"""Unit tests for self-describing CSV export DataFrame generators."""

import pandas as pd
import pytest

from simulation.base import Packet, PacketStatus
from simulation.consistency import run_consistency_checks
from simulation.export import (
    dataframe_to_csv_bytes,
    export_comparison_dataframe,
    export_consistency_dataframe,
    export_metrics_dataframe,
    export_packets_dataframe,
    export_sweep_aggregated_dataframe,
    export_sweep_raw_dataframe,
)
from simulation.metrics import calculate_all_metrics
from simulation.pure_aloha import simulate_pure_aloha
from simulation.sweep import run_comparison_simulation, run_load_sweep


@pytest.fixture
def base_params():
    return {
        "protocol": "Pure ALOHA",
        "num_nodes": 5,
        "sim_time": 100.0,
        "offered_load": 0.5,
        "packet_duration": 1.0,
        "propagation_delay": 0.01,
        "max_retries": 5,
        "seed": 42,
    }


def test_export_packets_dataframe(base_params):
    """Validates packet export contains full parameter provenance and lifecycle columns without NaNs."""
    packets = [
        Packet(1, 0, 1.23, 1.23, 2.23, 0, PacketStatus.SUCCESS),
        Packet(2, 1, 2.34, 2.34, None, 5, PacketStatus.DROPPED),
    ]
    df = export_packets_dataframe(packets, base_params)

    expected_cols = [
        "protocol",
        "num_nodes",
        "sim_time_s",
        "offered_load_g",
        "packet_duration_s",
        "propagation_delay_s",
        "max_retries",
        "seed",
        "packet_id",
        "source_node",
        "generation_time_s",
        "attempt_time_s",
        "successful_time_s",
        "retry_count",
        "status",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    assert len(df) == 2
    assert not df[expected_cols].isna().any().any()
    assert df.iloc[0]["protocol"] == "Pure ALOHA"
    assert df.iloc[0]["num_nodes"] == 5
    assert df.iloc[0]["status"] == "SUCCESS"
    assert df.iloc[1]["status"] == "DROPPED"


def test_export_metrics_dataframe(base_params):
    """Validates metrics export contains all key metrics including utilization and busy fraction."""
    sim_res = simulate_pure_aloha(**{k: v for k, v in base_params.items() if k != "protocol"})
    metrics = calculate_all_metrics(
        packets=sim_res["packets"],
        channel_log=sim_res["channel_log"],
        sim_time=base_params["sim_time"],
        packet_duration=base_params["packet_duration"],
    )

    df = export_metrics_dataframe(metrics, base_params)
    assert len(df) == 1

    required_cols = [
        "protocol",
        "num_nodes",
        "sim_time_s",
        "offered_load_g",
        "packet_duration_s",
        "propagation_delay_s",
        "max_retries",
        "seed",
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
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert not df[required_cols].isna().any().any()
    assert df.iloc[0]["channel_busy_fraction"] >= df.iloc[0]["channel_utilization"]


def test_export_comparison_dataframe(base_params):
    """Validates comparison export produces side-by-side protocol comparison table."""
    comp_res = run_comparison_simulation(
        num_nodes=base_params["num_nodes"],
        sim_time=base_params["sim_time"],
        offered_load=base_params["offered_load"],
        packet_duration=base_params["packet_duration"],
        propagation_delay=base_params["propagation_delay"],
        max_retries=base_params["max_retries"],
        seed=base_params["seed"],
    )

    df = export_comparison_dataframe(comp_res, base_params)

    assert "metric" in df.columns
    assert "pure_aloha" in df.columns
    assert "slotted_aloha" in df.columns
    assert "csma_1_persistent" in df.columns
    assert "num_nodes" in df.columns
    assert "offered_load_g" in df.columns

    assert len(df) >= 10
    assert not df.isna().any().any()


def test_export_consistency_dataframe(base_params):
    """Validates consistency checks export produces self-describing pass/fail table."""
    sim_res = simulate_pure_aloha(**{k: v for k, v in base_params.items() if k != "protocol"})
    checks = run_consistency_checks(sim_res, base_params)

    df = export_consistency_dataframe(checks, base_params)
    assert len(df) == len(checks)

    required_cols = ["protocol", "num_nodes", "check_name", "passed", "status", "details"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert not df[required_cols].isna().any().any()
    assert (df["status"] == "PASS").all()


def test_export_sweep_raw_and_aggregated_dataframe(base_params):
    """Validates raw sweep export for a fixed config (2 loads x 2 runs x 3 protocols = 12 raw rows)."""
    load_points = [0.2, 0.5]
    protocols = ["Pure ALOHA", "Slotted ALOHA", "1-Persistent CSMA"]
    num_runs = 2

    # Fixed config: 2 loads x 2 runs x 3 protocols = 12 raw rows
    agg_results, raw_results = run_load_sweep(
        load_points=load_points,
        protocols=protocols,
        num_nodes=3,
        sim_time=30.0,
        packet_duration=1.0,
        propagation_delay=0.01,
        max_retries=3,
        base_seed=100,
        num_runs=num_runs,
        run_sim_cached_fn=lambda **kwargs: (
            simulate_pure_aloha(**{k: v for k, v in kwargs.items() if k != "protocol"})
            if kwargs["protocol"] == "Pure ALOHA"
            else run_comparison_simulation(
                num_nodes=kwargs["num_nodes"],
                sim_time=kwargs["sim_time"],
                offered_load=kwargs["offered_load"],
                packet_duration=kwargs["packet_duration"],
                propagation_delay=kwargs["propagation_delay"],
                max_retries=kwargs["max_retries"],
                seed=kwargs["seed"],
            )["results"][kwargs["protocol"]]
        ),
        return_raw=True,
    )

    df_raw = export_sweep_raw_dataframe(raw_results)
    assert len(df_raw) == 12, f"Expected 12 raw rows (2x2x3), got {len(df_raw)}"

    required_raw_cols = [
        "protocol",
        "offered_load_g",
        "num_nodes",
        "seed",
        "run_index",
        "throughput_pkts_per_s",
        "collision_rate",
        "channel_utilization",
        "channel_busy_fraction",
    ]
    for col in required_raw_cols:
        assert col in df_raw.columns, f"Missing raw column: {col}"
    assert not df_raw[required_raw_cols].isna().any().any()

    # Aggregated export check
    df_agg = export_sweep_aggregated_dataframe(agg_results, base_params)
    assert len(df_agg) == 6  # 2 loads x 3 protocols
    assert not df_agg["throughput_mean"].isna().any()


def test_dataframe_to_csv_bytes():
    """Validates CSV serialization produces non-empty UTF-8 bytes with header."""
    df = pd.DataFrame([{"col1": "val1", "col2": 42}])
    csv_bytes = dataframe_to_csv_bytes(df)
    assert isinstance(csv_bytes, bytes)
    decoded = csv_bytes.decode("utf-8")
    assert "col1,col2" in decoded
    assert "val1,42" in decoded
