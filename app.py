"""Streamlit Web Application: ALOHA vs CSMA Performance Analyzer.

Supports:
- Single-simulation mode
- Side-by-side comparison mode on identical traffic traces
- Load sweep and node sweep with multi-run statistical analysis (mean ± std dev)
- State management via st.session_state & simulation caching via @st.cache_data
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from simulation.base import Packet, PacketStatus
from simulation.csma import simulate_csma
from simulation.metrics import calculate_all_metrics
from simulation.pure_aloha import simulate_pure_aloha
from simulation.slotted_aloha import simulate_slotted_aloha
from simulation.sweep import (
    clone_trace,
    compute_run_statistics,
    run_comparison_simulation,
    run_load_sweep,
    run_node_sweep,
)
from visualization.sweep_plots import (
    plot_load_vs_collision_rate,
    plot_load_vs_delay,
    plot_load_vs_retransmissions,
    plot_load_vs_throughput,
    plot_nodes_vs_throughput,
)
from visualization.timeline import create_timeline_figure

# Page configuration
st.set_page_config(
    page_title="ALOHA vs CSMA Performance Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# CACHED SIMULATION RUNNERS
# =====================================================================

@st.cache_data
def run_simulation_cached(
    protocol: str,
    num_nodes: int,
    sim_time: float,
    offered_load: float,
    packet_duration: float,
    propagation_delay: float,
    max_retries: int,
    seed: int,
) -> Dict[str, Any]:
    """Cached wrapper executing protocol simulation keyed on all input parameters."""
    if protocol == "Pure ALOHA":
        return simulate_pure_aloha(
            num_nodes=num_nodes,
            sim_time=sim_time,
            offered_load=offered_load,
            packet_duration=packet_duration,
            propagation_delay=propagation_delay,
            max_retries=max_retries,
            seed=seed,
        )
    elif protocol == "Slotted ALOHA":
        return simulate_slotted_aloha(
            num_nodes=num_nodes,
            sim_time=sim_time,
            offered_load=offered_load,
            packet_duration=packet_duration,
            propagation_delay=propagation_delay,
            max_retries=max_retries,
            seed=seed,
        )
    elif protocol == "1-Persistent CSMA":
        return simulate_csma(
            num_nodes=num_nodes,
            sim_time=sim_time,
            offered_load=offered_load,
            packet_duration=packet_duration,
            propagation_delay=propagation_delay,
            max_retries=max_retries,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


@st.cache_data
def run_comparison_cached(
    num_nodes: int,
    sim_time: float,
    offered_load: float,
    packet_duration: float,
    propagation_delay: float,
    max_retries: int,
    seed: int,
) -> Dict[str, Any]:
    """Cached wrapper executing all 3 protocols on an identical shared packet trace."""
    return run_comparison_simulation(
        num_nodes=num_nodes,
        sim_time=sim_time,
        offered_load=offered_load,
        packet_duration=packet_duration,
        propagation_delay=propagation_delay,
        max_retries=max_retries,
        seed=seed,
    )


# =====================================================================
# SIDEBAR CONFIGURATION PANEL
# =====================================================================

st.sidebar.title("📡 Configuration")

app_mode = st.sidebar.radio(
    "Mode",
    options=["Single Simulation", "Compare All Protocols", "Load & Node Sweep"],
    index=0,
    help="Select single protocol inspection, multi-protocol comparison on identical traffic, or parameter sweep.",
)

st.sidebar.markdown("---")

# Common Parameter Inputs
if app_mode == "Single Simulation":
    protocol = st.sidebar.selectbox(
        "Protocol",
        options=["Pure ALOHA", "Slotted ALOHA", "1-Persistent CSMA"],
        index=0,
        help="Select the MAC-layer random-access protocol to simulate.",
    )
else:
    protocol = "Pure ALOHA"

num_nodes = st.sidebar.number_input(
    "Number of Nodes",
    min_value=1,
    max_value=200,
    value=10,
    step=1,
    help="Total competing nodes sharing the channel.",
)

sim_time = st.sidebar.number_input(
    "Simulation Duration (seconds)",
    min_value=1.0,
    max_value=10000.0,
    value=1000.0,
    step=50.0,
    help="Total simulation time window.",
)

offered_load = st.sidebar.number_input(
    "Offered Load (G)",
    min_value=0.001,
    max_value=20.0,
    value=0.1,
    step=0.05,
    format="%.3f",
    help="Average packet attempts per packet transmission duration.",
)

packet_duration = st.sidebar.number_input(
    "Packet Transmission Duration (s)",
    min_value=0.01,
    max_value=100.0,
    value=1.0,
    step=0.1,
    format="%.2f",
    help="Time taken to transmit one complete packet over the channel.",
)

propagation_delay = st.sidebar.number_input(
    "Propagation Delay (s)",
    min_value=0.0,
    max_value=10.0,
    value=0.01,
    step=0.005,
    format="%.4f",
    help="Physical one-way signal propagation delay across the shared channel.",
)

max_retries = st.sidebar.number_input(
    "Maximum Retransmission Attempts",
    min_value=0,
    max_value=32,
    value=10,
    step=1,
    help="Maximum collision retries before dropping a packet.",
)

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=2147483647,
    value=42,
    step=1,
    help="Seed for reproducible pseudorandom number generation.",
)

# Mode-specific Action Buttons
if app_mode == "Single Simulation":
    run_button = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)
elif app_mode == "Compare All Protocols":
    st.sidebar.caption("All 3 protocols will be tested against the exact same generated arrival trace.")
    run_comp_button = st.sidebar.button("Run Comparison", type="primary", use_container_width=True)
else:
    st.sidebar.caption("Configure and launch parameter sweeps directly in the Load Analysis tab.")


# =====================================================================
# STATE MANAGEMENT
# =====================================================================

single_params = {
    "protocol": protocol,
    "num_nodes": int(num_nodes),
    "sim_time": float(sim_time),
    "offered_load": float(offered_load),
    "packet_duration": float(packet_duration),
    "propagation_delay": float(propagation_delay),
    "max_retries": int(max_retries),
    "seed": int(seed),
}

# 1. Initialize Single-Sim state on first load, or update when Single Sim run button clicked
if "simulation_result" not in st.session_state or (app_mode == "Single Simulation" and run_button):
    with st.spinner("Running simulation..."):
        st.session_state["simulation_result"] = run_simulation_cached(**single_params)
        st.session_state["last_run_params"] = single_params

# 2. Run Comparison when comparison run button clicked
if app_mode == "Compare All Protocols" and 'run_comp_button' in locals() and run_comp_button:
    with st.spinner("Running comparison on identical traffic trace..."):
        st.session_state["comparison_result"] = run_comparison_cached(
            num_nodes=int(num_nodes),
            sim_time=float(sim_time),
            offered_load=float(offered_load),
            packet_duration=float(packet_duration),
            propagation_delay=float(propagation_delay),
            max_retries=int(max_retries),
            seed=int(seed),
        )
        st.session_state["comparison_params"] = single_params

sim_result = st.session_state["simulation_result"]
active_params = st.session_state["last_run_params"]

packets = sim_result["packets"]
channel_log = sim_result["channel_log"]
active_sim_time = active_params["sim_time"]
active_duration = active_params["packet_duration"]

# Calculate metrics strictly via metrics.py (never recomputed inline)
metrics = calculate_all_metrics(
    packets=packets,
    channel_log=channel_log,
    sim_time=active_sim_time,
    packet_duration=active_duration,
)

# Header
st.title("ALOHA vs CSMA Performance Analyzer")
st.caption(
    "Empirical MAC-layer random-access evaluation: Pure ALOHA, Slotted ALOHA, and 1-Persistent CSMA"
)

# Tabs
tab_dashboard, tab_simulation, tab_timeline, tab_comparison, tab_load_analysis, tab_theory, tab_about = st.tabs(
    [
        "📊 Dashboard",
        "📋 Simulation",
        "📈 Timeline",
        "⚔️ Comparison",
        "📉 Load Analysis",
        "📖 Theory",
        "ℹ️ About",
    ]
)

# =====================================================================
# TAB 1: DASHBOARD
# =====================================================================
with tab_dashboard:
    st.subheader(f"Performance Overview ({active_params['protocol']})")

    # Counts
    total_generated = len(packets)
    successful_count = sum(1 for p in packets if p.status == PacketStatus.SUCCESS)
    collision_count = sum(1 for e in channel_log if e[3] == "COLLISION")
    dropped_count = metrics["dropped_packets"]

    # Row 1: Traffic & Delivery Totals
    r1_col1, r1_col2, r1_col3, r1_col4 = st.columns(4)
    with r1_col1:
        st.metric(
            label="Packets Generated",
            value=f"{total_generated:,}",
            help="Total unique packets generated by all nodes",
        )
    with r1_col2:
        st.metric(
            label="Successful Packets",
            value=f"{successful_count:,}",
            delta=f"{metrics['success_rate'] * 100:.1f}% delivery",
            help="Packets successfully received without unresolved collision",
        )
    with r1_col3:
        st.metric(
            label="Collisions",
            value=f"{collision_count:,}",
            help="Total transmission attempts that suffered collision",
        )
    with r1_col4:
        st.metric(
            label="Dropped Packets",
            value=f"{dropped_count:,}",
            help="Packets discarded after exceeding max retransmission attempts",
        )

    st.markdown("---")

    # Row 2: Throughput, Efficiency, and Utilization
    r2_col1, r2_col2, r2_col3 = st.columns(3)
    with r2_col1:
        st.metric(
            label="Throughput",
            value=f"{metrics['throughput']:.3f} pkts/s",
            help="Successfully delivered packets divided by simulation duration",
        )
    with r2_col2:
        st.metric(
            label="Normalized Throughput (S)",
            value=f"{metrics['normalized_throughput']:.3f}",
            help="Channel efficiency S = throughput * packet_duration",
        )
    with r2_col3:
        st.metric(
            label="Channel Utilization",
            value=f"{metrics['channel_utilization'] * 100:.1f}%",
            help="Fraction of simulation duration channel was actively occupied",
        )

    st.markdown("---")

    # Row 3: Rates & Quality of Service
    r3_col1, r3_col2, r3_col3 = st.columns(3)
    with r3_col1:
        st.metric(
            label="Collision Rate",
            value=f"{metrics['collision_rate'] * 100:.1f}%",
            help="Collision events divided by total transmission attempts in channel log",
        )
    with r3_col2:
        st.metric(
            label="Average Delay",
            value=f"{metrics['avg_delay']:.3f} s",
            help="Mean latency from generation to completion for successful packets",
        )
    with r3_col3:
        st.metric(
            label="Average Retransmissions",
            value=f"{metrics['avg_retransmissions']:.2f}",
            help="Total retransmissions divided by total generated packets",
        )


# =====================================================================
# TAB 2: SIMULATION
# =====================================================================
with tab_simulation:
    st.subheader("Packet-Level Transmission Table")

    st.info(
        f"**Active Configuration:** Protocol: `{active_params['protocol']}` | "
        f"Nodes: `{active_params['num_nodes']}` | "
        f"Sim Time: `{active_params['sim_time']}s` | "
        f"Offered Load G: `{active_params['offered_load']}` | "
        f"Packet Duration: `{active_params['packet_duration']}s` | "
        f"Prop Delay: `{active_params['propagation_delay']}s` | "
        f"Max Retries: `{active_params['max_retries']}` | "
        f"Seed: `{active_params['seed']}`"
    )

    if packets:
        table_rows = []
        for p in packets:
            succ_str = (
                "DROPPED"
                if p.status == PacketStatus.DROPPED
                else (f"{p.successful_time:.4f}" if p.successful_time is not None else "-")
            )
            table_rows.append(
                {
                    "Packet ID": p.packet_id,
                    "Source Node": f"Node {p.source_node}",
                    "Generation Time (s)": round(p.generation_time, 4),
                    "Successful Time (s)": succ_str,
                    "Retry Count": p.retry_count,
                    "Status": p.status,
                }
            )

        df_packets = pd.DataFrame(table_rows)

        col_filter1, col_filter2 = st.columns([2, 1])
        with col_filter1:
            st.caption(f"Showing all {len(df_packets):,} simulated packets. Click any column header to sort.")
        with col_filter2:
            status_filter = st.multiselect(
                "Filter by Status",
                options=["SUCCESS", "DROPPED", "PENDING"],
                default=["SUCCESS", "DROPPED", "PENDING"],
            )

        if status_filter:
            df_filtered = df_packets[df_packets["Status"].isin(status_filter)]
        else:
            df_filtered = df_packets

        st.dataframe(
            df_filtered,
            use_container_width=True,
            hide_index=True,
            height=500,
        )
    else:
        st.warning("No packets were generated in this simulation run. Try increasing Offered Load or Simulation Duration.")


# =====================================================================
# TAB 3: TIMELINE
# =====================================================================
with tab_timeline:
    st.subheader("Channel Activity Timeline")
    st.caption("Visualizes channel access, successful transmissions, and collisions per node over time.")

    time_window = st.slider(
        "Display Window Cap (seconds)",
        min_value=10.0,
        max_value=min(200.0, max(10.0, float(active_sim_time))),
        value=min(50.0, float(active_sim_time)),
        step=5.0,
        help="Caps horizontal time span to keep rendering crisp and responsive.",
    )

    fig, timeline_note = create_timeline_figure(
        channel_log=channel_log,
        num_nodes=active_params["num_nodes"],
        sim_time=active_sim_time,
        max_time_window=time_window,
    )

    if timeline_note:
        st.info(timeline_note)

    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# TAB 4: COMPARISON MODE
# =====================================================================
with tab_comparison:
    st.subheader("Side-by-Side Protocol Comparison")
    st.caption(
        "Evaluates Pure ALOHA, Slotted ALOHA, and 1-Persistent CSMA against an **identical packet arrival trace**."
    )

    # If comparison has not run yet, provide an inline execution button
    if "comparison_result" not in st.session_state:
        st.info("No comparison results generated yet for this session.")
        if st.button("Run Comparison Now (Identical Traffic)", type="primary"):
            with st.spinner("Generating shared traffic trace and running all 3 protocols..."):
                st.session_state["comparison_result"] = run_comparison_cached(
                    num_nodes=int(num_nodes),
                    sim_time=float(sim_time),
                    offered_load=float(offered_load),
                    packet_duration=float(packet_duration),
                    propagation_delay=float(propagation_delay),
                    max_retries=int(max_retries),
                    seed=int(seed),
                )
                st.session_state["comparison_params"] = single_params
                st.rerun()

    if "comparison_result" in st.session_state:
        comp_res = st.session_state["comparison_result"]
        c_params = st.session_state.get("comparison_params", single_params)
        m_pure = comp_res["metrics"]["Pure ALOHA"]
        m_slotted = comp_res["metrics"]["Slotted ALOHA"]
        m_csma = comp_res["metrics"]["1-Persistent CSMA"]

        # Objective parameter badge
        st.success(
            f"**Evaluation Parameters:** Seed: `{c_params['seed']}` (Shared Trace) | "
            f"Nodes: `{c_params['num_nodes']}` | "
            f"Duration: `{c_params['sim_time']}s` | "
            f"Offered Load G: `{c_params['offered_load']}` | "
            f"Packets Generated: `{len(comp_res['trace']):,}`"
        )

        # Build comparison table
        comp_table_data = [
            {
                "Metric": "Channel Sensing",
                "Pure ALOHA": "No",
                "Slotted ALOHA": "No",
                "1-Persistent CSMA": "Yes",
            },
            {
                "Metric": "Time Synchronization",
                "Pure ALOHA": "No",
                "Slotted ALOHA": "Required (Slots)",
                "1-Persistent CSMA": "No",
            },
            {
                "Metric": "Collision Rate",
                "Pure ALOHA": f"{m_pure['collision_rate'] * 100:.1f}%",
                "Slotted ALOHA": f"{m_slotted['collision_rate'] * 100:.1f}%",
                "1-Persistent CSMA": f"{m_csma['collision_rate'] * 100:.1f}%",
            },
            {
                "Metric": "Throughput",
                "Pure ALOHA": f"{m_pure['throughput']:.3f} pkts/s",
                "Slotted ALOHA": f"{m_slotted['throughput']:.3f} pkts/s",
                "1-Persistent CSMA": f"{m_csma['throughput']:.3f} pkts/s",
            },
            {
                "Metric": "Normalized Throughput (S)",
                "Pure ALOHA": f"{m_pure['normalized_throughput']:.3f}",
                "Slotted ALOHA": f"{m_slotted['normalized_throughput']:.3f}",
                "1-Persistent CSMA": f"{m_csma['normalized_throughput']:.3f}",
            },
            {
                "Metric": "Average Delay",
                "Pure ALOHA": f"{m_pure['avg_delay']:.3f} s",
                "Slotted ALOHA": f"{m_slotted['avg_delay']:.3f} s",
                "1-Persistent CSMA": f"{m_csma['avg_delay']:.3f} s",
            },
            {
                "Metric": "Average Retransmissions",
                "Pure ALOHA": f"{m_pure['avg_retransmissions']:.2f}",
                "Slotted ALOHA": f"{m_slotted['avg_retransmissions']:.2f}",
                "1-Persistent CSMA": f"{m_csma['avg_retransmissions']:.2f}",
            },
            {
                "Metric": "Successful Packets",
                "Pure ALOHA": f"{sum(1 for p in comp_res['results']['Pure ALOHA']['packets'] if p.status == PacketStatus.SUCCESS):,}",
                "Slotted ALOHA": f"{sum(1 for p in comp_res['results']['Slotted ALOHA']['packets'] if p.status == PacketStatus.SUCCESS):,}",
                "1-Persistent CSMA": f"{sum(1 for p in comp_res['results']['1-Persistent CSMA']['packets'] if p.status == PacketStatus.SUCCESS):,}",
            },
            {
                "Metric": "Dropped Packets",
                "Pure ALOHA": f"{m_pure['dropped_packets']:,}",
                "Slotted ALOHA": f"{m_slotted['dropped_packets']:,}",
                "1-Persistent CSMA": f"{m_csma['dropped_packets']:,}",
            },
            {
                "Metric": "Channel Utilization",
                "Pure ALOHA": f"{m_pure['channel_utilization'] * 100:.1f}%",
                "Slotted ALOHA": f"{m_slotted['channel_utilization'] * 100:.1f}%",
                "1-Persistent CSMA": f"{m_csma['channel_utilization'] * 100:.1f}%",
            },
        ]

        df_comp = pd.DataFrame(comp_table_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        if st.button("Re-run Comparison on Current Parameters"):
            with st.spinner("Re-running comparison..."):
                st.session_state["comparison_result"] = run_comparison_cached(
                    num_nodes=int(num_nodes),
                    sim_time=float(sim_time),
                    offered_load=float(offered_load),
                    packet_duration=float(packet_duration),
                    propagation_delay=float(propagation_delay),
                    max_retries=int(max_retries),
                    seed=int(seed),
                )
                st.session_state["comparison_params"] = single_params
                st.rerun()


# =====================================================================
# TAB 5: LOAD ANALYSIS (SWEEPS)
# =====================================================================
with tab_load_analysis:
    st.subheader("Parametric Load and Scaling Analysis")
    st.markdown("Executes multi-run parameter sweeps to evaluate protocol scaling across loads and node densities.")

    # Sweep configuration controls
    sweep_preset = st.selectbox(
        "Sweep Configuration Preset",
        options=["Quick Sweep (5 loads × 3 runs)", "Detailed Custom Sweep"],
        index=0,
        help="Quick Sweep uses a coarser step for fast execution; Custom Sweep allows full parameter range tuning.",
    )

    sw_col1, sw_col2, sw_col3, sw_col4 = st.columns(4)

    if sweep_preset.startswith("Quick"):
        sw_min_load = 0.1
        sw_max_load = 1.0
        sw_step = 0.225  # ~5 load points: 0.1, 0.325, 0.55, 0.775, 1.0
        sw_runs = 3
        sw_sim_time = 100.0  # Kept reasonable for responsiveness
    else:
        with sw_col1:
            sw_min_load = st.number_input("Minimum Load", min_value=0.05, max_value=5.0, value=0.1, step=0.05)
        with sw_col2:
            sw_max_load = st.number_input("Maximum Load", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
        with sw_col3:
            sw_step = st.number_input("Load Step", min_value=0.05, max_value=2.0, value=0.2, step=0.05)
        with sw_col4:
            sw_runs = st.number_input("Runs per (Load, Protocol)", min_value=1, max_value=10, value=3, step=1)
        sw_sim_time = st.number_input("Simulation Duration per Run (s)", min_value=20.0, max_value=1000.0, value=100.0, step=20.0)

    # Compute load points
    if sw_step <= 0:
        sw_step = 0.1
    load_points = list(np.round(np.arange(sw_min_load, sw_max_load + 1e-6, sw_step), 3))
    total_load_sims = len(load_points) * 3 * int(sw_runs)

    st.write(
        f"**Configured Load Points ({len(load_points)}):** `{load_points}` | "
        f"Runs per point: `{sw_runs}` | "
        f"Total Simulations: `{total_load_sims}`"
    )

    if total_load_sims > 150:
        st.warning(
            f"⚠️ High iteration count: {total_load_sims} simulations configured. "
            "This may take noticeable compute time. Consider increasing Load Step or reducing Runs."
        )

    run_sweep_button = st.button("🚀 Run Offered Load Sweep (Plots 1–4)", type="primary")

    if run_sweep_button:
        prog_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(completed: int, total: int):
            frac = completed / total
            prog_bar.progress(frac)
            status_text.text(f"Running simulation {completed} of {total} ({frac * 100:.1f}% complete)...")

        protocols_to_sweep = ["Pure ALOHA", "Slotted ALOHA", "1-Persistent CSMA"]
        sweep_data = run_load_sweep(
            load_points=load_points,
            protocols=protocols_to_sweep,
            num_nodes=int(num_nodes),
            sim_time=float(sw_sim_time),
            packet_duration=float(packet_duration),
            propagation_delay=float(propagation_delay),
            max_retries=int(max_retries),
            base_seed=int(seed),
            num_runs=int(sw_runs),
            run_sim_cached_fn=run_simulation_cached,
            progress_callback=update_progress,
        )

        prog_bar.empty()
        status_text.empty()
        st.session_state["load_sweep_df"] = pd.DataFrame(sweep_data)
        st.session_state["load_sweep_sim_time"] = sw_sim_time

    # Display Load Sweep Plots if data available
    if "load_sweep_df" in st.session_state:
        df_load_sweep = st.session_state["load_sweep_df"]

        st.markdown("### Offered Load Sweep Results (Plots 1–4)")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.plotly_chart(plot_load_vs_throughput(df_load_sweep), use_container_width=True)
            st.plotly_chart(plot_load_vs_delay(df_load_sweep), use_container_width=True)
        with p_col2:
            st.plotly_chart(plot_load_vs_collision_rate(df_load_sweep), use_container_width=True)
            st.plotly_chart(plot_load_vs_retransmissions(df_load_sweep), use_container_width=True)

        with st.expander("View Raw Load Sweep Data"):
            st.dataframe(df_load_sweep, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Plot 5: Number of Nodes vs Throughput
    st.markdown("### Plot 5: Number of Nodes vs Throughput")
    st.caption("Sweeps the number of contending nodes at a fixed offered load to evaluate scalability.")

    n_col1, n_col2, n_col3 = st.columns(3)
    with n_col1:
        fixed_sweep_load = st.number_input("Fixed Offered Load G", min_value=0.05, max_value=5.0, value=0.5, step=0.05)
    with n_col2:
        node_runs = st.number_input("Runs per Node Count", min_value=1, max_value=10, value=3, step=1)
    with n_col3:
        node_sim_time = st.number_input("Duration per Run (s)", min_value=20.0, max_value=1000.0, value=100.0, step=20.0)

    node_points = [2, 4, 8, 16, 24]
    total_node_sims = len(node_points) * 3 * int(node_runs)
    st.write(f"**Contending Node Points ({len(node_points)}):** `{node_points}` | Total Simulations: `{total_node_sims}`")

    run_node_sweep_btn = st.button("🚀 Run Node Count Sweep (Plot 5)")

    if run_node_sweep_btn:
        n_prog_bar = st.progress(0.0)
        n_status_text = st.empty()

        def update_node_progress(completed: int, total: int):
            frac = completed / total
            n_prog_bar.progress(frac)
            n_status_text.text(f"Running node simulation {completed} of {total} ({frac * 100:.1f}% complete)...")

        protocols_to_sweep = ["Pure ALOHA", "Slotted ALOHA", "1-Persistent CSMA"]
        node_sweep_data = run_node_sweep(
            node_points=node_points,
            protocols=protocols_to_sweep,
            offered_load=float(fixed_sweep_load),
            sim_time=float(node_sim_time),
            packet_duration=float(packet_duration),
            propagation_delay=float(propagation_delay),
            max_retries=int(max_retries),
            base_seed=int(seed),
            num_runs=int(node_runs),
            run_sim_cached_fn=run_simulation_cached,
            progress_callback=update_node_progress,
        )

        n_prog_bar.empty()
        n_status_text.empty()
        st.session_state["node_sweep_df"] = pd.DataFrame(node_sweep_data)

    if "node_sweep_df" in st.session_state:
        df_node_sweep = st.session_state["node_sweep_df"]
        st.plotly_chart(plot_nodes_vs_throughput(df_node_sweep), use_container_width=True)

        with st.expander("View Raw Node Sweep Data"):
            st.dataframe(df_node_sweep, use_container_width=True, hide_index=True)


# =====================================================================
# TAB 6: THEORY
# =====================================================================
with tab_theory:
    st.subheader("Theoretical MAC Foundations")
    st.warning("All theoretical formulas and values on this page represent analytical models, not empirical simulation results.")

    st.markdown("### Protocol Comparison Summary")
    st.markdown(
        """
        | Protocol | Channel Sensing | Vulnerable Window | Max Theoretical Throughput ($S_{max}$) | Optimal Offered Load ($G^*$) |
        | :--- | :--- | :--- | :--- | :--- |
        | **Pure ALOHA** | None (Immediate Tx) | $2T$ | $\\frac{1}{2e} \\approx 0.184$ (18.4%) | $G = 0.5$ |
        | **Slotted ALOHA** | None (Slot Boundaries) | $T$ | $\\frac{1}{e} \\approx 0.368$ (36.8%) | $G = 1.0$ |
        | **1-Persistent CSMA** | Carrier Sensing (Listen-before-talk) | $\\tau$ (propagation delay) | $\\approx 0.50 - 0.85$ (varies with $a = \\tau / T$) | Depends on $a$ |
        """
    )

    st.markdown("---")

    # 1. Pure ALOHA
    st.markdown("#### 1. Pure ALOHA (Abramson, 1970)")
    st.markdown(
        """
        - **Working Principle**: Continuous time. A node transmits the instant it has a packet ready, without any channel sensing.
        - **Collision Rule**: If two or more transmissions overlap in time even by a single bit, destructive collision occurs and all overlapping packets are lost.
        - **Vulnerable Period**: A packet transmitted at time $t$ will collide if any other transmission starts in the interval $(t - T, t + T)$. The vulnerable window duration is:
        """
    )
    st.latex(r"V_{\text{pure}} = 2T")
    st.markdown(
        """
        - **Analytical Throughput Formula**: Assuming Poisson arrivals with aggregate rate $G$ attempts per packet duration $T$:
        """
    )
    st.latex(r"S = G \cdot e^{-2G}")
    st.markdown(
        """
        - **Maximum Throughput**: Differentiating with respect to $G$ yields maximum at $G = 0.5$:
        """
    )
    st.latex(r"S_{\max} = \frac{1}{2e} \approx 0.1839 \quad (18.4\%)")
    st.markdown(
        """
        - **Advantages**: Completely decentralized, zero synchronization required, simple implementation.
        - **Limitations**: Severe vulnerability window ($2T$) causes rapid throughput collapse under moderate load.
        """
    )

    st.markdown("---")

    # 2. Slotted ALOHA
    st.markdown("#### 2. Slotted ALOHA (Roberts, 1972)")
    st.markdown(
        """
        - **Working Principle**: Time is divided into synchronized discrete slots equal to packet duration $T$. Nodes may only initiate transmission at exact slot boundaries. Packets arriving mid-slot must wait for the next boundary.
        - **Collision Rule**: Collisions occur only when 2 or more nodes transmit within the *same* slot.
        - **Vulnerable Period**: Halved compared to Pure ALOHA:
        """
    )
    st.latex(r"V_{\text{slotted}} = T")
    st.markdown(
        """
        - **Analytical Throughput Formula**:
        """
    )
    st.latex(r"S = G \cdot e^{-G}")
    st.markdown(
        """
        - **Maximum Throughput**: Differentiating yields maximum at $G = 1.0$:
        """
    )
    st.latex(r"S_{\max} = \frac{1}{e} \approx 0.3679 \quad (36.8\%)")
    st.markdown(
        """
        - **Advantages**: Exactly doubles Pure ALOHA channel efficiency.
        - **Limitations**: Requires global slot clock synchronization across all nodes.
        """
    )

    st.markdown("---")

    # 3. 1-Persistent CSMA
    st.markdown("#### 3. 1-Persistent CSMA (Kleinrock & Tobagi, 1975)")
    st.markdown(
        """
        - **Working Principle**: "Listen Before Talk". A node senses the channel before transmitting:
          1. If channel is **IDLE**: Transmit immediately.
          2. If channel is **BUSY**: Continuously listen until the channel becomes idle, then transmit immediately (probability $p = 1$).
        - **Why Collisions Still Happen**:
          1. **Propagation Delay ($\tau$)**: Signals travel at finite speed (approx $2 \\times 10^8$ m/s in copper/fiber). If Node A starts transmitting at $t$, Node B sensing at $t' \\in [t, t + \\tau)$ still hears idle and transmits, causing collision.
          2. **Accumulation During Busy Channel**: If two or more nodes generate packets while another node is transmitting, both wait. As soon as the channel clears, both transmit simultaneously and collide.
        - **Role of Normalized Propagation Delay ($a = \\tau / T$)**:
          As $a \\to 0$, vulnerable window vanishes and throughput approaches 1.0. As $a$ grows, performance degrades toward ALOHA.
        - **Advantages**: Greatly outperforms ALOHA variants by avoiding transmissions during active channel usage.
        - **Limitations**: Collisions still occur due to propagation delay; high contention causes throughput collapse when multiple waiting nodes collide at channel clear.
        """
    )


# =====================================================================
# TAB 7: ABOUT
# =====================================================================
with tab_about:
    st.subheader("About ALOHA vs CSMA Performance Analyzer")
    st.markdown(
        """
        ### Project Description
        This application is an educational, simulation-based Computer Networks performance analyzer designed to model and compare medium access control (MAC) protocols:
        - **Pure ALOHA**
        - **Slotted ALOHA**
        - **1-Persistent CSMA**

        > **Notice:** This is a **discrete-event mathematical simulation engine**, not a real-world physical packet transmission system. It models the random-access problem where multiple uncoordinated nodes share a single broadcast channel.

        ### Key Features
        - **Deterministic Pseudorandomness**: All Poisson arrivals and exponential backoffs flow through explicit seeded NumPy generators.
        - **Exact Physical Mechanics**:
          - Full continuous-time overlap detection for Pure ALOHA.
          - Exact slot-boundary enforcement for Slotted ALOHA.
          - Carrier sensing with physical propagation delay vulnerability windows for 1-Persistent CSMA.
        - **Multi-Protocol Comparison**: Tests all 3 protocols against an identical generated arrival trace.
        - **Parametric Sweeps**: Evaluates scaling across offered loads and node counts with multi-run statistical confidence bounds (mean ± std dev).
        - **Standard Performance Metrics**: Throughput, channel efficiency, collision rates, average packet delay, and channel utilization.

        ### Developed For
        Computer Networks Coursework / Laboratory Simulation.
        """
    )
