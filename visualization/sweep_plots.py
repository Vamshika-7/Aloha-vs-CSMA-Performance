"""Plotly visualization functions for parameter sweep and multi-run statistical analysis."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROTOCOL_COLORS = {
    "Pure ALOHA": "#3B82F6",       # Blue
    "Slotted ALOHA": "#F59E0B",    # Amber
    "1-Persistent CSMA": "#10B981", # Emerald
}

PROTOCOL_MARKERS = {
    "Pure ALOHA": "circle",
    "Slotted ALOHA": "square",
    "1-Persistent CSMA": "diamond",
}


def _create_sweep_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_mean_col: str,
    y_std_col: str,
    title: str,
    x_label: str,
    y_label: str,
    y_scale: float = 1.0,
    y_format: str = ".3f",
) -> go.Figure:
    """Helper creating a multi-line sweep plot with error bars for each protocol."""
    fig = go.Figure()

    if df.empty or x_col not in df.columns:
        fig.add_annotation(
            text="No sweep data available. Configure parameters and run the sweep.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6b7280"),
        )
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=380)
        return fig

    protocols = df["protocol"].unique()

    for proto in protocols:
        p_df = df[df["protocol"] == proto].sort_values(by=x_col)
        color = PROTOCOL_COLORS.get(proto, "#6B7280")
        marker_symbol = PROTOCOL_MARKERS.get(proto, "circle")

        x_vals = p_df[x_col].values
        y_vals = p_df[y_mean_col].values * y_scale
        has_std = y_std_col in p_df.columns
        std_vals = (p_df[y_std_col].values * y_scale) if has_std else None

        show_error = has_std and any(s > 1e-6 for s in std_vals)

        error_dict = None
        if show_error:
            # Clip the lower end of error bars at 0 so y - arrayminus >= 0
            std_minus = np.minimum(std_vals, np.maximum(y_vals, 0.0))
            error_dict = dict(
                type="data",
                symmetric=False,
                array=std_vals,
                arrayminus=std_minus,
                visible=True,
                thickness=1.5,
                width=4,
                color=color,
            )

        custom_data = []
        for i in range(len(x_vals)):
            std_str = f" ± {std_vals[i]:{y_format}}" if show_error else ""
            custom_data.append(std_str)

        fig.add_trace(
            go.Scatter(
                name=proto,
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(symbol=marker_symbol, size=8, color=color),
                error_y=error_dict,
                customdata=custom_data,
                hovertemplate=(
                    f"<b>{proto}</b><br>"
                    f"{x_label}: %{{x}}<br>"
                    f"{y_label}: %{{y:{y_format}}}%{{customdata}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1F2937")),
        xaxis=dict(
            title=x_label,
            showgrid=True,
            gridcolor="#E5E7EB",
            zeroline=True,
            zerolinecolor="#9CA3AF",
        ),
        yaxis=dict(
            title=y_label,
            showgrid=True,
            gridcolor="#E5E7EB",
            zeroline=True,
            zerolinecolor="#9CA3AF",
            rangemode="tozero",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=400,
        margin=dict(l=65, r=30, t=60, b=50),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


def plot_load_vs_throughput(sweep_df: pd.DataFrame, show_theory: bool = False) -> go.Figure:
    """Plot 1: Offered Load (G) vs Throughput, optionally with theoretical curves for ALOHA."""
    fig = _create_sweep_line_plot(
        df=sweep_df,
        x_col="load",
        y_mean_col="throughput_mean",
        y_std_col="throughput_std",
        title="Offered Load (G) vs Throughput",
        x_label="Offered Load (G) [arrivals / packet duration]",
        y_label="Throughput [packets / second]",
        y_scale=1.0,
        y_format=".3f",
    )

    if show_theory:
        from simulation.theory import pure_aloha_throughput, slotted_aloha_throughput

        max_g = 1.5
        if not sweep_df.empty and "load" in sweep_df.columns:
            max_g = max(float(sweep_df["load"].max()), 1.5)

        g_smooth = np.linspace(0.0, max_g, 200)

        # Pure ALOHA theoretical curve: S = G * exp(-2G)
        s_pure = pure_aloha_throughput(g_smooth)
        fig.add_trace(
            go.Scatter(
                name="Pure ALOHA (theoretical)",
                x=g_smooth,
                y=s_pure,
                mode="lines",
                line=dict(color=PROTOCOL_COLORS.get("Pure ALOHA", "#3B82F6"), dash="dash", width=2),
                hovertemplate="<b>Pure ALOHA (theoretical)</b><br>Offered Load (G): %{x:.3f}<br>Throughput: %{y:.3f}<extra></extra>",
            )
        )

        # Slotted ALOHA theoretical curve: S = G * exp(-G)
        s_slotted = slotted_aloha_throughput(g_smooth)
        fig.add_trace(
            go.Scatter(
                name="Slotted ALOHA (theoretical)",
                x=g_smooth,
                y=s_slotted,
                mode="lines",
                line=dict(color=PROTOCOL_COLORS.get("Slotted ALOHA", "#F59E0B"), dash="dash", width=2),
                hovertemplate="<b>Slotted ALOHA (theoretical)</b><br>Offered Load (G): %{x:.3f}<br>Throughput: %{y:.3f}<extra></extra>",
            )
        )

    return fig


def plot_load_vs_collision_rate(sweep_df: pd.DataFrame) -> go.Figure:
    """Plot 2: Offered Load (G) vs Collision Rate (%)."""
    return _create_sweep_line_plot(
        df=sweep_df,
        x_col="load",
        y_mean_col="collision_rate_mean",
        y_std_col="collision_rate_std",
        title="Offered Load (G) vs Collision Rate",
        x_label="Offered Load (G)",
        y_label="Collision Rate (%)",
        y_scale=100.0,
        y_format=".1f",
    )


def plot_load_vs_delay(sweep_df: pd.DataFrame) -> go.Figure:
    """Plot 3: Offered Load (G) vs Average Delay."""
    return _create_sweep_line_plot(
        df=sweep_df,
        x_col="load",
        y_mean_col="avg_delay_mean",
        y_std_col="avg_delay_std",
        title="Offered Load (G) vs Average Delay",
        x_label="Offered Load (G)",
        y_label="Average Delay [seconds]",
        y_scale=1.0,
        y_format=".3f",
    )


def plot_load_vs_retransmissions(sweep_df: pd.DataFrame) -> go.Figure:
    """Plot 4: Offered Load (G) vs Average Retransmissions."""
    return _create_sweep_line_plot(
        df=sweep_df,
        x_col="load",
        y_mean_col="avg_retransmissions_mean",
        y_std_col="avg_retransmissions_std",
        title="Offered Load (G) vs Average Retransmissions",
        x_label="Offered Load (G)",
        y_label="Average Retransmissions [per packet]",
        y_scale=1.0,
        y_format=".2f",
    )


def plot_nodes_vs_throughput(node_sweep_df: pd.DataFrame) -> go.Figure:
    """Plot 5: Number of Nodes vs Throughput."""
    return _create_sweep_line_plot(
        df=node_sweep_df,
        x_col="num_nodes",
        y_mean_col="throughput_mean",
        y_std_col="throughput_std",
        title="Number of Nodes vs Throughput",
        x_label="Number of Contending Nodes",
        y_label="Throughput [packets / second]",
        y_scale=1.0,
        y_format=".3f",
    )
