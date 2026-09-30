"""Per-node horizontal transmission timeline visualization module."""

from typing import List, Optional, Tuple
import plotly.graph_objects as go


def create_timeline_figure(
    channel_log: List[Tuple[float, float, int, str]],
    num_nodes: int,
    sim_time: float,
    max_time_window: float = 50.0,
) -> Tuple[go.Figure, Optional[str]]:
    """Creates an interactive Plotly horizontal timeline of channel transmission events.

    Args:
        channel_log: List of (start_time, end_time, node_id, status) tuples.
        num_nodes: Total number of nodes in the simulation.
        sim_time: Full simulation duration.
        max_time_window: Maximum time units to render to prevent browser lag.

    Returns:
        tuple of (plotly_figure, note_string_or_None)
    """
    # Determine the display window cap
    capped_time = min(sim_time, max_time_window)

    # Filter events within the capped window
    # Include any event starting before capped_time
    filtered_log = [entry for entry in channel_log if entry[0] < capped_time]

    # Generate note if the timeline is capped
    note: Optional[str] = None
    if sim_time > max_time_window:
        note = (
            f"Showing first {capped_time:.1f} of {sim_time:.1f} time units "
            f"({len(filtered_log)} of {len(channel_log)} events). "
            f"Reduce simulation duration to see the full timeline."
        )

    fig = go.Figure()

    if not filtered_log:
        fig.add_annotation(
            text="No transmission events logged within this time window.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6b7280"),
        )
        fig.update_layout(
            title="Channel Transmission Timeline (Empty)",
            xaxis_title="Time (s)",
            yaxis_title="Node",
            height=300,
            margin=dict(l=60, r=30, t=50, b=50),
        )
        return fig, note

    # Partition events by status for distinct legend & color grouping
    success_events = [e for e in filtered_log if e[3] == "SUCCESS"]
    collision_events = [e for e in filtered_log if e[3] == "COLLISION"]

    # Map node IDs to discrete categorical labels
    node_labels = [f"Node {i}" for i in range(num_nodes)]

    # Add SUCCESS trace
    if success_events:
        fig.add_trace(
            go.Bar(
                name="SUCCESS",
                y=[f"Node {e[2]}" for e in success_events],
                x=[round(min(e[1], capped_time) - e[0], 6) for e in success_events],
                base=[round(e[0], 6) for e in success_events],
                orientation="h",
                marker=dict(
                    color="#10B981",
                    line=dict(color="#047857", width=1),
                ),
                customdata=[[e[0], e[1], e[2]] for e in success_events],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Status: <span style='color:#10B981;'>SUCCESS</span><br>"
                    "Start: %{customdata[0]:.4f}s<br>"
                    "End: %{customdata[1]:.4f}s<br>"
                    "Duration: %{x:.4f}s<extra></extra>"
                ),
            )
        )

    # Add COLLISION trace
    if collision_events:
        fig.add_trace(
            go.Bar(
                name="COLLISION",
                y=[f"Node {e[2]}" for e in collision_events],
                x=[round(min(e[1], capped_time) - e[0], 6) for e in collision_events],
                base=[round(e[0], 6) for e in collision_events],
                orientation="h",
                marker=dict(
                    color="#EF4444",
                    line=dict(color="#B91C1C", width=1),
                ),
                customdata=[[e[0], e[1], e[2]] for e in collision_events],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Status: <span style='color:#EF4444;'>COLLISION</span><br>"
                    "Start: %{customdata[0]:.4f}s<br>"
                    "End: %{customdata[1]:.4f}s<br>"
                    "Duration: %{x:.4f}s<extra></extra>"
                ),
            )
        )

    # Calculate reasonable chart height based on node count
    chart_height = max(350, 45 * num_nodes + 120)

    fig.update_layout(
        title=dict(
            text="Channel Transmission Activity per Node",
            font=dict(size=16, color="#1F2937"),
        ),
        barmode="overlay",
        xaxis=dict(
            title="Simulation Time (seconds)",
            range=[0, capped_time],
            showgrid=True,
            gridcolor="#E5E7EB",
            zeroline=True,
            zerolinecolor="#9CA3AF",
        ),
        yaxis=dict(
            title="Node ID",
            categoryorder="array",
            categoryarray=list(reversed(node_labels)),  # Node 0 at top
            showgrid=True,
            gridcolor="#F3F4F6",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=chart_height,
        margin=dict(l=80, r=40, t=70, b=60),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )

    return fig, note
