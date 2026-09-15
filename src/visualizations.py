"""
Plotly figure builders for the Customer Intelligence Platform dashboard.

Every function here takes already-computed data (segmented customers and/or
cluster profiles) and returns a ready-to-render Plotly figure. No analysis
logic lives in this module - it is presentation only.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.segmentation import (
    SEGMENT_AT_RISK,
    SEGMENT_CHAMPIONS,
    SEGMENT_LOST,
    SEGMENT_LOYAL,
    SEGMENT_POTENTIAL,
)

# Fixed color mapping so a segment's color stays consistent across every
# chart on the dashboard. Falls back to Plotly's qualitative palette for any
# unrecognized segment name.
_SEGMENT_COLORS = {
    SEGMENT_CHAMPIONS: "#2E7D32",  # green
    SEGMENT_LOYAL: "#1E88E5",  # blue
    SEGMENT_POTENTIAL: "#00ACC1",  # teal
    SEGMENT_AT_RISK: "#FB8C00",  # orange
    SEGMENT_LOST: "#B0BEC5",  # grey
}

_TEMPLATE = "plotly_white"


def _color_map_for(segments: pd.Series) -> dict:
    palette = px.colors.qualitative.Set2
    color_map = {}
    fallback_i = 0
    for seg in sorted(segments.unique()):
        if seg in _SEGMENT_COLORS:
            color_map[seg] = _SEGMENT_COLORS[seg]
        else:
            color_map[seg] = palette[fallback_i % len(palette)]
            fallback_i += 1
    return color_map


def segment_distribution_donut(segmented_df: pd.DataFrame) -> go.Figure:
    counts = segmented_df["segment"].value_counts().reset_index()
    counts.columns = ["segment", "count"]
    color_map = _color_map_for(segmented_df["segment"])

    fig = px.pie(
        counts,
        names="segment",
        values="count",
        hole=0.5,
        color="segment",
        color_discrete_map=color_map,
        title="Customer Segment Distribution",
    )
    fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value} customers (%{percent})")
    fig.update_layout(template=_TEMPLATE, showlegend=True, legend_title_text="Segment")
    return fig


def rfm_scatter(segmented_df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    labels = {"recency": "Recency (days since last purchase)", "frequency": "Frequency (orders)", "monetary": "Monetary (total spend)"}
    color_map = _color_map_for(segmented_df["segment"])
    fig = px.scatter(
        segmented_df,
        x=x,
        y=y,
        color="segment",
        color_discrete_map=color_map,
        hover_data={"customer_id": True, "recency": True, "frequency": True, "monetary": ":.2f"},
        labels={x: labels.get(x, x), y: labels.get(y, y), "segment": "Segment"},
        title=title,
    )
    fig.update_layout(template=_TEMPLATE, legend_title_text="Segment")
    return fig


def segment_profile_bars(cluster_profiles: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of average Recency / Frequency / Monetary per segment.

    Each metric is min-max normalized to a 0-100 scale purely to determine
    bar height, so Recency/Frequency/Monetary can share a single y-axis
    despite living on very different real-world scales. The normalization
    is visual only - actual values are preserved and shown via on-bar
    labels and hover tooltips.
    """

    metrics = [
        ("avg_recency", "Avg Recency (days)", "#42A5F5", "days"),
        ("avg_frequency", "Avg Frequency (orders)", "#81D4FA", "orders"),
        ("avg_monetary", "Avg Monetary (spend)", "#EF5350", ""),
    ]

    fig = go.Figure()

    for col, name, color, unit in metrics:
        real_values = cluster_profiles[col]
        col_min = real_values.min()
        col_max = real_values.max()
        span = col_max - col_min
        # Avoid divide-by-zero if a metric is constant across segments,
        # and floor at 3 so genuinely small (but non-zero) values still
        # render a visible sliver of a bar instead of disappearing.
        if span > 0:
            normalized = (real_values - col_min) / span * 100
        else:
            normalized = real_values * 0 + 10
        normalized = normalized.clip(lower=3)

        label_suffix = f" {unit}" if unit else ""

        fig.add_trace(
            go.Bar(
                x=cluster_profiles["segment"],
                y=normalized,
                name=name,
                marker_color=color,
                customdata=real_values,
                text=real_values.round(1),
                texttemplate="%{text}" + label_suffix,
                textposition="auto",
                textfont=dict(size=10),
                hovertemplate=(
                    "Segment: %{x}<br>"
                    f"{name}: " + "%{customdata:.2f}" + label_suffix +
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text="Average RFM Values by Segment", x=0.0, xanchor="left"),
        template=_TEMPLATE,
        barmode="group",
        bargap=0.25,
        bargroupgap=0.15,
        xaxis=dict(title="Segment"),
        yaxis=dict(title="Normalized height (0-100)", range=[0, 100]),
        legend_title_text="Metric",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.97),
        margin=dict(t=90, b=90, l=70),
    )

    fig.add_annotation(
        text=(
            "(Bar height is normalized 0–100 per metric for visual comparison — "
            "it does not represent equal real values. See labels for actual numbers.)"
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.24,
        xanchor="center",
        yanchor="top",
        showarrow=False,
        font=dict(size=11, color="#999999"),
        align="center",
    )

    

    return fig


def pca_scatter(segmented_df: pd.DataFrame) -> go.Figure:
    color_map = _color_map_for(segmented_df["segment"])
    fig = px.scatter(
        segmented_df,
        x="pca_1",
        y="pca_2",
        color="segment",
        color_discrete_map=color_map,
        hover_data={"customer_id": True, "recency": True, "frequency": True, "monetary": ":.2f"},
        labels={"pca_1": "PCA Component 1", "pca_2": "PCA Component 2", "segment": "Segment"},
        title="Customer Segments in Two-Dimensional PCA Space",
    )
    fig.update_layout(template=_TEMPLATE, legend_title_text="Segment")
    return fig


def silhouette_by_k_chart(scores: dict[int, float]) -> go.Figure:
    ks = sorted(scores.keys())
    values = [scores[k] for k in ks]
    fig = go.Figure(
        go.Scatter(x=ks, y=values, mode="lines+markers", line=dict(color="#1E88E5"))
    )
    fig.update_layout(
        title="Silhouette Score by Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        template=_TEMPLATE,
        xaxis=dict(tickmode="linear"),
    )
    return fig
