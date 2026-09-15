"""
Rule-based business insights and segment recommendations.

Every insight here is computed directly from the segmented dataset - nothing
is a hard-coded statistic. These are business heuristics, not machine-learning
predictions, and are labeled as such in the UI.
"""

from __future__ import annotations

import pandas as pd

from src.segmentation import (
    SEGMENT_AT_RISK,
    SEGMENT_CHAMPIONS,
    SEGMENT_LOST,
    SEGMENT_LOYAL,
    SEGMENT_POTENTIAL,
)

# Static business-heuristic descriptions/recommendations per segment. These
# do not depend on the data - they explain what the segment generally means
# and what a business could consider doing about it. Per-segment *statistics*
# (size, revenue, avg RFM) are always computed live from the data elsewhere.
SEGMENT_INFO = {
    SEGMENT_CHAMPIONS: {
        "profile": (
            "Customers with the strongest combination of recent activity, "
            "frequent purchases, and high spending relative to the rest of "
            "the customer base."
        ),
        "action": (
            "Reward with VIP treatment: early access, exclusive products, "
            "loyalty perks, and referral incentives. Protect this segment - "
            "it is disproportionately valuable."
        ),
    },
    SEGMENT_LOYAL: {
        "profile": (
            "Customers who purchase regularly and generate solid revenue, "
            "even if not quite at the very top tier."
        ),
        "action": (
            "Encourage continued loyalty and upsell/cross-sell opportunities "
            "through targeted loyalty-program incentives."
        ),
    },
    SEGMENT_POTENTIAL: {
        "profile": (
            "Customers with encouraging recent activity whose purchase "
            "frequency or spending has not yet reached top-tier levels."
        ),
        "action": (
            "Nurture with personalized offers, onboarding content, and "
            "engagement campaigns to build habitual purchasing."
        ),
    },
    SEGMENT_AT_RISK: {
        "profile": (
            "Customers who previously showed meaningful engagement but have "
            "not purchased recently - a churn warning sign."
        ),
        "action": (
            "Launch targeted retention campaigns: personalized win-back "
            "offers, check-in communication, or limited-time incentives."
        ),
    },
    SEGMENT_LOST: {
        "profile": (
            "Customers with long inactivity and low historical engagement "
            "or spending."
        ),
        "action": (
            "Low-cost re-engagement campaigns only; avoid heavy investment "
            "in this group relative to other segments."
        ),
    },
}


def _fallback_info(segment_name: str) -> dict:
    return {
        "profile": f"A distinct customer group identified by the clustering model as '{segment_name}'.",
        "action": "Review this segment's RFM profile to decide on an appropriate engagement strategy.",
    }


def get_segment_info(segment_name: str) -> dict:
    return SEGMENT_INFO.get(segment_name, _fallback_info(segment_name))


def recommendations_table(cluster_profiles: pd.DataFrame) -> pd.DataFrame:
    """Build a segment -> recommended action table for the segments actually
    present in this run (in case fewer than 5 distinct labels appear)."""
    segments = sorted(cluster_profiles["segment"].unique())
    rows = [{"Segment": s, "Recommended Action": get_segment_info(s)["action"]} for s in segments]
    return pd.DataFrame(rows)


def revenue_concentration(segmented_df: pd.DataFrame, top_pct: float = 0.2) -> tuple[float, float]:
    """
    Return (top_pct*100 rounded, pct_of_revenue_generated_by_that_top_slice).
    E.g. revenue_concentration(df, 0.2) -> (20.0, 46.3) means the top 20% of
    customers by spend generate 46.3% of total revenue.
    """
    sorted_monetary = segmented_df["monetary"].sort_values(ascending=False)
    n_top = max(1, int(round(len(sorted_monetary) * top_pct)))
    top_revenue = sorted_monetary.iloc[:n_top].sum()
    total_revenue = sorted_monetary.sum()
    pct_revenue = 100.0 * top_revenue / total_revenue if total_revenue > 0 else 0.0
    return round(top_pct * 100, 0), round(pct_revenue, 1)


def generate_insights(segmented_df: pd.DataFrame, cluster_profiles: pd.DataFrame) -> list[str]:
    """
    Generate a short list of rule-based, data-driven insight strings.
    """
    insights = []
    total_customers = len(segmented_df)
    total_revenue = segmented_df["monetary"].sum()

    # Revenue concentration
    top_pct, rev_pct = revenue_concentration(segmented_df, 0.2)
    insights.append(
        f"The top {top_pct:.0f}% of customers by spending generate {rev_pct:.1f}% of total revenue."
    )

    # Segment sizes / revenue share
    seg_stats = (
        segmented_df.groupby("segment")
        .agg(n_customers=("customer_id", "count"), revenue=("monetary", "sum"))
        .reset_index()
    )
    seg_stats["pct_customers"] = 100.0 * seg_stats["n_customers"] / total_customers
    seg_stats["pct_revenue"] = 100.0 * seg_stats["revenue"] / total_revenue if total_revenue > 0 else 0.0

    at_risk_and_lost = seg_stats[seg_stats["segment"].isin([SEGMENT_AT_RISK, SEGMENT_LOST])]
    if not at_risk_and_lost.empty:
        pct_customers = at_risk_and_lost["pct_customers"].sum()
        pct_revenue = at_risk_and_lost["pct_revenue"].sum()
        insights.append(
            f"At-Risk and Lost customers make up {pct_customers:.0f}% of the customer base, "
            f"representing {pct_revenue:.0f}% of historical revenue that is at risk of churning."
        )

    champions = seg_stats[seg_stats["segment"] == SEGMENT_CHAMPIONS]
    if not champions.empty:
        pct_customers = champions["pct_customers"].iloc[0]
        pct_revenue = champions["pct_revenue"].iloc[0]
        insights.append(
            f"Champions represent only {pct_customers:.0f}% of customers but "
            f"{pct_revenue:.0f}% of total revenue."
        )

    # Segment with the highest average purchase frequency
    freq_leader = cluster_profiles.loc[cluster_profiles["avg_frequency"].idxmax()]
    insights.append(
        f"{freq_leader['segment']} customers have the highest average order "
        f"frequency ({freq_leader['avg_frequency']:.1f} orders)."
    )

    # Segment with the highest average recency (most inactive)
    recency_laggard = cluster_profiles.loc[cluster_profiles["avg_recency"].idxmax()]
    insights.append(
        f"{recency_laggard['segment']} customers have gone the longest without "
        f"purchasing, averaging {recency_laggard['avg_recency']:.0f} days since their last order."
    )

    return insights


def build_executive_summary(
    segmented_df: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    dataset_label: str,
    silhouette: float,
) -> str:
    """
    Build a short, dynamic executive-summary paragraph for the Overview page.
    Entirely derived from the current dataset/results - no hard-coded figures.
    """
    n_customers = len(segmented_df)
    n_segments = cluster_profiles["segment"].nunique()
    total_revenue = segmented_df["monetary"].sum()
    top_pct, rev_pct = revenue_concentration(segmented_df, 0.2)

    seg_stats = (
        segmented_df.groupby("segment")
        .agg(n_customers=("customer_id", "count"), revenue=("monetary", "sum"))
        .reset_index()
    )
    seg_stats["pct_customers"] = 100.0 * seg_stats["n_customers"] / n_customers

    at_risk_pct = seg_stats.loc[seg_stats["segment"] == SEGMENT_AT_RISK, "pct_customers"].sum()
    lost_pct = seg_stats.loc[seg_stats["segment"] == SEGMENT_LOST, "pct_customers"].sum()
    watch_pct = at_risk_pct + lost_pct

    summary = (
        f"Analyzing **{dataset_label}** ({n_customers:,} customers, "
        f"${total_revenue:,.0f} total revenue), the model identified **{n_segments} distinct "
        f"customer segments** (cluster quality: silhouette score {silhouette:.2f}). "
        f"The top {top_pct:.0f}% of customers by spend account for {rev_pct:.1f}% of revenue. "
    )
    if watch_pct > 0:
        summary += (
            f"Roughly {watch_pct:.0f}% of the customer base falls into At-Risk or Lost segments - "
            f"a clear opportunity for targeted retention efforts."
        )
    else:
        summary += "No customers currently fall into At-Risk or Lost segments in this run."

    return summary
