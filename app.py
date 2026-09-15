"""
Customer Intelligence Platform
--------------------------------
An end-to-end customer analytics dashboard: transaction data -> cleaning ->
RFM analysis -> K-Means segmentation -> business insights, all in one
Streamlit app.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import SchemaValidationError, load_and_validate
from src.insights import (
    build_executive_summary,
    generate_insights,
    get_segment_info,
    recommendations_table,
)
from src.preprocessing import check_sufficient_data, clean_transactions
from src.rfm import analysis_date_display, compute_rfm, get_default_analysis_date
from src.segmentation import (
    SEGMENT_AT_RISK,
    SEGMENT_LOST,
    compute_silhouette_for_range,
    run_segmentation,
)
from src.visualizations import (
    pca_scatter,
    rfm_scatter,
    segment_distribution_donut,
    segment_profile_bars,
    silhouette_by_k_chart,
)

APP_DIR = Path(__file__).parent
DEMO_DATA_PATH = APP_DIR / "data" / "sample_transactions.csv"

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed" # side bar hide initially
)


# --------------------------------------------------------------------------
# Cached pipeline steps
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_demo_bytes() -> bytes:
    return DEMO_DATA_PATH.read_bytes()


@st.cache_data(show_spinner=False)
def load_raw(file_bytes: bytes) -> pd.DataFrame:
    import io

    return load_and_validate(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def clean_data(raw_df: pd.DataFrame):
    return clean_transactions(raw_df)


@st.cache_data(show_spinner=False)
def compute_rfm_cached(clean_df: pd.DataFrame) -> pd.DataFrame:
    return compute_rfm(clean_df)


@st.cache_data(show_spinner=False)
def run_segmentation_cached(rfm_df: pd.DataFrame, k: int):
    return run_segmentation(rfm_df, k)


@st.cache_data(show_spinner=False)
def silhouette_range_cached(rfm_df: pd.DataFrame, k_values: tuple) -> dict:
    return compute_silhouette_for_range(rfm_df, list(k_values))


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.markdown("## 📊 Customer Intelligence")
    st.sidebar.markdown("---")

    st.sidebar.markdown("#### Data Source")
    source_choice = st.sidebar.radio(
        "Choose a data source",
        options=["Demo Dataset", "Upload CSV"],
        label_visibility="collapsed",
    )

    uploaded_file = None
    if source_choice == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload a transaction CSV", type=["csv"])
        st.sidebar.caption(
            "Required columns (name variations supported): Customer ID, "
            "Invoice/Transaction No, Invoice Date, Quantity, Unit Price."
        )

    st.sidebar.markdown("#### Analysis Settings")
    k = st.sidebar.slider("K-Means Clusters (K)", min_value=3, max_value=6, value=5, step=1)

    st.sidebar.markdown("#### Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Overview", "Customer Segments", "Customer Explorer", "Methodology"],
        label_visibility="collapsed",
    )

    return source_choice, uploaded_file, k, page


# --------------------------------------------------------------------------
# Data pipeline orchestration
# --------------------------------------------------------------------------
def build_pipeline(source_choice: str, uploaded_file, k: int):
    """
    Run the full load -> clean -> RFM -> segmentation pipeline, surfacing
    friendly errors instead of crashing. Returns None (after rendering an
    error) if the pipeline cannot proceed.
    """
    try:
        if source_choice == "Demo Dataset":
            file_bytes = _load_demo_bytes()
            dataset_label = "the demo dataset (synthetic-free sample of a real UK online retail transaction log)"
        else:
            if uploaded_file is None:
                st.info("👈 Upload a CSV file in the sidebar to get started, or switch to the Demo Dataset.")
                return None
            file_bytes = uploaded_file.getvalue()
            dataset_label = f"your uploaded dataset ({uploaded_file.name})"

        raw_df = load_raw(file_bytes)
    except SchemaValidationError as exc:
        st.error(f"❌ **Could not process this file.** {exc}")
        return None
    except ValueError as exc:
        st.error(f"❌ **Could not process this file.** {exc}")
        return None

    clean_df, report = clean_data(raw_df)

    is_sufficient, message = check_sufficient_data(clean_df)
    if not is_sufficient:
        st.error(f"❌ **Not enough usable data.** {message}")
        with st.expander("See data cleaning details"):
            render_cleaning_report(report)
        return None

    rfm_df = compute_rfm_cached(clean_df)
    analysis_date = get_default_analysis_date(clean_df)

    try:
        result = run_segmentation_cached(rfm_df, k)
    except ValueError as exc:
        st.error(f"❌ **Segmentation could not run.** {exc}")
        return None

    return {
        "dataset_label": dataset_label,
        "raw_df": raw_df,
        "clean_df": clean_df,
        "report": report,
        "rfm_df": rfm_df,
        "analysis_date": analysis_date,
        "result": result,
    }


def build_segment_summary_table(cluster_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cluster-level profiles into a per-business-segment summary
    table. If multiple K-Means clusters share the same business segment
    label, their customers/percentages are combined (summed) and their
    averages are combined (mean), so the table always reflects business
    segments, not raw cluster IDs.
    """
    summary_table = cluster_profiles[
        ["segment", "n_customers", "pct_customers", "avg_recency", "avg_frequency", "avg_monetary"]
    ].groupby("segment", as_index=False).agg(
        {
            "n_customers": "sum",
            "pct_customers": "sum",
            "avg_recency": "mean",
            "avg_frequency": "mean",
            "avg_monetary": "mean",
        }
    )
    return summary_table.rename(
        columns={
            "segment": "Segment",
            "n_customers": "Customers",
            "pct_customers": "% Customers",
            "avg_recency": "Avg Recency",
            "avg_frequency": "Avg Frequency",
            "avg_monetary": "Avg Monetary",
        }
    ).sort_values("Customers", ascending=False)


def render_segment_summary_table(cluster_profiles: pd.DataFrame):
    """Render the segment profile summary table with consistent formatting."""
    table = build_segment_summary_table(cluster_profiles)
    st.dataframe(
        table.style.format(
            {
                "% Customers": "{:.1f}%",
                "Avg Recency": "{:.0f} days",
                "Avg Frequency": "{:.1f}",
                "Avg Monetary": "${:,.0f}",
            }
        ),
        width='stretch',
        hide_index=True,
    )


def render_cleaning_report(report):
    st.write(
        f"Started with **{report.starting_rows:,}** transaction rows, kept "
        f"**{report.ending_rows:,}** after cleaning "
        f"({report.pct_removed:.1f}% removed)."
    )
    if report.steps:
        for step, n in report.steps:
            st.write(f"- {step}: **{n:,}** rows")
    else:
        st.write("No cleaning steps removed any rows - the data was already clean.")


# --------------------------------------------------------------------------
# Overview page
# --------------------------------------------------------------------------
def render_overview(ctx):
    st.title("Customer Intelligence Platform")
    st.caption("Transform transaction data into actionable customer segments.")

    with st.expander("How this works", expanded=False):
        st.markdown(
            "**Transaction Data** → **Data Cleaning** → **RFM Analysis** → "
            "**Customer Segmentation** → **Business Insights**"
        )
        st.write(
            "Raw transactions are cleaned, then summarized per customer using "
            "Recency, Frequency and Monetary (RFM) metrics. K-Means clustering "
            "groups customers with similar RFM behavior, and each group is "
            "translated into a business-friendly segment name."
        )

    result = ctx["result"]
    segmented_df = result.segmented_df
    cluster_profiles = result.cluster_profiles

    summary = build_executive_summary(
        segmented_df, cluster_profiles, ctx["dataset_label"], result.silhouette
    )
    st.info(summary)

    # KPI cards
    total_customers = len(segmented_df)
    total_revenue = segmented_df["monetary"].sum()
    total_orders = int(segmented_df["frequency"].sum())
    avg_customer_value = total_revenue / total_customers if total_customers else 0
    n_segments = cluster_profiles["segment"].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Total Revenue", f"${total_revenue:,.0f}")
    c3.metric("Total Orders", f"{total_orders:,}")
    c4.metric("Avg Customer Value", f"${avg_customer_value:,.0f}")
    c5.metric("Business Segments", f"{n_segments}")

    st.markdown("### Customer Segment Overview")
    col_chart, col_table = st.columns([1, 1.3])
    with col_chart:
        st.plotly_chart(segment_distribution_donut(segmented_df), width='stretch')
    with col_table:
        render_segment_summary_table(cluster_profiles)

    st.markdown("### Business Insights")
    for insight in generate_insights(segmented_df, cluster_profiles):
        st.markdown(f"- {insight}")
    st.caption("These insights are computed directly from your data using simple business rules - not machine-learning predictions.")

    with st.expander("How was your data cleaned?"):
        render_cleaning_report(ctx["report"])


# --------------------------------------------------------------------------
# Customer Segments page
# --------------------------------------------------------------------------
def render_segments(ctx):
    st.title("Customer Segments")

    result = ctx["result"]
    segmented_df = result.segmented_df
    cluster_profiles = result.cluster_profiles
    n_clusters = result.k
    n_segments = cluster_profiles["segment"].nunique()

    st.markdown("### K-Means Clusters vs. Business Segments")
    st.write(
        "K-Means groups customers into technical clusters purely by statistical "
        "similarity in their RFM behavior. Each cluster is then translated into "
        "an actionable **business segment** (Champions, Loyal Customers, Potential "
        "Loyalists, At Risk, Lost) using a documented rule-based mapping - see the "
        "Methodology page for details. The business segment is the primary, "
        "customer-facing output; the cluster is the underlying technical layer."
    )
    st.markdown(f"**{n_clusters} K-Means clusters → {n_segments} business segments**")
    if n_clusters != n_segments:
        st.caption(
            f"{n_clusters - n_segments} cluster(s) share a business segment label with another "
            "cluster, since those clusters represent similar customer behavior from a business "
            "standpoint. This is expected and does not indicate an error."
        )
    else:
        st.caption("Each K-Means cluster mapped to a distinct business segment in this run.")

        # Compare the selected K against all tested cluster counts.
    scores = silhouette_range_cached(ctx["rfm_df"], (3, 4, 5, 6))

    # Find the K with the highest silhouette score.
    best_k = max(scores, key=scores.get)
    best_silhouette = scores[best_k]

    st.markdown(
        f"Silhouette score for the selected **K={n_clusters}**: "
        f"**{result.silhouette:.2f}** (higher is better)."
    )

    if best_k != n_clusters:
        st.info(
            f"**Model selection note:** K={best_k} has the highest silhouette "
            f"score in the tested range (3–6), at **{best_silhouette:.2f}**, "
            f"compared with **{result.silhouette:.2f}** for the selected K={n_clusters}. "
            f"The current K={n_clusters} remains selected because K can also be "
            f"chosen based on the desired level of customer differentiation and "
            f"business interpretability."
        )
    else:
        st.success(
            f"**Model selection note:** K={n_clusters} has the highest silhouette "
            f"score in the tested range (3–6), at **{best_silhouette:.2f}**."
        )

    with st.expander("Compare silhouette score across cluster counts"):
        st.plotly_chart(silhouette_by_k_chart(scores), width='stretch')
        st.caption(
            "The silhouette score evaluates cluster cohesion and separation. "
            "Use it together with business interpretability when selecting K."
        )

    st.markdown("### Segment Profile")
    render_segment_summary_table(cluster_profiles)

    st.markdown("### RFM Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            rfm_scatter(segmented_df, "recency", "monetary", "Recency vs Monetary by Segment"),
            width='stretch',
        )
    with col2:
        st.plotly_chart(
            rfm_scatter(segmented_df, "frequency", "monetary", "Frequency vs Monetary by Segment"),
            width='stretch',
        )
    st.plotly_chart(segment_profile_bars(cluster_profiles), width='stretch')

    st.markdown("### PCA Visualization")
    st.caption(
        "PCA is used only to visualize the standardized customer features in two dimensions. "
        "It is not the clustering algorithm itself - clustering is done on the original scaled RFM features."
    )
    st.plotly_chart(pca_scatter(segmented_df), width='stretch')

    st.markdown("### Segment Details")
    segment_names = sorted(cluster_profiles["segment"].unique())
    selected_segment = st.selectbox("Select Segment", segment_names)

    seg_rows = segmented_df[segmented_df["segment"] == selected_segment]
    seg_revenue = seg_rows["monetary"].sum()
    total_revenue = segmented_df["monetary"].sum()
    info = get_segment_info(selected_segment)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Customers", f"{len(seg_rows):,}")
    sc2.metric("% of Customer Base", f"{100 * len(seg_rows) / len(segmented_df):.1f}%")
    sc3.metric("Total Revenue Contributed", f"${seg_revenue:,.0f}")
    sc4.metric("% of Total Revenue", f"{100 * seg_revenue / total_revenue:.1f}%" if total_revenue else "0%")

    r1, r2, r3 = st.columns(3)
    r1.metric("Avg Recency", f"{seg_rows['recency'].mean():.0f} days")
    r2.metric("Avg Frequency", f"{seg_rows['frequency'].mean():.1f} orders")
    r3.metric("Avg Monetary", f"${seg_rows['monetary'].mean():,.0f}")

    st.markdown(f"**Profile:** {info['profile']}")
    st.markdown(f"**Recommended Action:** {info['action']}")
    st.caption("Recommended actions are business heuristics, not guaranteed outcomes.")

    st.markdown("### Recommendations by Segment")
    st.dataframe(recommendations_table(cluster_profiles), width='stretch', hide_index=True)

    st.markdown("### At-Risk & Lost Customers")
    watch_df = segmented_df[segmented_df["segment"].isin([SEGMENT_AT_RISK, SEGMENT_LOST])]
    if watch_df.empty:
        st.write("No customers currently fall into the At-Risk or Lost segments.")
    else:
        display_cols = ["customer_id", "segment", "recency", "frequency", "monetary"]
        st.dataframe(
            watch_df[display_cols].sort_values("monetary", ascending=False),
            width='stretch',
            hide_index=True,
        )
        st.download_button(
            "Download At-Risk & Lost Customers (CSV)",
            data=watch_df[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="at_risk_and_lost_customers.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------
# Customer Explorer page
# --------------------------------------------------------------------------
def render_customer_explorer(ctx):
    st.title("Customer Explorer")

    result = ctx["result"]
    segmented_df = result.segmented_df

    st.markdown("### Search Customer")
    customer_ids = sorted(segmented_df["customer_id"].astype(str).unique())
    selected_id = st.selectbox("Select Customer ID", customer_ids)

    row = segmented_df[segmented_df["customer_id"].astype(str) == selected_id].iloc[0]
    info = get_segment_info(row["segment"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segment", row["segment"])
    c2.metric("Recency", f"{row['recency']:.0f} days")
    c3.metric("Frequency", f"{row['frequency']:.0f} orders")
    c4.metric("Monetary", f"${row['monetary']:,.0f}")
    st.caption(f"Technical detail: this customer belongs to K-Means cluster #{int(row['cluster'])}.")

    st.write(f"**Last purchase date:** {row['last_purchase_date'].date()}")
    st.write(f"**Customer since:** {row['first_purchase_date'].date()}")
    st.markdown(f"**Behavior:** {info['profile']}")
    st.markdown(f"**Suggested action:** {info['action']}")

    st.markdown("### Top Customers by Revenue")
    top_customers = segmented_df[
        ["customer_id", "segment", "cluster", "monetary", "frequency", "recency"]
    ].rename(
        columns={
            "customer_id": "Customer ID",
            "segment": "Segment",
            "cluster": "Cluster (Technical)",
            "monetary": "Revenue",
            "frequency": "Orders",
            "recency": "Recency",
        }
    ).sort_values("Revenue", ascending=False)
    st.dataframe(
        top_customers.head(50).style.format({"Revenue": "${:,.0f}"}),
        width='stretch',
        hide_index=True,
    )
    st.caption("Showing top 50 customers by revenue. Click a column header to sort.")

    st.markdown("### Download Full Segmentation Results")
    export_df = segmented_df[["customer_id", "recency", "frequency", "monetary", "cluster", "segment"]].rename(
        columns={
            "customer_id": "Customer_ID",
            "recency": "Recency",
            "frequency": "Frequency",
            "monetary": "Monetary",
            "cluster": "Cluster",
            "segment": "Segment",
        }
    )
    st.download_button(
        "Download Customer Segmentation (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="customer_segmentation.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------
# Methodology page
# --------------------------------------------------------------------------
def render_methodology(ctx):
    st.title("Methodology")

    analysis_date_str = analysis_date_display(ctx["analysis_date"]) if ctx else "N/A"

    st.markdown(
        "**Transactions → Data Cleaning → RFM Analysis → Feature Transformation/Scaling "
        "→ K-Means Clustering → Business Segment Labeling → Business Recommendations**"
    )

    st.markdown(
        f"""
### 1. Data Preparation
Transaction records are cleaned in a fixed, transparent sequence: rows with
missing Customer IDs, unparseable dates, cancelled invoices, non-positive
quantities/prices, exact duplicates, and non-positive revenue are removed.
Every removal is counted and shown in the Overview page's cleaning report -
nothing is silently transformed.

### 2. RFM Analysis
Each customer is summarized using three metrics:
- **Recency**: days since their most recent purchase (lower is better)
- **Frequency**: number of distinct orders placed (higher is better)
- **Monetary**: total revenue generated (higher is better)

The analysis (reference) date for this run is **{analysis_date_str}**,
computed as one day after the most recent transaction in the cleaned
dataset - this avoids using any information from "the future" relative to
the data itself.

### 3. Feature Transformation
Frequency and Monetary are typically heavily right-skewed in transaction
data (a small number of customers place far more orders or spend far more
than everyone else). Extreme values are first clipped to the 1st/99th
percentile to reduce the influence of outliers without discarding valuable
customers, then a log1p transform is applied to Frequency and Monetary to
reduce skew before clustering.

### 4. Standardization
K-Means is a distance-based algorithm, so all features must be on a
comparable scale. `StandardScaler` is applied to the (clipped,
log-transformed) Recency/Frequency/Monetary features so no single metric
dominates the distance calculation purely due to its units.

### 5. Clustering
`K-Means` groups customers into a user-selected number of clusters (3-6)
based on their standardized RFM features. Cluster assignments are arbitrary
numbers (0, 1, 2, ...) with no inherent business meaning on their own.

### 6. Business Segment Labeling
Raw cluster numbers are translated into business-friendly names
(Champions, Loyal Customers, Potential Loyalists, At Risk, Lost) using an
explicit, documented rule table: each cluster's mean Recency, Frequency and
Monetary is scored relative to the other clusters in the same run, bucketed
into high/medium/low, and mapped to a segment name. **This labeling is a
human-defined, post-hoc interpretation of the cluster centroids - it is not
something K-Means learns on its own.**

### 7. Cluster Evaluation
The **Silhouette Score** (range -1 to 1) measures how well-separated and
internally cohesive the clusters are. Higher is generally better; scores
above roughly 0.3-0.5 are considered reasonable for real-world customer
behavior data, which rarely forms perfectly separated clusters.

### 8. PCA (Principal Component Analysis)
PCA reduces the standardized RFM features to two dimensions purely for
**visualization** - so clusters that live in 3D feature space can be plotted
on a 2D scatter plot. PCA is not used to perform the clustering itself.

### 9. Business Recommendations
Each business segment (not each raw cluster) is paired with a practical,
heuristic recommendation - for example, retention offers for At-Risk
customers or VIP treatment for Champions. These are standard RFM-segment
business practices, not predictions from a trained model, and are shown on
the Customer Segments page.
"""
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    source_choice, uploaded_file, k, page = render_sidebar()
    ctx = build_pipeline(source_choice, uploaded_file, k)

    if ctx is None:
        if page == "Methodology":
            render_methodology(None)
        return

    if page == "Overview":
        render_overview(ctx)
    elif page == "Customer Segments":
        render_segments(ctx)
    elif page == "Customer Explorer":
        render_customer_explorer(ctx)
    elif page == "Methodology":
        render_methodology(ctx)


if __name__ == "__main__":
    main()
