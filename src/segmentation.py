"""
Customer segmentation via K-Means on RFM features.

Pipeline (fixed and deterministic, no dataset-dependent branching):
  1. Percentile clipping (1st / 99th) on Recency, Frequency, Monetary to limit
     the influence of extreme outliers without discarding customers.
  2. log1p transform on Frequency and Monetary, which are typically heavily
     right-skewed in transaction data (a small number of customers place far
     more orders / spend far more than the rest). Recency is left untransformed
     since it is not similarly skewed.
  3. StandardScaler on the three resulting features (K-Means is distance-based
     and requires comparable feature scales).
  4. K-Means clustering with a user-chosen k.
  5. Silhouette score to report cluster cohesion/separation quality.
  6. PCA to 2 components, purely for visualizing the scaled feature space
     (PCA is NOT used as the clustering algorithm).

Business segment labeling
--------------------------
K-Means only produces arbitrary cluster numbers (0, 1, 2, ...). To turn these
into business-meaningful names, we profile each cluster's *mean* Recency,
Frequency and Monetary value, convert each into a relative "goodness" score
against the other clusters found in this run, and apply an explicit,
documented rule table (see `_SEGMENT_RULES` / `assign_segment_labels`) to
decide the label. This is a post-hoc, human-defined interpretation of the
cluster centroids - it is not something K-Means learns on its own, and it is
intentionally kept simple and transparent rather than a black-box scoring
system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CLUSTER = "cluster"
SEGMENT = "segment"

LOWER_PCT = 0.01
UPPER_PCT = 0.99
RANDOM_STATE = 42

# silhouette_score computes a full pairwise-distance matrix by default, which
# is O(n^2) in memory and becomes infeasible for large customer counts
# (confirmed to crash around ~100k+ customers). Above this threshold, we
# bound it to a random sample, which is scikit-learn's documented approach
# for large-scale silhouette evaluation. This does not affect clustering,
# labeling, or results for normal-sized datasets (below the threshold).
MAX_SILHOUETTE_SAMPLE = 5000


def _silhouette_sample_size(n_rows: int) -> "int | None":
    return MAX_SILHOUETTE_SAMPLE if n_rows > MAX_SILHOUETTE_SAMPLE else None

# Segments ordered from most to least valuable, used as a fallback ordering.
SEGMENT_CHAMPIONS = "Champions"
SEGMENT_LOYAL = "Loyal Customers"
SEGMENT_POTENTIAL = "Potential Loyalists"
SEGMENT_AT_RISK = "At Risk"
SEGMENT_LOST = "Lost"


@dataclass
class SegmentationResult:
    segmented_df: pd.DataFrame  # customer_id, recency, frequency, monetary, cluster, segment, pca_1, pca_2
    cluster_profiles: pd.DataFrame  # per-cluster stats + assigned segment + rationale scores
    silhouette: float
    k: int


def _clip_outliers(series: pd.Series) -> pd.Series:
    lower = series.quantile(LOWER_PCT)
    upper = series.quantile(UPPER_PCT)
    return series.clip(lower=lower, upper=upper)


def _prepare_features(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Clip outliers, then log-transform Frequency/Monetary. Returns a new
    DataFrame with columns recency, frequency, monetary (transformed)."""
    features = pd.DataFrame(index=rfm_df.index)
    features["recency"] = _clip_outliers(rfm_df["recency"].astype(float))
    features["frequency"] = np.log1p(_clip_outliers(rfm_df["frequency"].astype(float)))
    features["monetary"] = np.log1p(_clip_outliers(rfm_df["monetary"].astype(float)))
    return features


def compute_silhouette_for_range(rfm_df: pd.DataFrame, k_values: list[int]) -> dict[int, float]:
    """Compute silhouette scores for several candidate k values, reusing the
    same feature preparation as the main pipeline."""
    features = _prepare_features(rfm_df)
    scaled = StandardScaler().fit_transform(features)

    scores = {}
    for k in k_values:
        if k < 2 or k >= len(rfm_df):
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        sample_size = _silhouette_sample_size(len(scaled))
        scores[k] = round(
            silhouette_score(scaled, labels, sample_size=sample_size, random_state=RANDOM_STATE), 3
        )
    return scores


def _relative_goodness(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    Convert per-cluster mean values into a 0-1 'goodness' score relative to
    the other clusters in this run: 1.0 = best cluster on this dimension,
    0.0 = worst. Ties are averaged. This is what lets the labeling rule work
    the same way regardless of the absolute scale of the data.
    """
    ranks = values.rank(method="average", ascending=not higher_is_better)
    n = len(values)
    if n <= 1:
        return pd.Series(1.0, index=values.index)
    # rank 1 (best) -> goodness 1.0 ; rank n (worst) -> goodness 0.0
    return 1.0 - (ranks - 1) / (n - 1)


def _bucket(score: float) -> str:
    """Turn a 0-1 goodness score into a coarse 'high' / 'medium' / 'low' bucket."""
    if score >= 0.6:
        return "high"
    if score <= 0.4:
        return "low"
    return "medium"


# Explicit business rule table mapping (recency_bucket, engagement_bucket) to
# a segment name. Engagement combines Frequency and Monetary goodness, since
# a cluster that buys often but spends little (or vice versa) still reads as
# a "loyal" or "promising" customer rather than being split into two
# dimensions the business would need to reconcile manually.
_SEGMENT_RULES = {
    ("high", "high"): SEGMENT_CHAMPIONS,
    ("high", "medium"): SEGMENT_LOYAL,
    ("high", "low"): SEGMENT_POTENTIAL,
    ("medium", "high"): SEGMENT_LOYAL,
    ("medium", "medium"): SEGMENT_POTENTIAL,
    ("medium", "low"): SEGMENT_POTENTIAL,
    ("low", "high"): SEGMENT_AT_RISK,
    ("low", "medium"): SEGMENT_AT_RISK,
    ("low", "low"): SEGMENT_LOST,
}


def build_cluster_profiles(rfm_with_cluster: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mean RFM values per cluster and compute the relative goodness
    scores used for business labeling.
    """
    profiles = rfm_with_cluster.groupby(CLUSTER).agg(
        n_customers=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    )
    profiles["pct_customers"] = 100.0 * profiles["n_customers"] / profiles["n_customers"].sum()

    profiles["recency_goodness"] = _relative_goodness(profiles["avg_recency"], higher_is_better=False)
    profiles["frequency_goodness"] = _relative_goodness(profiles["avg_frequency"], higher_is_better=True)
    profiles["monetary_goodness"] = _relative_goodness(profiles["avg_monetary"], higher_is_better=True)
    profiles["engagement_goodness"] = (profiles["frequency_goodness"] + profiles["monetary_goodness"]) / 2

    profiles["recency_bucket"] = profiles["recency_goodness"].apply(_bucket)
    profiles["engagement_bucket"] = profiles["engagement_goodness"].apply(_bucket)

    return profiles.reset_index()


def assign_segment_labels(cluster_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the business rule table to each cluster's (recency_bucket,
    engagement_bucket) pair to produce a human-readable segment name.

    If two clusters land on the same rule (and therefore the same label),
    they are intentionally left with the same name - the business
    interpretation is that they represent the same customer archetype, even
    though K-Means found them to be statistically distinct groups. Everywhere
    downstream, results are aggregated by segment name, so this merges
    cleanly.
    """
    profiles = cluster_profiles.copy()
    profiles[SEGMENT] = profiles.apply(
        lambda row: _SEGMENT_RULES[(row["recency_bucket"], row["engagement_bucket"])],
        axis=1,
    )
    return profiles


def run_segmentation(rfm_df: pd.DataFrame, k: int) -> SegmentationResult:
    """
    Run the full segmentation pipeline on a customer-level RFM DataFrame.

    Args:
        rfm_df: output of rfm.compute_rfm() - must contain customer_id,
            recency, frequency, monetary.
        k: number of clusters for K-Means.

    Returns:
        SegmentationResult with the segmented customer table, per-cluster
        profile/labeling table, silhouette score, and k used.
    """
    if k < 2:
        raise ValueError("Number of clusters must be at least 2.")
    if k >= len(rfm_df):
        raise ValueError(
            f"Number of clusters ({k}) must be smaller than the number of "
            f"customers ({len(rfm_df)})."
        )

    features = _prepare_features(rfm_df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    clusters = model.fit_predict(scaled)
    sample_size = _silhouette_sample_size(len(scaled))
    sil_score = round(
        silhouette_score(scaled, clusters, sample_size=sample_size, random_state=RANDOM_STATE), 3
    )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(scaled)

    result_df = rfm_df.copy()
    result_df[CLUSTER] = clusters
    result_df["pca_1"] = coords[:, 0]
    result_df["pca_2"] = coords[:, 1]

    profiles = build_cluster_profiles(result_df)
    profiles = assign_segment_labels(profiles)

    cluster_to_segment = dict(zip(profiles[CLUSTER], profiles[SEGMENT]))
    result_df[SEGMENT] = result_df[CLUSTER].map(cluster_to_segment)

    return SegmentationResult(
        segmented_df=result_df,
        cluster_profiles=profiles,
        silhouette=sil_score,
        k=k,
    )
