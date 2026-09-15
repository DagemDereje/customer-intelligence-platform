import numpy as np
import pandas as pd
import pytest

from src.segmentation import (
    run_segmentation,
    compute_silhouette_for_range,
    SEGMENT_CHAMPIONS,
    SEGMENT_LOST,
)


def _two_group_rfm(n_per_group: int = 30) -> pd.DataFrame:
    """
    Build a synthetic but well-separated RFM dataset with two obvious
    archetypes: recent/frequent/high-spend "champion-like" customers, and
    long-inactive/rare/low-spend "lost-like" customers.
    """
    rng = np.random.default_rng(0)

    champions = pd.DataFrame(
        {
            "customer_id": range(0, n_per_group),
            "recency": rng.integers(1, 10, n_per_group),
            "frequency": rng.integers(15, 25, n_per_group),
            "monetary": rng.uniform(3000, 6000, n_per_group),
        }
    )
    lost = pd.DataFrame(
        {
            "customer_id": range(n_per_group, 2 * n_per_group),
            "recency": rng.integers(300, 400, n_per_group),
            "frequency": rng.integers(1, 3, n_per_group),
            "monetary": rng.uniform(10, 100, n_per_group),
        }
    )
    return pd.concat([champions, lost], ignore_index=True)


def test_run_segmentation_rejects_k_less_than_two():
    rfm = _two_group_rfm()
    with pytest.raises(ValueError):
        run_segmentation(rfm, k=1)


def test_run_segmentation_rejects_k_greater_than_customers():
    rfm = _two_group_rfm(n_per_group=5)  # 10 customers total
    with pytest.raises(ValueError):
        run_segmentation(rfm, k=20)


def test_run_segmentation_produces_requested_number_of_clusters():
    rfm = _two_group_rfm()
    result = run_segmentation(rfm, k=2)
    assert result.segmented_df["cluster"].nunique() == 2
    assert result.k == 2


def test_run_segmentation_assigns_a_segment_to_every_customer():
    rfm = _two_group_rfm()
    result = run_segmentation(rfm, k=2)
    assert result.segmented_df["segment"].notna().all()


def test_cluster_profiles_percentages_sum_to_100():
    rfm = _two_group_rfm()
    result = run_segmentation(rfm, k=2)
    assert abs(result.cluster_profiles["pct_customers"].sum() - 100.0) < 0.01


def test_business_rules_label_obvious_archetypes_correctly():
    """
    With two clearly separated groups (recent/frequent/high-spend vs.
    long-inactive/rare/low-spend), the rule-based labeling should recognize
    the best group as Champions and the worst as Lost.
    """
    rfm = _two_group_rfm()
    result = run_segmentation(rfm, k=2)
    segments_present = set(result.segmented_df["segment"].unique())
    assert SEGMENT_CHAMPIONS in segments_present
    assert SEGMENT_LOST in segments_present

    champion_customers = result.segmented_df[result.segmented_df["segment"] == SEGMENT_CHAMPIONS]
    lost_customers = result.segmented_df[result.segmented_df["segment"] == SEGMENT_LOST]
    assert champion_customers["monetary"].mean() > lost_customers["monetary"].mean()
    assert champion_customers["recency"].mean() < lost_customers["recency"].mean()


def test_compute_silhouette_for_range_skips_invalid_k_values():
    rfm = _two_group_rfm(n_per_group=5)  # 10 customers
    scores = compute_silhouette_for_range(rfm, [1, 2, 3, 20])
    # k=1 is invalid (<2), k=20 is invalid (>= n customers)
    assert 1 not in scores
    assert 20 not in scores
    assert 2 in scores and 3 in scores


def test_silhouette_score_is_reasonably_high_for_well_separated_data():
    rfm = _two_group_rfm()
    result = run_segmentation(rfm, k=2)
    # Two very distinct, well-separated groups should cluster cleanly.
    assert result.silhouette > 0.5
