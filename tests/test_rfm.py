import pandas as pd

from src.rfm import compute_rfm, get_default_analysis_date


def _clean_df():
    # customer 1: two invoices, last on 2023-01-10
    # customer 2: one invoice, last on 2023-01-05
    return pd.DataFrame(
        {
            "invoice_no": ["A1", "A1", "A2", "B1"],
            "customer_id": [1, 1, 1, 2],
            "invoice_date": pd.to_datetime(
                ["2023-01-01", "2023-01-01", "2023-01-10", "2023-01-05"]
            ),
            "quantity": [2, 1, 3, 4],
            "unit_price": [10.0, 10.0, 5.0, 2.0],
            "revenue": [20.0, 10.0, 15.0, 8.0],
        }
    )


def test_default_analysis_date_is_day_after_max_date():
    df = _clean_df()
    analysis_date = get_default_analysis_date(df)
    assert analysis_date == pd.Timestamp("2023-01-11")


def test_compute_rfm_frequency_counts_unique_invoices():
    df = _clean_df()
    rfm = compute_rfm(df)
    cust1 = rfm[rfm["customer_id"] == 1].iloc[0]
    # customer 1 has two distinct invoices (A1, A2) despite 3 line items
    assert cust1["frequency"] == 2


def test_compute_rfm_monetary_sums_revenue_per_customer():
    df = _clean_df()
    rfm = compute_rfm(df)
    cust1 = rfm[rfm["customer_id"] == 1].iloc[0]
    cust2 = rfm[rfm["customer_id"] == 2].iloc[0]
    assert cust1["monetary"] == 45.0
    assert cust2["monetary"] == 8.0


def test_compute_rfm_recency_uses_most_recent_purchase():
    df = _clean_df()
    analysis_date = pd.Timestamp("2023-01-11")
    rfm = compute_rfm(df, analysis_date=analysis_date)
    cust1 = rfm[rfm["customer_id"] == 1].iloc[0]
    cust2 = rfm[rfm["customer_id"] == 2].iloc[0]
    # customer 1 last purchased 2023-01-10 -> 1 day recency
    assert cust1["recency"] == 1
    # customer 2 last purchased 2023-01-05 -> 6 days recency
    assert cust2["recency"] == 6


def test_compute_rfm_returns_one_row_per_customer():
    df = _clean_df()
    rfm = compute_rfm(df)
    assert len(rfm) == df["customer_id"].nunique()
