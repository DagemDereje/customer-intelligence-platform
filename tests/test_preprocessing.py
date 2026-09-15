import pandas as pd
import pytest

from src.data_loader import normalize_columns, validate_schema, SchemaValidationError, load_and_validate
from src.preprocessing import clean_transactions, check_sufficient_data


def _raw_df():
    return pd.DataFrame(
        {
            "InvoiceNo": ["1001", "1001", "1002", "C1003", "1004", "1004", "1005"],
            "CustomerID": [1, 1, 2, 3, None, 5, 6],
            "InvoiceDate": [
                "2023-01-01", "2023-01-01", "2023-01-02", "2023-01-03",
                "2023-01-04", "2023-01-04", "not-a-date",
            ],
            "Quantity": [2, 2, -1, 5, 3, 3, 4],
            "UnitPrice": [10.0, 10.0, 5.0, 0.0, 2.0, 2.0, 3.0],
            "Country": ["UK"] * 7,
        }
    )


def test_normalize_columns_maps_known_aliases():
    df = pd.DataFrame(columns=["CustomerID", "Invoice No", "invoice_date", "QTY", "Unit Price"])
    normalized = normalize_columns(df)
    assert set(normalized.columns) == {"customer_id", "invoice_no", "invoice_date", "quantity", "unit_price"}


def test_validate_schema_detects_missing_columns():
    df = pd.DataFrame(columns=["customer_id", "invoice_no"])  # missing date/qty/price
    is_valid, missing = validate_schema(df)
    assert not is_valid
    assert set(missing) == {"invoice_date", "quantity", "unit_price"}


def test_load_and_validate_raises_on_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"CustomerID": [1, 2], "Amount": [10, 20]}).to_csv(bad_csv, index=False)
    with pytest.raises(SchemaValidationError):
        load_and_validate(str(bad_csv))


def _renamed_raw_df():
    return _raw_df().rename(
        columns={
            "InvoiceNo": "invoice_no",
            "CustomerID": "customer_id",
            "InvoiceDate": "invoice_date",
            "Quantity": "quantity",
            "UnitPrice": "unit_price",
        }
    )


def test_clean_transactions_removes_missing_customer_id():
    df = _renamed_raw_df()
    cleaned, report = clean_transactions(df)
    assert cleaned["customer_id"].notna().all()
    assert any("missing Customer ID" in step for step, _ in report.steps)


def test_clean_transactions_removes_cancelled_and_invalid_rows():
    df = _renamed_raw_df()
    cleaned, report = clean_transactions(df)
    # Row with cancelled invoice (C1003), missing customer id, invalid date,
    # negative quantity, and zero price should all be gone.
    assert not cleaned["invoice_no"].astype(str).str.upper().str.startswith("C").any()
    assert (cleaned["quantity"] > 0).all()
    assert (cleaned["unit_price"] > 0).all()
    assert cleaned["customer_id"].notna().all()
    assert report.starting_rows == 7
    assert report.ending_rows < report.starting_rows


def test_clean_transactions_removes_exact_duplicates():
    df = pd.DataFrame(
        {
            "invoice_no": ["1", "1"],
            "customer_id": [1, 1],
            "invoice_date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "quantity": [2, 2],
            "unit_price": [5.0, 5.0],
        }
    )
    cleaned, report = clean_transactions(df)
    assert len(cleaned) == 1
    assert report.total_removed == 1


def test_check_sufficient_data_flags_small_datasets():
    small_df = pd.DataFrame({"customer_id": [1, 2, 3]})
    is_ok, message = check_sufficient_data(small_df, min_customers=10)
    assert not is_ok
    assert "10" in message


def test_check_sufficient_data_passes_with_enough_customers():
    df = pd.DataFrame({"customer_id": list(range(20))})
    is_ok, message = check_sufficient_data(df, min_customers=10)
    assert is_ok
    assert message == ""
