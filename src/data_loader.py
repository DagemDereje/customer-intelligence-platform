"""
Data loading and schema validation for the Customer Intelligence Platform.

Responsibilities:
- Read a raw transactions CSV (file path or file-like object).
- Normalize common column-name variations to a canonical schema.
- Validate that all required columns are present before any analysis runs.

This module does NOT clean the data (see preprocessing.py) - it only makes
sure the incoming file has the columns we need, under whatever name the
user happened to use.
"""

from __future__ import annotations

import re
from typing import Union

import pandas as pd

# Canonical column names used everywhere downstream.
CUSTOMER_ID = "customer_id"
INVOICE_NO = "invoice_no"
INVOICE_DATE = "invoice_date"
QUANTITY = "quantity"
UNIT_PRICE = "unit_price"
DESCRIPTION = "description"
COUNTRY = "country"

REQUIRED_COLUMNS = [CUSTOMER_ID, INVOICE_NO, INVOICE_DATE, QUANTITY, UNIT_PRICE]
OPTIONAL_COLUMNS = [DESCRIPTION, COUNTRY]

# Maps a normalized (lowercased, non-alphanumeric stripped) source column name
# to its canonical name. This intentionally stays a flat lookup table rather
# than a "smart" schema-inference system, per project scope.
_COLUMN_ALIASES = {
    # customer id
    "customerid": CUSTOMER_ID,
    "customer_id": CUSTOMER_ID,
    "custid": CUSTOMER_ID,
    "clientid": CUSTOMER_ID,
    # invoice / order number
    "invoiceno": INVOICE_NO,
    "invoice_no": INVOICE_NO,
    "invoicenumber": INVOICE_NO,
    "invoice_number": INVOICE_NO,
    "orderid": INVOICE_NO,
    "order_id": INVOICE_NO,
    "transactionid": INVOICE_NO,
    "transaction_id": INVOICE_NO,
    # invoice date
    "invoicedate": INVOICE_DATE,
    "invoice_date": INVOICE_DATE,
    "transactiondate": INVOICE_DATE,
    "transaction_date": INVOICE_DATE,
    "orderdate": INVOICE_DATE,
    "order_date": INVOICE_DATE,
    "date": INVOICE_DATE,
    # quantity
    "quantity": QUANTITY,
    "qty": QUANTITY,
    # unit price
    "unitprice": UNIT_PRICE,
    "unit_price": UNIT_PRICE,
    "price": UNIT_PRICE,
    "itemprice": UNIT_PRICE,
    # description
    "description": DESCRIPTION,
    "productname": DESCRIPTION,
    "product_name": DESCRIPTION,
    "itemname": DESCRIPTION,
    "product": DESCRIPTION,
    # country / region
    "country": COUNTRY,
    "region": COUNTRY,
    "customercountry": COUNTRY,
}


class SchemaValidationError(ValueError):
    """Raised when an uploaded dataset is missing required columns."""

    def __init__(self, missing_columns: list[str], found_columns: list[str]):
        self.missing_columns = missing_columns
        self.found_columns = found_columns
        message = (
            "The uploaded file is missing required column(s): "
            f"{', '.join(missing_columns)}. "
            f"Columns found in file: {', '.join(found_columns)}."
        )
        super().__init__(message)


def _normalize_name(name: str) -> str:
    """Lowercase and strip non-alphanumeric characters for alias matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to canonical names where a known alias is recognized.
    Unrecognized columns are left untouched (and simply ignored downstream).
    """
    rename_map = {}
    for col in df.columns:
        key = _normalize_name(col)
        if key in _COLUMN_ALIASES:
            rename_map[col] = _COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)


def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check whether all required canonical columns are present.

    Returns:
        (is_valid, missing_columns)
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return (len(missing) == 0, missing)


def read_csv(source: Union[str, "pd.io.common.IOHandleLike", object]) -> pd.DataFrame:
    """
    Read a CSV from a path or file-like object (e.g. a Streamlit UploadedFile).
    Raises a clear ValueError if the file is empty or unreadable.
    """
    try:
        df = pd.read_csv(source, encoding="latin1")
    except UnicodeDecodeError:
        # fall back for files not saved as latin-1 / utf-8 mixed encodings
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, encoding="utf-8")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded file is empty.") from exc

    if df.empty:
        raise ValueError("The uploaded file contains no rows.")

    return df


def load_and_validate(source: Union[str, object]) -> pd.DataFrame:
    """
    Full load step: read CSV, normalize column names, validate schema.

    Raises:
        ValueError: if the file is empty/unreadable.
        SchemaValidationError: if required columns are missing after
            attempting to map common aliases.

    Returns:
        DataFrame with canonical column names (not yet cleaned).
    """
    df = read_csv(source)
    df = normalize_columns(df)
    is_valid, missing = validate_schema(df)
    if not is_valid:
        raise SchemaValidationError(missing, list(df.columns))
    return df
