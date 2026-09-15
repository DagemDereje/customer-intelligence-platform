"""
Data cleaning for the Customer Intelligence Platform.

Cleaning is performed as an explicit, ordered sequence of steps. Each step
records how many rows it removed so the application can show the user
exactly what happened to their data - nothing is silently transformed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data_loader import (
    CUSTOMER_ID,
    INVOICE_DATE,
    INVOICE_NO,
    QUANTITY,
    UNIT_PRICE,
)

REVENUE = "revenue"


@dataclass
class CleaningReport:
    """Row-count audit trail produced by clean_transactions()."""

    starting_rows: int
    steps: list[tuple[str, int]] = field(default_factory=list)  # (description, rows_removed)
    ending_rows: int = 0

    def add_step(self, description: str, rows_removed: int) -> None:
        if rows_removed > 0:
            self.steps.append((description, rows_removed))

    @property
    def total_removed(self) -> int:
        return self.starting_rows - self.ending_rows

    @property
    def pct_removed(self) -> float:
        if self.starting_rows == 0:
            return 0.0
        return 100.0 * self.total_removed / self.starting_rows


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Apply a fixed sequence of cleaning rules to a raw (schema-validated)
    transactions DataFrame.

    Steps, in order:
      1. Drop rows with missing Customer ID.
      2. Parse Invoice Date; drop rows where it fails to parse.
      3. Drop cancelled invoices (invoice number starting with 'C').
      4. Drop non-positive quantities.
      5. Drop non-positive unit prices.
      6. Drop exact duplicate rows.
      7. Compute Revenue = Quantity * UnitPrice and drop non-positive revenue
         (a safety net after the two per-field checks above).

    Returns:
        (clean_df, CleaningReport)
    """
    report = CleaningReport(starting_rows=len(df))
    working = df.copy()

    # 1. Missing customer IDs
    before = len(working)
    working = working[working[CUSTOMER_ID].notna()]
    report.add_step("Removed transactions with missing Customer ID", before - len(working))

    # 2. Invalid / unparseable dates
    working[INVOICE_DATE] = pd.to_datetime(working[INVOICE_DATE], errors="coerce")
    before = len(working)
    working = working[working[INVOICE_DATE].notna()]
    report.add_step("Removed transactions with invalid or missing date", before - len(working))

    # 3. Cancelled invoices (convention: invoice number starts with 'C')
    before = len(working)
    is_cancelled = working[INVOICE_NO].astype(str).str.strip().str.upper().str.startswith("C")
    working = working[~is_cancelled]
    report.add_step("Removed cancelled invoices", before - len(working))

    # 4. Invalid (non-positive) quantities
    working[QUANTITY] = pd.to_numeric(working[QUANTITY], errors="coerce")
    before = len(working)
    working = working[working[QUANTITY].notna() & (working[QUANTITY] > 0)]
    report.add_step("Removed non-positive or invalid quantities", before - len(working))

    # 5. Invalid (non-positive) unit prices
    working[UNIT_PRICE] = pd.to_numeric(working[UNIT_PRICE], errors="coerce")
    before = len(working)
    working = working[working[UNIT_PRICE].notna() & (working[UNIT_PRICE] > 0)]
    report.add_step("Removed non-positive or invalid unit prices", before - len(working))

    # 6. Exact duplicate records
    before = len(working)
    working = working.drop_duplicates()
    report.add_step("Removed duplicate transaction records", before - len(working))

    # 7. Revenue safety net
    working[REVENUE] = working[QUANTITY] * working[UNIT_PRICE]
    before = len(working)
    working = working[working[REVENUE] > 0]
    report.add_step("Removed transactions with non-positive revenue", before - len(working))

    working = working.reset_index(drop=True)
    report.ending_rows = len(working)
    return working, report


def check_sufficient_data(df: pd.DataFrame, min_customers: int = 10) -> tuple[bool, str]:
    """
    Guard against running RFM/clustering on a dataset that is too small to
    produce meaningful segments.
    """
    n_customers = df[CUSTOMER_ID].nunique() if CUSTOMER_ID in df.columns else 0
    if len(df) == 0:
        return False, "No transactions remain after cleaning. Please check your data."
    if n_customers < min_customers:
        return (
            False,
            f"Only {n_customers} unique customer(s) remain after cleaning. "
            f"At least {min_customers} are needed for meaningful segmentation.",
        )
    return True, ""
