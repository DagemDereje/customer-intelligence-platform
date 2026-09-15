# Customer Intelligence Platform

An end-to-end customer analytics system that transforms raw transaction data into actionable customer segments and business insights - built as an interactive Streamlit dashboard.

## Overview

Businesses sitting on transaction logs often can't easily answer: *who are our best customers, who's slipping away, and what should we do about it?* This project answers that with a practical, unsupervised-learning pipeline:

```
Transaction Data → Data Cleaning → RFM Analysis → Customer Features → K-Means Clustering → Segment Profiling → Business Insights → Interactive Dashboard
```

Upload any compatible transaction CSV (or use the included demo dataset) and get customer segments, RFM visualizations, and rule-based business recommendations in seconds - no machine-learning knowledge required to use it.

## Key Features

- **Automatic data cleaning** with a transparent, step-by-step report of what was removed and why
- **RFM analysis** (Recency, Frequency, Monetary) computed per customer
- **K-Means segmentation** with outlier clipping, log-transformation, and standardization
- **Business-friendly segment names** (Champions, Loyal Customers, Potential Loyalists, At Risk, Lost) derived from explicit, documented rules applied to cluster characteristics - not arbitrary cluster numbers
- **Silhouette score evaluation**, including a comparison chart across candidate cluster counts
- **PCA visualization** of the customer feature space
- **Dynamic, data-driven business insights and an executive summary** - nothing hard-coded
- **Customer Explorer** to search individual customers and see their segment/behavior
- **At-Risk/Lost customer export** and full segmentation CSV download
- **Upload-your-own-CSV** mode with flexible column-name matching and clear validation errors

## Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** - data manipulation
- **Scikit-learn** - StandardScaler, K-Means, PCA, Silhouette Score
- **Plotly** - interactive visualizations
- **Streamlit** - web application
- **Pytest** - unit testing

## Methodology

1. **Data Preprocessing** - remove rows with missing Customer IDs, unparseable dates, cancelled invoices, non-positive quantity/price, exact duplicates, and non-positive revenue. Every step's row-count impact is logged and shown in the app.
2. **RFM** - Recency (days since last purchase), Frequency (distinct orders), Monetary (total revenue), computed relative to an analysis date of `max(invoice date) + 1 day` (never using "future" information).
3. **Outlier handling** - 1st/99th percentile clipping on R/F/M before transformation, to limit the influence of extreme values without discarding valuable customers.
4. **Transformation** - `log1p` on Frequency and Monetary (typically right-skewed in transaction data).
5. **Standardization** - `StandardScaler` on the resulting features, since K-Means is distance-based.
6. **K-Means** - user-selectable cluster count (3-6, default 5).
7. **Silhouette Score** - reported for the chosen k, with an optional comparison chart across k=3..6.
8. **PCA** - reduces the standardized feature space to 2D purely for visualization; it does not perform the clustering itself.
9. **Business labeling** - each cluster's mean R/F/M is scored relative to the other clusters found in that run, bucketed into high/medium/low, and mapped to a segment name via an explicit rule table. This is a documented, post-hoc business interpretation of the K-Means output, not a label the model learns directly.

## Project Structure

```
customer-intelligence-platform/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── sample_transactions.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # CSV reading, column-name mapping, schema validation
│   ├── preprocessing.py     # data cleaning + cleaning report
│   ├── rfm.py                # Recency/Frequency/Monetary computation
│   ├── segmentation.py       # outlier handling, transform, scale, K-Means, silhouette, PCA, labeling
│   ├── insights.py           # rule-based business insights & recommendations
│   └── visualizations.py     # Plotly figure builders
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_rfm.py
│   └── test_segmentation.py
│
└── screenshots/
```

## Installation

```bash
git clone <repository-url>
cd customer-intelligence-platform

python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app opens with the demo dataset loaded automatically - no setup required. Switch to "Upload CSV" in the sidebar to analyze your own transaction data.

## Testing

```bash
pytest tests/ -v
```

21 tests cover schema validation, data cleaning, RFM calculations, and segmentation behavior (cluster counts, business-rule labeling, silhouette scoring, and error handling for edge cases like too-small datasets or an invalid cluster count).

## Dataset

The included `data/sample_transactions.csv` is a representative sample (700 customers, ~18,900 transaction line items) drawn from the well-known **UCI "Online Retail" dataset** - real transaction records from a UK-based online gift retailer (Dec 2010 - Dec 2011). The sample intentionally retains some of the source data's real-world messiness (a handful of missing Customer IDs, cancelled orders, and duplicate rows) so the cleaning step has genuine, visible work to do. It was stratified by customer purchase-activity level so the demo shows a realistic spread of behaviors - from one-time shoppers to high-frequency, high-spend customers.

If you upload your own CSV, the app expects transaction-level data with (at minimum) a Customer ID, an Invoice/Order/Transaction number, an Invoice/Transaction date, a Quantity, and a Unit Price. Common column-name variations (e.g. `CustomerID`, `Customer_ID`, `customer id`) are automatically recognized.

## Example Use Case

A retail or e-commerce business could run this on their transaction export to quickly identify which customers deserve VIP treatment, which loyal customers are worth cross-selling to, and - critically - which previously valuable customers have gone quiet and are worth a retention campaign before they're lost for good.

## Limitations

- RFM is purely behavior-based; it does not explain *why* a customer behaves a certain way.
- K-Means requires choosing a cluster count in advance; the silhouette-score comparison helps but doesn't remove the judgment call.
- Business recommendations are heuristics based on common RFM segmentation practice, not predictions from a trained model, and are not guaranteed to produce specific outcomes.
- The public dataset used for the demo (a single UK gift retailer) may not represent every business's customer behavior patterns.
- Segment names for two statistically distinct clusters can occasionally merge under the same business label (e.g. two different "Champions" clusters) when their relative RFM characteristics fall into the same rule bucket - this is an intentional simplification for business interpretability.

## Future Improvements

- Customer lifetime value (CLV) modeling
- Churn prediction
- Product recommendation engine
- Automated marketing-campaign integration
- Real-time transaction ingestion

---

## Portfolio Description

**One-line description:**
An end-to-end customer analytics platform that turns raw transaction data into actionable customer segments using RFM analysis and K-Means clustering.

**Short portfolio description:**
Built an interactive Streamlit application that analyzes retail transaction data and automatically segments customers (Champions, Loyal, At Risk, Lost, etc.) using RFM analysis and K-Means clustering. The app cleans raw transaction data end-to-end, evaluates cluster quality with silhouette scoring, visualizes results with PCA and Plotly, and generates data-driven business insights and recommendations - all without requiring the end user to understand machine learning.

**Technical description:**
Implemented a modular Python pipeline (pandas/NumPy for cleaning and feature engineering, scikit-learn for StandardScaler/K-Means/PCA/silhouette evaluation, Plotly for interactive visualization) wrapped in a Streamlit dashboard. Includes outlier-robust preprocessing (percentile clipping + log transformation), a rule-based post-hoc labeling system that translates arbitrary cluster indices into business-meaningful segment names, flexible CSV schema validation for user uploads, and a pytest suite covering the core data and ML logic.

**Key skills demonstrated:**
- Data cleaning & validation
- Feature engineering (RFM)
- Unsupervised learning (K-Means, PCA)
- Model evaluation (silhouette score)
- Interactive data visualization (Plotly)
- Business interpretation of ML output
- Full-stack analytics app development (Streamlit)
- Unit testing (pytest)

## Recommended Screenshots

For your portfolio, capture:
1. Overview page - KPI cards + segment distribution donut + executive summary
2. Customer Segments page - RFM scatter plots and PCA visualization
3. Segment Details - a selected segment's profile and recommended action
4. Customer Explorer - an individual customer lookup
5. Methodology page - showing the documented pipeline
