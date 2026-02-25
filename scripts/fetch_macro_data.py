# Databricks notebook source
# MAGIC %md
# MAGIC # Macro Data Fetch — Statistics Canada
# MAGIC
# MAGIC Fetches three Statistics Canada time series and appends them to `macro_indicators_raw`:
# MAGIC - **14-10-0017-01**: Labour force characteristics by province (unemployment rate, monthly)
# MAGIC - **18-10-0205-01**: New housing price index by province (monthly)
# MAGIC - **34-10-0158-01**: Housing starts by province (monthly)
# MAGIC
# MAGIC **Pattern:** Each run appends with a unique `batch_id`. The SCD Type 2 silver layer
# MAGIC (`silver_macro_indicators`) captures any value revisions StatCan publishes to prior months.
# MAGIC
# MAGIC **No authentication required** — StatCan CSV API is publicly accessible.

# COMMAND ----------

dbutils.widgets.text("catalog", "my_catalog",        "UC Catalog")
dbutils.widgets.text("schema",  "actuarial_workshop", "UC Schema")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA  = dbutils.widgets.get("schema")

# COMMAND ----------

import io
import uuid
import zipfile
from datetime import datetime

import pandas as pd
import requests
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)

# COMMAND ----------

# ── StatCan CSV download API ──────────────────────────────────────────────────
# Full-table bulk CSV download (no authentication required).
# URL format: https://www150.statcan.gc.ca/n1/tbl/csv/{pid}-eng.zip
# The PID is the table ID with dashes removed and the trailing edition group (-01) dropped.
STATCAN_BASE = "https://www150.statcan.gc.ca/n1/tbl/csv"

# Three tables to fetch. `filters` are applied in order — each is a (column, value) pair.
# StatCan CSV columns vary per table; unknown filter columns are skipped with a warning.
STATCAN_TABLES = {
    # Table 14-10-0017-01: Labour force characteristics by province, monthly.
    # Monthly YYYY-MM REF_DATE; includes Gender, Age group, and Labour force characteristics columns.
    "14100017": {
        "source_table":   "14-10-0017-01",
        "indicator_name": "unemployment_rate",
        "unit_keyword":   "Percent",
        "filters": [
            ("Labour force characteristics", "Unemployment rate"),
            ("Gender",    "Total - Gender"),   # column is "Gender" not "Sex" in this table
            ("Age group", "15 years and over"),
        ],
    },
    # Table 18-10-0205-01: New housing price index (total, house + land) by province, monthly.
    "18100205": {
        "source_table":   "18-10-0205-01",
        "indicator_name": "hpi_index",
        "unit_keyword":   "Index",
        "filters": [
            ("New housing price indexes", "Total (house and land)"),
        ],
    },
    # Table 34-10-0158-01: Housing starts by province, monthly (CMHC survey).
    # Single row per GEO × REF_DATE — no extra filter columns.
    "34100158": {
        "source_table":   "34-10-0158-01",
        "indicator_name": "housing_starts",
        "unit_keyword":   "Units",
        "filters": [],   # no further breakdown; each province has one row per month
    },
}

# Map StatCan GEO province names → workshop region names (underscore-separated)
PROVINCE_MAP = {
    "Ontario":                   "Ontario",
    "Quebec":                    "Quebec",
    "British Columbia":          "British_Columbia",
    "Alberta":                   "Alberta",
    "Manitoba":                  "Manitoba",
    "Saskatchewan":              "Saskatchewan",
    "New Brunswick":             "New_Brunswick",
    "Nova Scotia":               "Nova_Scotia",
    "Prince Edward Island":      "Prince_Edward_Island",
    "Newfoundland and Labrador": "Newfoundland",
}

# COMMAND ----------

def fetch_statcan_csv(pid_nodashes: str) -> pd.DataFrame:
    """Download a StatCan table ZIP and return the data CSV as a DataFrame."""
    url = f"{STATCAN_BASE}/{pid_nodashes}-eng.zip"
    print(f"  Fetching {url} ...")
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()

    # The ZIP contains two files: the data CSV and a _MetaData.csv.
    # We want the data file (the one that does NOT end with _MetaData.csv).
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        data_name = next(
            name for name in zf.namelist()
            if not name.endswith("_MetaData.csv") and name.endswith(".csv")
        )
        with zf.open(data_name) as f:
            df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)

    print(f"    Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def parse_statcan_table(raw_df: pd.DataFrame, table_config: dict, batch_id: str) -> pd.DataFrame:
    """
    Filter a StatCan raw DataFrame to the target provinces and indicator breakdown.
    Returns rows in the macro_indicators_raw schema.
    """
    df = raw_df.copy()

    # 1. Filter to the 10 provinces we care about (drop Canada total, CMAs, territories)
    if "GEO" not in df.columns:
        raise ValueError(f"No GEO column. Available: {df.columns.tolist()}")
    df = df[df["GEO"].isin(PROVINCE_MAP.keys())].copy()

    # 2. Apply table-specific row filters (skip if column not present)
    for fc, fv in table_config.get("filters", []):
        if fc in df.columns:
            df = df[df[fc] == fv].copy()
        else:
            print(f"    Note: filter column '{fc}' not in table — skipping")

    # 3. Require REF_DATE and VALUE
    if "REF_DATE" not in df.columns or "VALUE" not in df.columns:
        raise ValueError(f"Missing REF_DATE or VALUE. Columns: {df.columns.tolist()}")

    # 4. Numeric value (StatCan uses 'x' / 'F' suppression codes → NaN)
    df["value_clean"] = pd.to_numeric(df["VALUE"], errors="coerce")

    # 5. Deduplicate: if multiple rows remain per (GEO, REF_DATE), take mean
    df = (df.groupby(["GEO", "REF_DATE"], as_index=False)["value_clean"].mean())

    # 6. Unit string
    unit_str = ""
    if "UOM" in raw_df.columns:
        # Take the most common unit across all rows after province filter (before row filters)
        province_rows = raw_df[raw_df["GEO"].isin(PROVINCE_MAP.keys())]
        if len(province_rows) > 0:
            unit_str = str(province_rows["UOM"].mode().iloc[0]) if len(province_rows) > 0 else ""

    ingested_at = datetime.utcnow().isoformat()
    result = pd.DataFrame({
        "source_table":   table_config["source_table"],
        "province":       df["GEO"].values,
        "ref_date":       df["REF_DATE"].astype(str).values,
        "indicator_name": table_config["indicator_name"],
        "value":          df["value_clean"].values,
        "unit":           unit_str,
        "ingested_at":    ingested_at,
        "batch_id":       batch_id,
    })

    # Drop StatCan-suppressed cells (no value available)
    result = result.dropna(subset=["value"]).reset_index(drop=True)
    print(f"    → {len(result):,} rows for indicator '{table_config['indicator_name']}'")
    return result

# COMMAND ----------

# ── Run fetch + parse for all three StatCan tables ────────────────────────────
batch_id = f"{datetime.utcnow().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
print(f"Batch ID: {batch_id}\n")

all_rows = []
for pid, config in STATCAN_TABLES.items():
    try:
        raw_df  = fetch_statcan_csv(pid)
        parsed  = parse_statcan_table(raw_df, config, batch_id)
        all_rows.append(parsed)
        print(f"  ✓  {config['source_table']}  ({config['indicator_name']}) → {len(parsed):,} rows\n")
    except Exception as e:
        print(f"  ✗  {config['source_table']} FAILED: {type(e).__name__}: {e}\n")

if not all_rows:
    print("\nWARNING: All StatCan fetches failed.")
    print("  This is expected on workspaces with restricted outbound internet access.")
    print("  The DLT pipeline will process an empty macro_indicators_raw.")
    print("  Module 4 SARIMAX will fall back to baseline SARIMA (no macro exog).")
    print("  To populate macro data manually: run this notebook on a cluster with")
    print("  outbound HTTPS access to www150.statcan.gc.ca")

combined_pdf = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
if all_rows:
    print(f"Total rows to append: {len(combined_pdf):,}")

# COMMAND ----------

# ── Schema and write ──────────────────────────────────────────────────────────
MACRO_RAW_SCHEMA = StructType([
    StructField("source_table",   StringType(), False),
    StructField("province",       StringType(), False),
    StructField("ref_date",       StringType(), False),
    StructField("indicator_name", StringType(), False),
    StructField("value",          DoubleType(), True),
    StructField("unit",           StringType(), True),
    StructField("ingested_at",    StringType(), False),
    StructField("batch_id",       StringType(), False),
])

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

if all_rows:
    # Append fetched rows to macro_indicators_raw
    macro_sdf = spark.createDataFrame(combined_pdf, schema=MACRO_RAW_SCHEMA)
    (macro_sdf.write
        .format("delta")
        .mode("append")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.macro_indicators_raw"))
    total = spark.table(f"{CATALOG}.{SCHEMA}.macro_indicators_raw").count()
    print(f"\nAppended {len(combined_pdf):,} rows → {CATALOG}.{SCHEMA}.macro_indicators_raw")
    print(f"Table total: {total:,} rows")
    print(f"\nIndicator breakdown:")
    display(
        spark.table(f"{CATALOG}.{SCHEMA}.macro_indicators_raw")
        .groupBy("indicator_name")
        .agg(F.count("*").alias("rows"), F.countDistinct("province").alias("provinces"),
             F.min("ref_date").alias("first_date"), F.max("ref_date").alias("last_date"))
        .orderBy("indicator_name")
    )
else:
    # Ensure the landing zone table exists so DLT streaming source doesn't fail.
    # The DLT bronze_macro_indicators table will be empty but won't error.
    if not spark.catalog.tableExists(f"{CATALOG}.{SCHEMA}.macro_indicators_raw"):
        (spark.createDataFrame([], MACRO_RAW_SCHEMA).write
             .format("delta")
             .saveAsTable(f"{CATALOG}.{SCHEMA}.macro_indicators_raw"))
        print(f"Created empty macro_indicators_raw → {CATALOG}.{SCHEMA}.macro_indicators_raw")
    else:
        print(f"macro_indicators_raw already exists — no new rows added.")
    print("Macro data will be empty. SARIMAX will fall back to baseline SARIMA.")
