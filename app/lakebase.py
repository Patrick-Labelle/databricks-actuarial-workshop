import os

import streamlit as st
import pandas as pd

from auth import get_workspace_client, get_psycopg2, get_psycopg2_error
from config import LAKEBASE_ENDPOINT_PATH, PG_DATABASE_DEFAULT, LAKEBASE_HOST

_lakebase_cred_cache: dict = {"token": None, "expires_at": 0.0}


def _get_lakebase_host() -> str:
    """Return the Lakebase endpoint hostname.

    PGHOST env var takes precedence (can be overridden at runtime).
    Falls back to LAKEBASE_HOST from _bundle_config.py, which is written by
    deploy.sh after lakebase_setup.py resolves and confirms the hostname.
    """
    return os.environ.get("PGHOST") or LAKEBASE_HOST


def _get_lakebase_credential() -> tuple:
    """Return (client_id, token) for the app SP via generate_database_credential.

    Uses w.postgres.generate_database_credential() (Databricks SDK >= 0.81.0).
    The token is used as the Postgres password — Lakebase Autoscaling validates it
    via the databricks_auth extension. Credentials are cached until 60 s before expiry.

    DATABRICKS_CLIENT_ID is injected at runtime by the Databricks Apps platform.
    """
    import time
    now = time.time()
    client_id = os.environ.get("DATABRICKS_CLIENT_ID", "")
    if _lakebase_cred_cache["token"] and _lakebase_cred_cache["expires_at"] > now + 60:
        return client_id, _lakebase_cred_cache["token"]

    w = get_workspace_client()
    if w is None:
        return client_id, None
    try:
        # SDK >= 0.81.0: w.postgres.generate_database_credential(endpoint=<resource_path>)
        cred = w.postgres.generate_database_credential(endpoint=LAKEBASE_ENDPOINT_PATH)
        token = cred.token
    except AttributeError:
        # Fallback: low-level API call (older SDK versions without w.postgres)
        try:
            cred = w.api_client.do(
                "POST",
                "/api/2.0/postgres/generateDatabaseCredential",
                body={"endpoint": LAKEBASE_ENDPOINT_PATH},
            )
            token = cred.get("token", "") if isinstance(cred, dict) else ""
        except Exception:
            return client_id, None
    except Exception:
        return client_id, None
    # Credentials typically expire in 1 hour; cache conservatively
    _lakebase_cred_cache["token"] = token
    _lakebase_cred_cache["expires_at"] = now + 3300  # 55 min
    return client_id, token


def get_lakebase_conn():
    """Connect to the Lakebase Autoscaling endpoint as the app service principal.

    Authentication uses generate_database_credential — the returned token is used
    directly as the Postgres password. The SP role is created by databricks_create_role()
    in app_setup.py (Task 7 of the setup job).

    All DB operations run as the SP. The human analyst's identity is captured from
    the forwarded user token and stored in the `analyst` column.
    """
    psycopg2 = get_psycopg2()
    if psycopg2 is None:
        raise RuntimeError(
            f"psycopg2 is not available ({get_psycopg2_error() or 'unknown reason'}). "
            "Lakebase features are disabled."
        )

    host = _get_lakebase_host()
    if not host:
        raise RuntimeError(
            "Lakebase hostname not configured. "
            "Re-run deploy.sh to regenerate app/_bundle_config.py with LAKEBASE_HOST, "
            "or set the PGHOST environment variable."
        )

    sp_user, sp_token = _get_lakebase_credential()
    if not sp_token:
        raise RuntimeError("Could not obtain Lakebase credential — check DATABRICKS_CLIENT_ID env var.")

    port     = int(os.environ.get("PGPORT", "5432"))
    database = PG_DATABASE_DEFAULT
    sslmode  = os.environ.get("PGSSLMODE", "require")
    return psycopg2.connect(
        host=host, port=port, database=database,
        user=sp_user, password=sp_token, sslmode=sslmode,
    )


_annotations_table_ensured = False


def _ensure_annotations_table():
    """Create scenario_annotations if it doesn't exist (idempotent, runs once per process)."""
    global _annotations_table_ensured
    if _annotations_table_ensured:
        return
    try:
        conn = get_lakebase_conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS public.scenario_annotations (
                id              SERIAL        PRIMARY KEY,
                segment_id      TEXT          NOT NULL,
                note            TEXT,
                analyst         TEXT,
                scenario_type   TEXT,
                adjustment_pct  NUMERIC(10,2),
                approval_status TEXT          DEFAULT 'Draft',
                created_at      TIMESTAMP     DEFAULT NOW()
            )
        """)
        conn.commit()
        conn.close()
        _annotations_table_ensured = True
    except Exception:
        pass  # Non-fatal — table may already exist or Lakebase may be unavailable


def save_scenario_annotation(
    segment_id: str,
    note: str,
    analyst: str,
    scenario_type: str = "",
    adjustment_pct: float | None = None,
    approval_status: str = "Draft",
):
    try:
        _ensure_annotations_table()
        conn = get_lakebase_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO public.scenario_annotations "
            "(segment_id, note, analyst, scenario_type, adjustment_pct, approval_status, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, NOW())",
            (segment_id, note, analyst,
             scenario_type or None,
             adjustment_pct,
             approval_status or "Draft"),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.warning(f"Could not save annotation: {e}")
        return False


def load_annotations(segment_id: str):
    try:
        _ensure_annotations_table()
        conn = get_lakebase_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT analyst, scenario_type, adjustment_pct, approval_status, note, created_at "
            "FROM public.scenario_annotations "
            "WHERE segment_id = %s ORDER BY created_at DESC LIMIT 20",
            (segment_id,)
        )
        rows = cur.fetchall()
        conn.close()
        if rows:
            return pd.DataFrame(
                rows,
                columns=["Analyst", "Type", "Adj %", "Status", "Note", "Created At"],
            )
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
