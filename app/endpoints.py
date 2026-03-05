import streamlit as st
import pandas as pd

from auth import get_workspace_client, get_auth_init_error
from config import ENDPOINT_NAME, MC_ENDPOINT_NAME


def call_serving_endpoint(horizon: int) -> pd.DataFrame:
    """Call SARIMA Model Serving endpoint via Databricks SDK."""
    w = get_workspace_client()
    if w is None:
        st.error(f"Databricks SDK unavailable: {get_auth_init_error() or 'unknown error'}")
        return pd.DataFrame()
    try:
        response = w.serving_endpoints.query(
            name=ENDPOINT_NAME,
            dataframe_records=[{"horizon": horizon}],
        )
        predictions = response.predictions
        if predictions:
            return pd.DataFrame(predictions)
        return pd.DataFrame()
    except Exception as exc:
        st.warning(f"Endpoint unavailable: {exc}")
        return pd.DataFrame()


def call_monte_carlo_endpoint(scenario: dict) -> dict | None:
    """Call the Monte Carlo serving endpoint with a scenario parameter dict.

    Returns a dict of risk metrics or None on error.
    """
    w = get_workspace_client()
    if w is None:
        st.error(f"Databricks SDK unavailable: {get_auth_init_error() or 'unknown error'}")
        return None
    try:
        response = w.serving_endpoints.query(
            name=MC_ENDPOINT_NAME,
            dataframe_records=[scenario],
        )
        predictions = response.predictions
        if predictions:
            p = predictions[0] if isinstance(predictions, list) else predictions
            _str_keys = {"copula", "model_type", "simulation_mode"}
            return {k: v if k in _str_keys else float(v) for k, v in p.items()}
        return None
    except Exception as exc:
        st.warning(f"Monte Carlo endpoint unavailable: {exc}")
        return None
