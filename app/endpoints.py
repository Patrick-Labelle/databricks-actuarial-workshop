import streamlit as st
import pandas as pd

from auth import get_workspace_client, get_auth_init_error
from config import ENDPOINT_NAME, MC_ENDPOINT_NAME
from db import load_chain_ladder_params


def call_frequency_forecast_endpoint(horizon: int) -> pd.DataFrame:
    """Call Frequency Forecaster (SARIMAX) Model Serving endpoint via Databricks SDK."""
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


def call_bootstrap_endpoint(scenario: dict) -> dict | None:
    """Call the Bootstrap Reserve Simulator serving endpoint.

    Returns a dict of reserve risk metrics or None on error.
    """
    w = get_workspace_client()
    if w is None:
        st.error(f"Databricks SDK unavailable: {get_auth_init_error() or 'unknown error'}")
        return None
    # Ensure required fields have defaults
    scenario.setdefault("scenario", "baseline")
    scenario.setdefault("n_replications", 50_000)
    scenario.setdefault("ldf_multiplier", 1.0)
    scenario.setdefault("inflation_adj", 0.0)
    # Per-line IBNR and CV from chain ladder (read from predictions_ldf_volatility)
    cl_params = load_chain_ladder_params()
    for key, value in cl_params.items():
        scenario.setdefault(key, value)
    if not cl_params:
        st.warning("Chain ladder parameters not available — run Module 3 to populate predictions_ldf_volatility.")
    try:
        response = w.serving_endpoints.query(
            name=MC_ENDPOINT_NAME,
            dataframe_records=[scenario],
        )
        predictions = response.predictions
        if predictions:
            p = predictions[0] if isinstance(predictions, list) else predictions
            _str_keys = {"scenario"}
            return {k: v if k in _str_keys else float(v) for k, v in p.items()}
        return None
    except Exception as exc:
        st.warning(f"Bootstrap endpoint unavailable: {exc}")
        return None
