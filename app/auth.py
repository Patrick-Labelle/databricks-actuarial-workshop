import base64
import json

import streamlit as st

# ─── Lazy import for psycopg2 (may not be available in all environments) ─────
_psycopg2 = None
_psycopg2_error = None


def get_psycopg2():
    """Lazy-load psycopg2 on first use; cache the result."""
    global _psycopg2, _psycopg2_error
    if _psycopg2 is not None:
        return _psycopg2
    if _psycopg2_error is not None:
        return None
    try:
        import psycopg2 as _pg
        _psycopg2 = _pg
        return _psycopg2
    except ImportError as exc:
        _psycopg2_error = str(exc)
        return None


def get_psycopg2_error():
    return _psycopg2_error


# ─── Lazy WorkspaceClient ────────────────────────────────────────────────────
_workspace_client = None
_auth_init_error = None


def get_workspace_client():
    """Lazily initialise the Databricks WorkspaceClient."""
    global _workspace_client, _auth_init_error
    if _workspace_client is not None:
        return _workspace_client
    if _auth_init_error is not None:
        return None
    try:
        from databricks import sdk
        _workspace_client = sdk.WorkspaceClient()
        return _workspace_client
    except Exception as exc:
        _auth_init_error = str(exc)
        return None


def get_auth_init_error():
    return _auth_init_error


def get_token():
    """Get OAuth token for user-context operations (Lakebase, model serving).
    Prefers the forwarded user token, falls back to the SDK service principal token."""
    user_token = st.context.headers.get("X-Forwarded-Access-Token")
    if user_token:
        return user_token
    w = get_workspace_client()
    if w is not None:
        try:
            if w.config.token:
                return w.config.token
            headers = {}
            w.config.authenticate(headers)
            if "Authorization" in headers:
                return headers["Authorization"].replace("Bearer ", "")
        except Exception:
            pass
    return None


def email_from_token(token: str) -> str:
    """Decode the JWT payload (without verification) to extract the user identity."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("sub", payload.get("email", ""))
    except Exception:
        return ""
