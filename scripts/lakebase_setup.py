#!/usr/bin/env python3
"""
Lakebase Autoscaling setup — run locally from deploy.sh.

Lakebase Autoscaling endpoints authenticate via the `databricks_auth`
PostgreSQL extension, which validates OAuth JWTs (standard RFC 7519 tokens
issued by the workspace OIDC endpoint). Internal Databricks cluster tokens
(from apiToken() or DATABRICKS_TOKEN on job compute) are NOT accepted —
they are opaque credentials that cannot be validated by the extension's
JWT verification logic.

This script therefore runs on the LOCAL MACHINE (not on Databricks compute)
where the Databricks CLI can supply a proper OAuth JWT via
`databricks auth token`.

Steps performed:
  1. Poll the Lakebase endpoint until IDLE or ACTIVE (up to 10 min).
  2. Connect to `databricks_postgres` (default admin database).
  3. Create the workshop database if it doesn't already exist.
  4. Grant CONNECT on the new database to the app service principal.
  5. Connect to the workshop database.
  6. Enable the `databricks_auth` extension.
  7. Create a Postgres role for the app SP via databricks_create_role().
  8. Create the `public.scenario_annotations` table (idempotent).
  9. Grant schema + table + sequence privileges to the app SP.

Usage (called by deploy.sh):
    python3 scripts/lakebase_setup.py \\
        --workspace-host https://your-workspace.cloud.databricks.com \\
        --endpoint-path  projects/.../branches/main/endpoints/primary \\
        --pg-database    actuarial_workshop_db \\
        --app-sp-client-id <sp-client-id>  \\
        [--profile <databricks-cli-profile>]
"""

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

try:
    import psycopg2
    import psycopg2.extensions
except ImportError:
    print(
        "ERROR: psycopg2-binary not installed.\n"
        "       Install with:  pip install psycopg2-binary\n"
        "       Then re-run deploy.sh."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_get(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}")


def get_oauth_jwt(workspace_host: str, profile: str | None) -> str:
    """Return an OAuth JWT from the Databricks CLI for the running user."""
    cmd = ["databricks", "auth", "token", "--host", workspace_host]
    if profile:
        cmd += ["--profile", profile]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not obtain OAuth JWT from Databricks CLI:\n{result.stderr}"
        )
    return json.loads(result.stdout)["access_token"]


def decode_jwt_sub(token: str) -> str:
    """Decode the 'sub' claim from a JWT without verifying the signature."""
    parts = token.split(".")
    if len(parts) < 2:
        return ""
    payload = parts[1] + "=="  # add padding
    try:
        return json.loads(base64.urlsafe_b64decode(payload)).get("sub", "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize the Lakebase database for the actuarial workshop."
    )
    parser.add_argument("--workspace-host", required=True,
                        help="Workspace URL, e.g. https://adb-xxx.azuredatabricks.net")
    parser.add_argument("--endpoint-path", required=True,
                        help="Lakebase endpoint resource path from bundle validate")
    parser.add_argument("--pg-database", required=True,
                        help="Postgres database name to create")
    parser.add_argument("--app-sp-client-id", default="",
                        help="App service principal client ID (UUID)")
    parser.add_argument("--profile", default="",
                        help="Databricks CLI profile name")
    args = parser.parse_args()

    workspace_host = args.workspace_host.rstrip("/")
    profile = args.profile or None

    # ── 0. Get OAuth JWT ─────────────────────────────────────────────────────
    print("==> Getting OAuth JWT from Databricks CLI...")
    token = get_oauth_jwt(workspace_host, profile)
    current_user = decode_jwt_sub(token)
    print(f"    User:  {current_user}")
    print(f"    Token: prefix={token[:6]}, len={len(token)}")

    # ── 1. Poll endpoint until IDLE or ACTIVE (up to 10 min) ─────────────────
    max_wait_s = 600
    poll_s = 20
    waited = 0
    host = None

    print(f"==> Polling Lakebase endpoint: {args.endpoint_path}")
    while waited <= max_wait_s:
        ep = _api_get(
            f"{workspace_host}/api/2.0/postgres/{args.endpoint_path}", token
        )
        state = ep.get("status", {}).get("current_state", "UNKNOWN")
        host = ep.get("status", {}).get("hosts", {}).get("host", "")
        print(f"    State: {state} (waited {waited}s)")
        if state in ("IDLE", "ACTIVE"):
            break
        if state not in ("PROVISIONING", "STARTING", "PENDING", "UNKNOWN"):
            raise RuntimeError(f"Unexpected Lakebase state: {state}. Response: {ep}")
        time.sleep(poll_s)
        waited += poll_s

    if not host:
        raise RuntimeError(f"Lakebase endpoint not ready after {max_wait_s}s.")
    print(f"    Host:  {host}")

    # ── 2–3. Connect to admin DB and create workshop DB ──────────────────────
    print(f"==> Connecting to databricks_postgres ...")
    conn = psycopg2.connect(
        host=host, port=5432, database="databricks_postgres",
        user=current_user, password=token, sslmode="require",
        connect_timeout=30,
    )
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (args.pg_database,))
    if cur.fetchone():
        print(f"    [OK]      Database '{args.pg_database}' already exists.")
    else:
        cur.execute(f'CREATE DATABASE "{args.pg_database}"')
        print(f"    [CREATED] Database '{args.pg_database}' created.")

    conn.close()

    # ── 4–9. Connect to workshop DB; extension; SP role; table; grants ────────
    # NOTE: GRANT CONNECT must happen AFTER databricks_create_role() creates the
    # Postgres role.  The role is created inside the workshop DB, but GRANT
    # CONNECT on a database can be executed from any connection.
    print(f"==> Connecting to {args.pg_database} ...")
    conn = psycopg2.connect(
        host=host, port=5432, database=args.pg_database,
        user=current_user, password=token, sslmode="require",
        connect_timeout=30,
    )
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS databricks_auth")
    conn.commit()
    print("    [OK] Extension 'databricks_auth' enabled.")

    if args.app_sp_client_id:
        cur.execute(
            "SELECT databricks_create_role(%s, %s)",
            (args.app_sp_client_id, "service_principal"),
        )
        conn.commit()
        print(f"    [OK] Postgres role created for SP: {args.app_sp_client_id}")

        # Grant CONNECT now that the role exists.
        cur.execute(
            f'GRANT CONNECT ON DATABASE "{args.pg_database}" TO "{args.app_sp_client_id}"'
        )
        conn.commit()
        print(f"    [OK] Granted CONNECT on '{args.pg_database}' to SP: {args.app_sp_client_id}")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS public.scenario_annotations (
            id          SERIAL      PRIMARY KEY,
            segment_id  TEXT        NOT NULL,
            note        TEXT,
            analyst     TEXT,
            created_at  TIMESTAMP   DEFAULT NOW()
        )
    """)
    conn.commit()
    print("    [OK] Table 'public.scenario_annotations' ensured.")

    if args.app_sp_client_id:
        cur.execute(f'GRANT USAGE ON SCHEMA public TO "{args.app_sp_client_id}"')
        cur.execute(
            f'GRANT SELECT, INSERT ON TABLE public.scenario_annotations TO "{args.app_sp_client_id}"'
        )
        cur.execute(
            f'GRANT USAGE ON SEQUENCE public.scenario_annotations_id_seq TO "{args.app_sp_client_id}"'
        )
        conn.commit()
        print(f"    [OK] Granted schema + table + sequence to SP: {args.app_sp_client_id}")
    else:
        print("    [SKIP] No app_sp_client_id — skipping Lakebase PostgreSQL grants.")

    conn.close()
    print("\n==> Lakebase setup complete!")


if __name__ == "__main__":
    main()
