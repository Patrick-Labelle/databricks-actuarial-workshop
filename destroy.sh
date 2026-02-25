#!/usr/bin/env bash
# destroy.sh — full teardown: workspace assets + bundle destroy
#
# Usage:  ./destroy.sh [--target <target>] [other bundle flags]
#
# Sequence
# --------
#   1. Resolve bundle variables and workspace coordinates via `bundle validate`.
#   2. Remove workspace assets created by the setup job (not managed by the bundle):
#        • UC schema + all tables  (Statement Execution API — auto-starts warehouse)
#        • Online Table            (REST API)
#        • Model Serving endpoint  (REST API)
#        • UC registered model     (REST API)
#        • MLflow experiments      (REST API)
#      Each step prints a clear success/skip/error message. Failures are logged
#      but do not abort the script so the remaining assets are still cleaned up.
#   3. `databricks bundle destroy` — removes bundle-managed resources:
#        App, Lakebase instance, jobs, DLT pipeline.
#   4. Remove the workspace bundle folder left behind by bundle destroy.
#
# Note: The Lakebase PostgreSQL database is dropped implicitly when the Lakebase
# instance is deleted in step 3 — no separate DROP DATABASE step is needed.

set -euo pipefail

# ── Minimum CLI version check ─────────────────────────────────────────────────
_CLI_VER=$(databricks --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "0.0.0")
_CLI_MAJOR=$(echo "$_CLI_VER" | cut -d. -f1)
_CLI_MINOR=$(echo "$_CLI_VER" | cut -d. -f2)
if [ "$_CLI_MAJOR" -lt 1 ] && [ "$_CLI_MINOR" -lt 287 ]; then
    echo "ERROR: Databricks CLI >= 0.287.0 required (found ${_CLI_VER})." >&2
    echo "       Install: https://docs.databricks.com/aws/en/dev-tools/cli/install" >&2
    exit 1
fi

BUNDLE_ARGS=("$@")

# Extract only the flags that `bundle validate` accepts (--target, --profile, --var)
# so that destroy-only flags like --auto-approve don't cause validate to fail.
VALIDATE_ARGS=()
i=0
while [ $i -lt ${#BUNDLE_ARGS[@]} ]; do
    arg="${BUNDLE_ARGS[$i]}"
    case "$arg" in
        --target|-t|--profile|-p|--var)
            VALIDATE_ARGS+=("$arg" "${BUNDLE_ARGS[$((i+1))]}")
            i=$((i+2))
            ;;
        --target=*|--profile=*|--var=*)
            VALIDATE_ARGS+=("$arg")
            i=$((i+1))
            ;;
        *)
            i=$((i+1))
            ;;
    esac
done

# ── Step 1: resolve variables ─────────────────────────────────────────────────
echo "==> Resolving bundle variables..."
VALIDATE_JSON=$(databricks bundle validate --output json "${VALIDATE_ARGS[@]}" 2>/dev/null)

ROOT_PATH=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('workspace',{}).get('root_path',''))")
PROFILE=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('workspace',{}).get('profile','') or '')")
WORKSPACE_HOST=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('workspace',{}).get('host','').rstrip('/'))")
CURRENT_USER=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('workspace',{}).get('current_user',{}).get('userName',''))")
CATALOG=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('variables',{}).get('catalog',{}).get('value',''))")
SCHEMA=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('variables',{}).get('schema',{}).get('value',''))")
ENDPOINT_NAME=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('variables',{}).get('endpoint_name',{}).get('value',''))")
MC_ENDPOINT_NAME=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('variables',{}).get('mc_endpoint_name',{}).get('value',''))")
WAREHOUSE_ID=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('variables',{}).get('warehouse_id',{}).get('value',''))")

PROFILE_ARGS=()
[ -n "$PROFILE" ] && PROFILE_ARGS=(--profile "$PROFILE")

echo "    Host:      ${WORKSPACE_HOST}"
echo "    User:      ${CURRENT_USER}"
echo "    Catalog:   ${CATALOG}.${SCHEMA}"
echo "    Endpoint:  ${ENDPOINT_NAME}"
echo "    Warehouse: ${WAREHOUSE_ID}"

# Obtain a bearer token via the CLI so all REST calls below use the same auth.
TOKEN=$(databricks auth token --host "${WORKSPACE_HOST}" "${PROFILE_ARGS[@]}" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('access_token',''))" 2>/dev/null || echo "")
if [ -z "$TOKEN" ]; then
    echo "ERROR: could not obtain auth token. Check your CLI profile." >&2
    exit 1
fi

# ── Step 2: remove workspace assets ──────────────────────────────────────────
echo ""
echo "==> Removing workspace assets (created by setup job, not bundle-managed)..."

# Helper: DELETE request, prints OK / not-found / error.
# Uses -s (silent) without -f so that curl always writes the 3-digit HTTP status
# to stdout via -w "%{http_code}" regardless of whether the response is 4xx/5xx.
api_delete() {
    local label="$1"
    local path="$2"
    local resp
    resp=$(curl -s -o /dev/null -w "%{http_code}" \
        -X DELETE "${WORKSPACE_HOST}${path}" \
        -H "Authorization: Bearer ${TOKEN}" 2>/dev/null)
    case "$resp" in
        200|204) echo "    [OK]       ${label}" ;;
        404)     echo "    [SKIP]     ${label} — not found" ;;
        *)       echo "    [WARN]     ${label} — unexpected response ${resp}" ;;
    esac
}

# ── 2a. Drop UC schema (CASCADE — removes all tables, views, functions, volumes)
# Uses the Statement Execution API which auto-starts the warehouse if it is
# stopped, so the drop never silently fails due to an inactive warehouse.
echo ""
echo "    Dropping UC schema ${CATALOG}.${SCHEMA} (CASCADE)..."
if [ -n "$WAREHOUSE_ID" ] && [ -n "$CATALOG" ] && [ -n "$SCHEMA" ]; then
    python3 - <<PYEOF
import json, time, sys, urllib.request, urllib.error

host       = "${WORKSPACE_HOST}"
token      = "${TOKEN}"
wh_id      = "${WAREHOUSE_ID}"
catalog    = "${CATALOG}"
schema     = "${SCHEMA}"
headers    = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def req(method, path, body=None):
    url = f"{host}{path}"
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'{}')

# Submit statement — wait_timeout=50s means the API blocks up to 50 s
# (including warehouse auto-start time) before returning a PENDING result.
status, data = req("POST", "/api/2.0/sql/statements", {
    "warehouse_id": wh_id,
    "statement":    f"DROP SCHEMA IF EXISTS \`{catalog}\`.\`{schema}\` CASCADE",
    "wait_timeout": "50s",
    "on_wait_timeout": "CONTINUE",
})

stmt_id = data.get("statement_id", "")
state   = data.get("status", {}).get("state", "UNKNOWN")

# Poll if still running after the initial wait
for _ in range(30):
    if state in ("SUCCEEDED", "FAILED", "CANCELED", "CLOSED"):
        break
    time.sleep(10)
    _, data = req("GET", f"/api/2.0/sql/statements/{stmt_id}")
    state = data.get("status", {}).get("state", "UNKNOWN")

if state == "SUCCEEDED":
    print(f"    [OK]       Schema {catalog}.{schema} dropped (CASCADE)")
elif state in ("FAILED",):
    err = data.get("status", {}).get("error", {})
    print(f"    [WARN]     Schema drop failed: {err.get('message', state)}", file=sys.stderr)
else:
    print(f"    [WARN]     Schema drop ended with state: {state}", file=sys.stderr)
PYEOF
else
    echo "    [SKIP]     No warehouse_id or catalog/schema configured — skipping schema drop"
fi

# ── 2b. Delete Online Table
api_delete "Online Table ${CATALOG}.${SCHEMA}.segment_features_online" \
    "/api/2.0/online-tables/${CATALOG}.${SCHEMA}.segment_features_online"

# ── 2c. Delete Model Serving endpoints (SARIMA + Monte Carlo)
api_delete "Serving endpoint ${ENDPOINT_NAME}" \
    "/api/2.0/serving-endpoints/${ENDPOINT_NAME}"
api_delete "Serving endpoint ${MC_ENDPOINT_NAME}" \
    "/api/2.0/serving-endpoints/${MC_ENDPOINT_NAME}"

# ── 2d. Delete UC registered models (SARIMA + Monte Carlo, all versions)
MODEL_NAME="${CATALOG}.${SCHEMA}.sarima_claims_forecaster"
api_delete "UC model ${MODEL_NAME}" \
    "/api/2.1/unity-catalog/models/${MODEL_NAME}"
MC_MODEL_NAME="${CATALOG}.${SCHEMA}.monte_carlo_portfolio"
api_delete "UC model ${MC_MODEL_NAME}" \
    "/api/2.1/unity-catalog/models/${MC_MODEL_NAME}"

# ── 2e. Delete Lakebase Autoscaling project
# `bundle destroy` cannot delete read-write endpoints via the Terraform provider,
# so we delete the project explicitly here (this cascades to all branches and
# endpoints). The subsequent bundle destroy will see 404s and treat them as
# already-deleted, completing cleanly.
#
# Limitation: if the branch was deployed with is_protected=true, the Lakebase
# REST API will refuse to delete the project (HTTP 400). In that case we print
# a warning and proceed; bundle destroy will also fail for the same resource,
# and a note about manual cleanup is printed at the end.
echo ""
echo "    Deleting Lakebase project actuarial-workshop-lakebase..."
# Use `|| _LAKEBASE_DELETED=$?` so that set -e does NOT abort the script when
# the Python block exits non-zero (protected branch case). Without `||`, bash
# exits immediately on the non-zero exit code before the capture assignment runs.
_LAKEBASE_DELETED=0
python3 - <<PYEOF || _LAKEBASE_DELETED=$?
import urllib.request, urllib.error, json, time, sys

host    = "${WORKSPACE_HOST}"
token   = "${TOKEN}"
headers = {"Authorization": f"Bearer {token}"}

def req(method, path, body=None):
    url  = f"{host}{path}"
    hdrs = dict(headers)
    if body is not None:
        hdrs["Content-Type"] = "application/json"
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read() or b'{}')
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'{}')

PROJECT = "actuarial-workshop-lakebase"

# Delete the project (cascades to branches and endpoints).
# The branch is created without is_protected (defaults to false) so standard
# project deletion works. If a previous deploy used is_protected=true, the
# project delete returns HTTP 400; in that case we print a warning and let
# bundle destroy handle cleanup via the Terraform provider.
# Retry for HTTP 409 (endpoint reconciliation in progress from a previous
# operation — can take several minutes to settle).
deleted = False
for attempt in range(8):
    status, data = req("DELETE", f"/api/2.0/postgres/projects/{PROJECT}")
    if status in (200, 202, 204):
        print("    [OK]       Lakebase project deleted (async deletion may continue in background)")
        deleted = True
        break
    elif status == 404:
        print("    [SKIP]     Lakebase project not found")
        deleted = True
        break
    elif status == 409:
        wait = 20 * (attempt + 1)
        print(f"    [WAIT]     Endpoint reconciliation in progress, retrying in {wait}s...")
        time.sleep(wait)
    elif status == 400 and "protected" in data.get("message", ""):
        # Branch has is_protected=true from a previous deploy — bundle destroy
        # will handle the cleanup via the Terraform provider.
        print("    [WARN]     Protected branch — bundle destroy will handle Lakebase cleanup",
              file=sys.stderr)
        break
    else:
        print(f"    [WARN]     Lakebase project delete returned HTTP {status}: {data.get('message', data)}",
              file=sys.stderr)
        break

if not deleted:
    # Signal to the outer shell that Lakebase was not pre-deleted.
    sys.exit(1)
PYEOF
_LAKEBASE_DELETED=$?

# ── 2f. Delete MLflow experiments
# The setup job creates two experiments under /Users/<user>/actuarial_workshop_*
echo ""
echo "    Deleting MLflow experiments for ${CURRENT_USER}..."
python3 - <<PYEOF
import json, urllib.request, urllib.error, urllib.parse, sys

host    = "${WORKSPACE_HOST}"
token   = "${TOKEN}"
user    = "${CURRENT_USER}"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
prefixes = [
    f"/Users/{user}/actuarial_workshop_sarima_claims_forecaster",
    f"/Users/{user}/actuarial_workshop_claims_sarima",
    f"/Users/{user}/actuarial_workshop_monte_carlo_portfolio",
]

def req(method, path, body=None):
    url = f"{host}{path}"
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'{}')

for name in prefixes:
    status, data = req("GET", f"/api/2.0/mlflow/experiments/get-by-name?experiment_name={urllib.parse.quote(name)}")
    if status == 404 or "experiment" not in data:
        print(f"    [SKIP]     Experiment not found: {name}")
        continue
    exp_id = data["experiment"]["experiment_id"]
    status, _ = req("POST", "/api/2.0/mlflow/experiments/delete", {"experiment_id": exp_id})
    if status == 200:
        print(f"    [OK]       Experiment deleted: {name}")
    else:
        print(f"    [WARN]     Could not delete experiment {name}: {status}", file=sys.stderr)

PYEOF

echo ""

# ── Step 3: bundle destroy ────────────────────────────────────────────────────
echo "==> Running databricks bundle destroy ${BUNDLE_ARGS[*]}"
_BUNDLE_EXIT=0
databricks bundle destroy --auto-approve "${BUNDLE_ARGS[@]}" || _BUNDLE_EXIT=$?
if [ "$_BUNDLE_EXIT" -ne 0 ] && [ "$_LAKEBASE_DELETED" -ne 0 ]; then
    # bundle destroy failed AND Lakebase was not pre-deleted (protected branch).
    # The failure is expected for the Lakebase endpoint. All other resources
    # (jobs, DLT pipeline, app) were still deleted. Warn and continue.
    echo ""
    echo "    [WARN] bundle destroy reported errors (likely the Lakebase read-write endpoint)."
    echo "    [WARN] To complete Lakebase cleanup, delete the project manually:"
    echo "    [WARN]   Workspace UI → Catalog → Lakebase → actuarial-workshop-lakebase → Delete"
    echo "    [WARN] Or run: databricks postgres delete-project projects/actuarial-workshop-lakebase --profile ${PROFILE}"
elif [ "$_BUNDLE_EXIT" -ne 0 ]; then
    # Lakebase was deleted but bundle destroy still failed — unexpected error.
    echo "ERROR: bundle destroy failed (exit code ${_BUNDLE_EXIT})." >&2
    exit "$_BUNDLE_EXIT"
fi

# ── Step 4: remove workspace bundle folder ────────────────────────────────────
# bundle destroy does not clean up the workspace file sync directories.
if [ -n "$ROOT_PATH" ]; then
    BUNDLE_DIR=$(dirname "${ROOT_PATH}")
    echo "==> Removing workspace bundle folder: ${BUNDLE_DIR}"
    databricks workspace delete --recursive "${BUNDLE_DIR}" "${PROFILE_ARGS[@]}" 2>/dev/null \
        && echo "    Removed." \
        || echo "    (folder not found or already removed)"
fi
