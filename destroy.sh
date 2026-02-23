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

# ── 2c. Delete Model Serving endpoint
api_delete "Serving endpoint ${ENDPOINT_NAME}" \
    "/api/2.0/serving-endpoints/${ENDPOINT_NAME}"

# ── 2d. Delete UC registered model (and all versions)
MODEL_NAME="${CATALOG}.${SCHEMA}.sarima_claims_forecaster"
api_delete "UC model ${MODEL_NAME}" \
    "/api/2.1/unity-catalog/models/${MODEL_NAME}"

# ── 2e. Delete MLflow experiments
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
databricks bundle destroy "${BUNDLE_ARGS[@]}"

# ── Step 4: remove workspace bundle folder ────────────────────────────────────
# bundle destroy does not clean up the workspace file sync directories.
if [ -n "$ROOT_PATH" ]; then
    BUNDLE_DIR=$(dirname "${ROOT_PATH}")
    echo "==> Removing workspace bundle folder: ${BUNDLE_DIR}"
    databricks workspace delete --recursive "${BUNDLE_DIR}" "${PROFILE_ARGS[@]}" 2>/dev/null \
        && echo "    Removed." \
        || echo "    (folder not found or already removed)"
fi
