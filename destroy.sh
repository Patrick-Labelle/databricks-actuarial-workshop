#!/usr/bin/env bash
# destroy.sh — full teardown: workspace assets + bundle destroy
#
# Usage:  ./destroy.sh [--target <target>] [other bundle flags]
#
# Sequence:
#   1. Resolve bundle variables via bundle validate
#   2. Remove workspace assets not managed by the bundle:
#      UC schema (CASCADE), Online Table, serving endpoints, UC models,
#      Genie space, Lakebase project, MLflow experiments
#   3. databricks bundle destroy
#   4. Remove workspace bundle folder

set -euo pipefail

# ── CLI version check ───────────────────────────────────────────────────────
_CLI_VER=$(databricks --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "0.0.0")
if [ "$(echo "$_CLI_VER" | cut -d. -f2)" -lt 287 ] && [ "$(echo "$_CLI_VER" | cut -d. -f1)" -lt 1 ]; then
    echo "ERROR: Databricks CLI >= 0.287.0 required (found ${_CLI_VER})." >&2
    exit 1
fi

BUNDLE_ARGS=("$@")

# Extract only validate-compatible flags
VALIDATE_ARGS=()
i=0
while [ $i -lt ${#BUNDLE_ARGS[@]} ]; do
    arg="${BUNDLE_ARGS[$i]}"
    case "$arg" in
        --target|-t|--profile|-p|--var)
            VALIDATE_ARGS+=("$arg" "${BUNDLE_ARGS[$((i+1))]}")
            i=$((i+2)) ;;
        --target=*|--profile=*|--var=*)
            VALIDATE_ARGS+=("$arg")
            i=$((i+1)) ;;
        *) i=$((i+1)) ;;
    esac
done

# ── Step 1: resolve variables ───────────────────────────────────────────────
echo "==> Resolving bundle variables..."
VALIDATE_JSON=$(databricks bundle validate --output json "${VALIDATE_ARGS[@]}" 2>/dev/null)

_var() { echo "$VALIDATE_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('variables',{}).get('$1',{}).get('value',''))"; }
_ws()  { echo "$VALIDATE_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('workspace',{}).get('$1',''))"; }

ROOT_PATH=$(_ws root_path)
PROFILE=$(_ws profile)
WORKSPACE_HOST=$(echo "$VALIDATE_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('workspace',{}).get('host','').rstrip('/'))")
CURRENT_USER=$(echo "$VALIDATE_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('workspace',{}).get('current_user',{}).get('userName',''))")
CATALOG=$(_var catalog)
SCHEMA=$(_var schema)
ENDPOINT_NAME=$(_var endpoint_name)
MC_ENDPOINT_NAME=$(_var mc_endpoint_name)
WAREHOUSE_ID=$(_var warehouse_id)
GENIE_SPACE_ID=$(_var genie_space_id)

PROFILE_ARGS=()
[ -n "$PROFILE" ] && PROFILE_ARGS=(--profile "$PROFILE")

echo "    Host: ${WORKSPACE_HOST}  Catalog: ${CATALOG}.${SCHEMA}"

TOKEN=$(databricks auth token --host "${WORKSPACE_HOST}" "${PROFILE_ARGS[@]}" 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null || echo "")
[ -z "$TOKEN" ] && { echo "ERROR: could not obtain auth token." >&2; exit 1; }

# ── Step 2: remove workspace assets ─────────────────────────────────────────
echo "==> Removing workspace assets..."

api_delete() {
    local label="$1" path="$2"
    local resp
    resp=$(curl -s -o /dev/null -w "%{http_code}" \
        -X DELETE "${WORKSPACE_HOST}${path}" \
        -H "Authorization: Bearer ${TOKEN}" 2>/dev/null)
    case "$resp" in
        200|204) echo "    [OK]   ${label}" ;;
        404)     echo "    [SKIP] ${label} — not found" ;;
        *)       echo "    [WARN] ${label} — HTTP ${resp}" ;;
    esac
}

# 2a. Drop UC schema (CASCADE)
if [ -n "$WAREHOUSE_ID" ] && [ -n "$CATALOG" ] && [ -n "$SCHEMA" ]; then
    echo "    Dropping ${CATALOG}.${SCHEMA} (CASCADE)..."
    python3 - <<PYEOF
import json, time, sys, urllib.request, urllib.error
host, token, wh_id = "${WORKSPACE_HOST}", "${TOKEN}", "${WAREHOUSE_ID}"
catalog, schema = "${CATALOG}", "${SCHEMA}"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
def req(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(f"{host}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp: return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b'{}')
status, data = req("POST", "/api/2.0/sql/statements", {
    "warehouse_id": wh_id,
    "statement": f"DROP SCHEMA IF EXISTS \`{catalog}\`.\`{schema}\` CASCADE",
    "wait_timeout": "50s", "on_wait_timeout": "CONTINUE",
})
state = data.get("status", {}).get("state", "UNKNOWN")
stmt_id = data.get("statement_id", "")
for _ in range(30):
    if state in ("SUCCEEDED", "FAILED", "CANCELED", "CLOSED"): break
    time.sleep(10)
    _, data = req("GET", f"/api/2.0/sql/statements/{stmt_id}")
    state = data.get("status", {}).get("state", "UNKNOWN")
if state == "SUCCEEDED": print(f"    [OK]   Schema {catalog}.{schema} dropped")
else: print(f"    [WARN] Schema drop: {state}", file=sys.stderr)
PYEOF
fi

# 2b. Online Table
api_delete "Online Table" "/api/2.0/online-tables/${CATALOG}.${SCHEMA}.segment_features_online"

# 2c. Serving endpoints
api_delete "Endpoint ${ENDPOINT_NAME}" "/api/2.0/serving-endpoints/${ENDPOINT_NAME}"
api_delete "Endpoint ${MC_ENDPOINT_NAME}" "/api/2.0/serving-endpoints/${MC_ENDPOINT_NAME}"

# 2d. UC models
api_delete "Model frequency_forecaster" "/api/2.1/unity-catalog/models/${CATALOG}.${SCHEMA}.frequency_forecaster"
api_delete "Model bootstrap_reserve_simulator" "/api/2.1/unity-catalog/models/${CATALOG}.${SCHEMA}.bootstrap_reserve_simulator"

# 2e. Genie space
if [ -n "$GENIE_SPACE_ID" ]; then
    api_delete "Genie space ${GENIE_SPACE_ID}" "/api/2.0/genie/spaces/${GENIE_SPACE_ID}"
else
    echo "    [SKIP] Genie space — no genie_space_id configured"
fi

# 2g. Lakebase project
echo "    Deleting Lakebase project..."
_LAKEBASE_DELETED=0
python3 - <<PYEOF || _LAKEBASE_DELETED=$?
import urllib.request, urllib.error, json, time, sys
host, token = "${WORKSPACE_HOST}", "${TOKEN}"
headers = {"Authorization": f"Bearer {token}"}
def req(method, path):
    r = urllib.request.Request(f"{host}{path}", headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp: return resp.status, json.loads(resp.read() or b'{}')
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b'{}')
for attempt in range(8):
    status, data = req("DELETE", "/api/2.0/postgres/projects/actuarial-workshop-lakebase")
    if status in (200, 202, 204): print("    [OK]   Lakebase project deleted"); break
    elif status == 404: print("    [SKIP] Lakebase project not found"); break
    elif status == 409:
        print(f"    Lakebase reconciling, retrying in {20*(attempt+1)}s...")
        time.sleep(20 * (attempt + 1))
    else: print(f"    [WARN] Lakebase delete: HTTP {status}"); sys.exit(1); break
PYEOF

# 2h. MLflow experiments
echo "    Deleting MLflow experiments..."
python3 - <<PYEOF
import json, urllib.request, urllib.error, urllib.parse
host, token, user = "${WORKSPACE_HOST}", "${TOKEN}", "${CURRENT_USER}"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
def req(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(f"{host}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp: return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b'{}')
for name in [f"/Users/{user}/actuarial_workshop_frequency_forecaster",
             f"/Users/{user}/actuarial_workshop_bootstrap_reserve_simulator",
             "/Shared/actuarial-workshop-app-traces"]:
    status, data = req("GET", f"/api/2.0/mlflow/experiments/get-by-name?experiment_name={urllib.parse.quote(name)}")
    if status == 404 or "experiment" not in data:
        print(f"    [SKIP] {name.split('/')[-1]}"); continue
    exp_id = data["experiment"]["experiment_id"]
    s, _ = req("POST", "/api/2.0/mlflow/experiments/delete", {"experiment_id": exp_id})
    print(f"    [{'OK' if s==200 else 'WARN'}]   {name.split('/')[-1]}")
PYEOF

# ── Step 3: bundle destroy ──────────────────────────────────────────────────
echo ""
echo "==> Running databricks bundle destroy..."
_BUNDLE_EXIT=0
databricks bundle destroy --auto-approve "${BUNDLE_ARGS[@]}" || _BUNDLE_EXIT=$?
if [ "$_BUNDLE_EXIT" -ne 0 ] && [ "$_LAKEBASE_DELETED" -ne 0 ]; then
    echo "    [WARN] bundle destroy failed (likely Lakebase read-write endpoint)."
    echo "    [WARN] Delete manually: Catalog → Lakebase → actuarial-workshop-lakebase → Delete"
elif [ "$_BUNDLE_EXIT" -ne 0 ]; then
    echo "ERROR: bundle destroy failed (exit code ${_BUNDLE_EXIT})." >&2
    exit "$_BUNDLE_EXIT"
fi

# ── Step 4: remove workspace bundle folder ──────────────────────────────────
if [ -n "$ROOT_PATH" ]; then
    BUNDLE_DIR=$(dirname "${ROOT_PATH}")
    echo "==> Removing workspace folder: ${BUNDLE_DIR}"
    databricks workspace delete --recursive "${BUNDLE_DIR}" "${PROFILE_ARGS[@]}" 2>/dev/null \
        && echo "    Removed." || echo "    (not found)"
fi

echo "==> Destroy complete!"
