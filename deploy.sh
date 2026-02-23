#!/usr/bin/env bash
# deploy.sh — wrapper around `databricks bundle deploy`
#
# Usage:  ./deploy.sh [--target <target>] [other bundle flags]
#
# Why this script exists:
#   Databricks Apps uploads app/app.yaml as-is (no bundle variable substitution).
#   This script reads the resolved variable values via `bundle validate`, writes
#   app/_bundle_config.py with the actual values, then runs `bundle deploy`.
#   The app imports _bundle_config.py at startup to get CATALOG / SCHEMA /
#   PG_DATABASE when they are not already set via environment variables.
#
# app/_bundle_config.py is gitignored and re-generated on every deploy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse flags to forward to bundle commands ──────────────────────────────
BUNDLE_ARGS=("$@")

echo "==> Resolving bundle variables..."
# `bundle validate --output json` returns the fully-resolved bundle config
VALIDATE_JSON=$(databricks bundle validate --output json "${BUNDLE_ARGS[@]}" 2>/dev/null)

CATALOG=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('variables',{}).get('catalog',{}).get('value','my_catalog'))")
SCHEMA=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('variables',{}).get('schema',{}).get('value','actuarial_workshop'))")
PG_DATABASE=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('variables',{}).get('pg_database',{}).get('value','actuarial_workshop_db'))")

echo "==> Generating app/_bundle_config.py"
echo "    CATALOG=${CATALOG}, SCHEMA=${SCHEMA}, PG_DATABASE=${PG_DATABASE}"
python3 "${SCRIPT_DIR}/scripts/gen_bundle_config.py" "$CATALOG" "$SCHEMA" "$PG_DATABASE"

echo "==> Running databricks bundle deploy ${BUNDLE_ARGS[*]}"
databricks bundle deploy "${BUNDLE_ARGS[@]}"
