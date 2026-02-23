#!/usr/bin/env bash
# destroy.sh — wrapper around `databricks bundle destroy`
#
# Usage:  ./destroy.sh [--target <target>] [other bundle flags]
#
# In addition to destroying bundle-managed resources, this script removes
# the .bundle/<bundle-name> workspace folder that `bundle destroy` leaves
# behind after removing the target-specific subfolder.

set -euo pipefail

BUNDLE_ARGS=("$@")

# Resolve the bundle workspace path before destroying so we know what to clean up.
echo "==> Resolving bundle workspace path..."
VALIDATE_JSON=$(databricks bundle validate --output json "${BUNDLE_ARGS[@]}" 2>/dev/null)

BUNDLE_NAME=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('bundle',{}).get('name',''))")
ROOT_PATH=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('workspace',{}).get('root_path',''))")
PROFILE=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('workspace',{}).get('profile','') or '')")

echo "==> Running databricks bundle destroy ${BUNDLE_ARGS[*]}"
databricks bundle destroy "${BUNDLE_ARGS[@]}"

# Remove the leftover .bundle/<bundle-name> workspace folder.
# bundle destroy removes the target subfolder (e.g. .bundle/actuarial-workshop/fevm-serverless)
# but leaves the parent bundle-named directory.
if [ -n "$BUNDLE_NAME" ] && [ -n "$ROOT_PATH" ]; then
    BUNDLE_DIR="${ROOT_PATH}/.bundle/${BUNDLE_NAME}"
    echo "==> Removing workspace bundle folder: ${BUNDLE_DIR}"
    PROFILE_ARGS=()
    [ -n "$PROFILE" ] && PROFILE_ARGS=(--profile "$PROFILE")
    databricks workspace delete --recursive "${BUNDLE_DIR}" "${PROFILE_ARGS[@]}" 2>/dev/null \
        && echo "    Removed." \
        || echo "    (folder not found or already removed)"
fi
