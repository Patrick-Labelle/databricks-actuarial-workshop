#!/usr/bin/env bash
# destroy.sh — wrapper around `databricks bundle destroy`
#
# Usage:  ./destroy.sh [--target <target>] [other bundle flags]
#
# In addition to destroying bundle-managed resources, this script removes
# the entire .bundle/<bundle-name> workspace folder (all target subfolders)
# that `bundle destroy` leaves behind.

set -euo pipefail

BUNDLE_ARGS=("$@")

# Resolve the bundle workspace path before destroying so we know what to clean up.
# root_path is e.g. /Workspace/Users/you@co.com/.bundle/actuarial-workshop/<target>
# dirname strips the target, giving the bundle-named parent directory.
echo "==> Resolving bundle workspace path..."
VALIDATE_JSON=$(databricks bundle validate --output json "${BUNDLE_ARGS[@]}" 2>/dev/null)

ROOT_PATH=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('workspace',{}).get('root_path',''))")
PROFILE=$(echo "$VALIDATE_JSON" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('workspace',{}).get('profile','') or '')")

echo "==> Running databricks bundle destroy ${BUNDLE_ARGS[*]}"
databricks bundle destroy "${BUNDLE_ARGS[@]}"

# Remove the entire .bundle/<bundle-name> directory (all target subfolders).
# bundle destroy does not clean up the workspace file sync directories.
if [ -n "$ROOT_PATH" ]; then
    BUNDLE_DIR=$(dirname "${ROOT_PATH}")
    echo "==> Removing workspace bundle folder: ${BUNDLE_DIR}"
    PROFILE_ARGS=()
    [ -n "$PROFILE" ] && PROFILE_ARGS=(--profile "$PROFILE")
    databricks workspace delete --recursive "${BUNDLE_DIR}" "${PROFILE_ARGS[@]}" 2>/dev/null \
        && echo "    Removed." \
        || echo "    (folder not found or already removed)"
fi
