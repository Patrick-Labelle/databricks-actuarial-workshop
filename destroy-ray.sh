#!/usr/bin/env bash
# destroy-ray.sh — full teardown for the Ray-enabled deployment (e2-demo-ray).
#
# Identical to ./destroy.sh but targets e2-demo-ray.
#
# Usage:  ./destroy-ray.sh [--auto-approve]

set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/destroy.sh" --target e2-demo-ray "$@"
