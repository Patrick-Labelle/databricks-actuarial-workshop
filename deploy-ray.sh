#!/usr/bin/env bash
# deploy-ray.sh — deploy the Ray-enabled variant of the workshop to e2-demo.
#
# Identical to ./deploy.sh but targets e2-demo-ray, which overrides Task 5
# (fit_statistical_models) to run on a classic DBR 17.4 ML job cluster so
# that Ray-on-Spark can execute.  All other tasks remain on serverless.
#
# Usage:  ./deploy-ray.sh
#
# Timing note: classic cluster spin-up adds ~5-10 min to the setup job.
# Total expected time: ~25-30 min.

set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/deploy.sh" --target e2-demo-ray "$@"
