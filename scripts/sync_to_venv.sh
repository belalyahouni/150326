#!/usr/bin/env bash
# Sync vllm source files from the in-repo copy into the venv installation so
# that changes to vllm/vllm/... are picked up by `venv/bin/python` without a
# reinstall.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/vllm/vllm"
DST="$REPO_ROOT/venv/lib/python3.12/site-packages/vllm"

if [[ ! -d "$SRC" ]]; then
    echo "source not found: $SRC" >&2
    exit 1
fi
if [[ ! -d "$DST" ]]; then
    echo "venv target not found: $DST" >&2
    exit 1
fi

rsync -a --include='*/' --include='*.py' --include='*.pyi' --exclude='*' \
      "$SRC/" "$DST/"

echo "synced python sources: $SRC -> $DST"
