#!/usr/bin/env bash
# slop skill bootstrap (macOS/Linux). Idempotent — run once, or after dependency changes.
set -euo pipefail

# Locate the SlopMachine repo root (the dir with pyproject.toml) by walking up from this script.
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ -n "$dir" ] && [ ! -f "$dir/pyproject.toml" ]; do
  parent="$(dirname "$dir")"; [ "$parent" = "$dir" ] && break; dir="$parent"
done
if [ ! -f "$dir/pyproject.toml" ]; then
  echo "ERROR: could not locate the SlopMachine repo (no pyproject.toml above this skill)." >&2; exit 1
fi
cd "$dir"
echo "SlopMachine repo: $dir"

command -v uv >/dev/null 2>&1 || { echo "ERROR: Astral 'uv' not found. Install: https://docs.astral.sh/uv/" >&2; exit 1; }

echo "Syncing dependencies (core + 'dance' extra) ..."
uv sync --extra dance

echo
echo "Environment / GPU check:"
uv run slop info

echo
echo "Bootstrap complete. Weights download on first use, or pre-fetch:"
echo "  uv run slop models download image       # ~7GB"
echo "  uv run slop models download animation   # ~77GB (dance engine)"
echo "  uv run slop assets download pose         # small ONNX models for dance"
