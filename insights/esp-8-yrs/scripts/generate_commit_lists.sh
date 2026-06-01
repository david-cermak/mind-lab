#!/usr/bin/env bash
# Generate commit lists and summary.md from git repos (author or committer email match).
#
# Usage:
#   generate_commit_lists.sh              # default ESP repos and emails
#   generate_commit_lists.sh --help       # all options (via Python script)
#
# Examples:
#   ./scripts/generate_commit_lists.sh
#
#   ./scripts/generate_commit_lists.sh \
#     --repo ext/esp-idf:esp-idf \
#     --repo ext/my-fork:my-fork \
#     --ref my-fork=main \
#     --email me@example.com \
#     --output-dir ./out
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -eq 0 ]]; then
  exec python3 "${SCRIPT_DIR}/generate_commit_lists.py" \
    --repo "${ROOT}/ext/esp-idf:esp-idf" \
    --repo "${ROOT}/ext/esp-lwip:esp-lwip" \
    --repo "${ROOT}/ext/esp-protocols:esp-protocols" \
    --email cermak@espressif.com \
    --email david.cermak@espressif.com \
    --output-dir "${ROOT}"
fi

exec python3 "${SCRIPT_DIR}/generate_commit_lists.py" "$@"
