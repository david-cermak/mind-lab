#!/usr/bin/env bash
# Print commits line by line (~20s total) with colored sha and repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST="${1:-${SCRIPT_DIR}/../list_non_merge.txt}"
DURATION="${2:-20}"

GREEN=$'\033[32m'
RED=$'\033[31m'
RESET=$'\033[0m'

if [[ ! -f "$LIST" ]]; then
  echo "File not found: $LIST" >&2
  exit 1
fi

count=$(grep -c . "$LIST" || true)
if [[ "$count" -eq 0 ]]; then
  exit 0
fi

delay=$(awk -v d="$DURATION" -v n="$count" 'BEGIN { printf "%.6f", d / n }')
use_color=0
[[ -t 1 ]] && use_color=1

print_commit() {
  local line=$1
  local repo sha title rest

  repo=${line##* }
  case "$repo" in
    esp-idf|esp-lwip|esp-protocols) ;;
    *)
      printf '%s\n' "$line"
      return
      ;;
  esac

  rest=${line% "$repo"}
  rest=${rest% }
  sha=${rest##* }
  if [[ ! "$sha" =~ ^[0-9a-f]{8}$ ]]; then
    printf '%s\n' "$line"
    return
  fi

  title=${rest% "$sha"}
  title=${title% }

  if (( use_color )); then
    printf '%s %s%s%s %s%s%s\n' "$title" "$GREEN" "$sha" "$RESET" "$RED" "$repo" "$RESET"
  else
    printf '%s %s %s\n' "$title" "$sha" "$repo"
  fi
}

while IFS= read -r line || [[ -n "$line" ]]; do
  print_commit "$line"
  sleep "$delay"
done < "$LIST"
