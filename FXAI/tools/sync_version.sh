#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION_FILE="$ROOT_DIR/VERSION.txt"

if [[ ! -f "$VERSION_FILE" ]]; then
  echo "FXAI v1" > "$VERSION_FILE"
fi

version="$(tr -d '\r' < "$VERSION_FILE")"
if [[ -z "$version" ]]; then
  echo "FXAI v1" > "$VERSION_FILE"
  version="FXAI v1"
fi

if [[ "${1:-}" == "--bump" ]]; then
  n="$(printf '%s' "$version" | sed -E 's/^FXAI v([0-9]+)$/\1/')"
  if [[ -z "$n" || "$n" == "$version" ]]; then
    echo "Invalid version format in $VERSION_FILE: $version" >&2
    exit 1
  fi
  n="$((n + 1))"
  version="FXAI v${n}"
  printf '%s\n' "$version" > "$VERSION_FILE"
fi

while IFS= read -r -d '' file; do
  first_line="$(head -n 1 "$file" || true)"
  if [[ "$first_line" =~ ^//[[:space:]]FXAI[[:space:]]v[0-9]+$ ]]; then
    perl -0777 -i -pe "s#\A//\s*FXAI\s*v\d+#// ${version}#" "$file"
  else
    tmp="$(mktemp)"
    {
      printf '// %s\n' "$version"
      cat "$file"
    } > "$tmp"
    mv "$tmp" "$file"
  fi
done < <(find "$ROOT_DIR" -type f \( -name '*.mq5' -o -name '*.mqh' -o -name '*.mqh.bak_*' \) -print0)

echo "Synchronized version header to '${version}'"
