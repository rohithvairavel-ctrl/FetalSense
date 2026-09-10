#!/usr/bin/env bash
# Download PhysioNet fetal ECG corpora into the configured downloads root.
# Requires: curl or wget; optional wfdb-python `wfdb` package for record IO.
set -euo pipefail

ROOT="${FETALSENSE_DATA:-/workspace/fetalsense/data/downloads}"
mkdir -p "$ROOT"

echo "Data root: $ROOT"
echo "Cite PhysioNet / individual database papers when publishing."
echo ""

download_zip() {
  local url="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -d "$dest" ]]; then
    echo "Already exists: $dest"
    return 0
  fi
  echo "Fetching $url → $dest"
  local tmp="${dest}.tmp.zip"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$tmp" "$url"
  else
    wget -O "$tmp" "$url"
  fi
  mkdir -p "$dest"
  unzip -q -o "$tmp" -d "$dest" || true
  rm -f "$tmp"
}

# Challenge 2013 Set A (public training)
# https://physionet.org/content/challenge-2013/
download_zip \
  "https://physionet.org/static/published-projects/challenge-2013/1.0.0/set-a.zip" \
  "$ROOT/challenge2013" || echo "WARN: challenge2013 download failed (network?)"

# ADFECGDB
download_zip \
  "https://physionet.org/static/published-projects/adfecgdb/1.0.0/adfecgdb.zip" \
  "$ROOT/adfecgdb" || echo "WARN: adfecgdb download failed"

# FECGSYNDB (large — optional)
# download_zip "https://physionet.org/static/published-projects/fecgsyndb/1.0.0/fecgsyndb.zip" "$ROOT/fecgsyndb"

echo ""
echo "Done. Point configs/default.yaml paths.* at $ROOT/<dataset>."
echo "PhysioNet citation: Goldberger et al., Circulation 2000; PhysioBank/PhysioToolkit/PhysioNet."
