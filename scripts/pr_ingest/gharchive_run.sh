#!/usr/bin/env bash
# Tier-2 PR ingest, component (2): bq runner for gharchive_query.sql.
#
# Drives the GH Archive extraction with the user's OWN GCP project + creds.
# It ALWAYS dry-runs first (prints bytes-scanned + a cost estimate), then runs
# the real query into a destination table, then exports that table to local
# newline-delimited JSON for component (3) pr_store.py to ingest.
#
# Creds the user must provide (RULE #1: no silent fallback — fails if missing):
#   * A GCP project with billing + BigQuery enabled (BQ_PROJECT).
#   * `gcloud auth login` (or a service account) so `bq` is authenticated.
#   * A GCS bucket (BQ_GCS_BUCKET) for the export step (BigQuery exports go to
#     GCS, then we `gsutil cp` to local).
#
# Usage:
#   BQ_PROJECT=my-gcp-proj \
#   BQ_DATASET=pr_ingest \
#   BQ_GCS_BUCKET=gs://my-bucket/pr_ingest \
#   REPO_LIST=/path/to/repo_list.json \
#   SUFFIX_START=1501 SUFFIX_END=2606 \
#   OUT_DIR=/mnt/nvme/cppmega_data/pr_ingest \
#   ./scripts/pr_ingest/gharchive_run.sh
#
# Set DRY_RUN_ONLY=1 to stop after the dry run (recommended first pass).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL="${here}/gharchive_query.sql"

: "${BQ_PROJECT:?set BQ_PROJECT to your GCP project id}"
: "${BQ_DATASET:?set BQ_DATASET (e.g. pr_ingest) — it must exist or be creatable}"
: "${BQ_GCS_BUCKET:?set BQ_GCS_BUCKET (gs://bucket/prefix) for the export step}"
: "${REPO_LIST:?set REPO_LIST to the repo_list.json from build_repo_list.py}"
: "${SUFFIX_START:=1501}"   # GH Archive monthly shards begin 2015-01
: "${SUFFIX_END:=$(date +%y%m)}"  # current month as YYMM
: "${OUT_DIR:=./pr_ingest_out}"
: "${DEST_TABLE:=pr_discussion_raw}"

# --- Build the repo_names ARRAY parameter from repo_list.json --------------
if ! command -v jq >/dev/null 2>&1; then
  echo "[gharchive_run] FAIL: jq is required to read ${REPO_LIST}" >&2
  exit 1
fi
mapfile -t REPO_NAMES < <(jq -r '.repo_names[]' "${REPO_LIST}")
if [ "${#REPO_NAMES[@]}" -eq 0 ]; then
  echo "[gharchive_run] FAIL: repo_list has 0 repo_names (${REPO_LIST})" >&2
  exit 1
fi
# bq array param syntax: name:ARRAY<STRING>:["a","b"]
REPO_JSON="$(printf '%s\n' "${REPO_NAMES[@]}" | jq -R . | jq -cs .)"
REPO_PARAM="repo_names:ARRAY<STRING>:${REPO_JSON}"
echo "[gharchive_run] ${#REPO_NAMES[@]} repos; shards ${SUFFIX_START}..${SUFFIX_END}" >&2

# --- 1) DRY RUN (cost estimate; scans nothing, bills nothing) ---------------
echo "[gharchive_run] === DRY RUN (no bytes billed) ===" >&2
DRY_JSON="$(
  bq --project_id="${BQ_PROJECT}" query \
    --use_legacy_sql=false --dry_run --format=prettyjson \
    --parameter="suffix_start:STRING:${SUFFIX_START}" \
    --parameter="suffix_end:STRING:${SUFFIX_END}" \
    --parameter="${REPO_PARAM}" \
    "$(cat "${SQL}")"
)"
echo "${DRY_JSON}"
BYTES="$(printf '%s' "${DRY_JSON}" | jq -r '.statistics.query.totalBytesProcessed // .statistics.totalBytesProcessed // empty')"
if [ -n "${BYTES}" ]; then
  TB=$(python3 -c "print(f'{${BYTES}/1e12:.4f}')")
  USD=$(python3 -c "print(f'{${BYTES}/1e12*6.25:.2f}')")  # on-demand ~\$6.25/TB
  echo "[gharchive_run] dry-run: ${BYTES} bytes (~${TB} TB) -> ~\$${USD} on-demand" >&2
fi

if [ "${DRY_RUN_ONLY:-0}" = "1" ]; then
  echo "[gharchive_run] DRY_RUN_ONLY=1 set; stopping before real query." >&2
  exit 0
fi

# --- 2) REAL QUERY -> destination table -------------------------------------
echo "[gharchive_run] === REAL QUERY -> ${BQ_PROJECT}:${BQ_DATASET}.${DEST_TABLE} ===" >&2
bq --project_id="${BQ_PROJECT}" mk -f --dataset "${BQ_PROJECT}:${BQ_DATASET}" >/dev/null 2>&1 || true
bq --project_id="${BQ_PROJECT}" query \
  --use_legacy_sql=false \
  --destination_table="${BQ_PROJECT}:${BQ_DATASET}.${DEST_TABLE}" \
  --replace \
  --parameter="suffix_start:STRING:${SUFFIX_START}" \
  --parameter="suffix_end:STRING:${SUFFIX_END}" \
  --parameter="${REPO_PARAM}" \
  "$(cat "${SQL}")"

# --- 3) EXPORT table -> GCS (NDJSON) -> local -------------------------------
GCS_GLOB="${BQ_GCS_BUCKET%/}/${DEST_TABLE}-*.json"
echo "[gharchive_run] === EXPORT -> ${GCS_GLOB} ===" >&2
bq --project_id="${BQ_PROJECT}" extract \
  --destination_format=NEWLINE_DELIMITED_JSON \
  "${BQ_PROJECT}:${BQ_DATASET}.${DEST_TABLE}" \
  "${GCS_GLOB}"

mkdir -p "${OUT_DIR}"
gsutil -m cp "${GCS_GLOB}" "${OUT_DIR}/"
echo "[gharchive_run] downloaded NDJSON shards -> ${OUT_DIR}/" >&2
echo "[gharchive_run] next: python3 scripts/pr_ingest/pr_store.py ingest-gharchive \\" >&2
echo "                  --store ${OUT_DIR}/pr_store.sqlite --input '${OUT_DIR}/${DEST_TABLE}-*.json'" >&2
