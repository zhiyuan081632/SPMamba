#!/usr/bin/env bash
set -euo pipefail

# ====================================================================
# LRS3-2mix data generation pipeline for SPMamba
#
# Uses the same mixture pair list as ClearerVoice-Studio for fair
# audio-only vs audio-visual comparison.
#
# Step 1: Create 2-speaker mixtures from ClearerVoice CSV
#         (optional --reverb for synthetic RIR reverberation)
# Step 2: Generate JSON manifests for SPMamba training/testing
#
# Usage:
#   ./create_lrs3.sh                              # train+val, dry
#   ./create_lrs3.sh --reverb                     # train+val, with reverb
#   ./create_lrs3.sh val --max_pairs 10           # quick test
#   ./create_lrs3.sh val --reverb --max_pairs 10  # quick test with reverb
# ====================================================================

SPMAMBA_ROOT=/mnt/d/project/prjANS/src/AVSE/SPMamba
LRS3_ROOT=/mnt/e/data/LRS3
CSV_INPUT="/mnt/d/project/prjANS/src/AVSE/ClearerVoice-Studio/train/target_speaker_extraction/data/LRS3/mixture_data_list_2mix.csv"
OUT_ROOT="$LRS3_ROOT/2mix"
JSON_OUT_ROOT="$SPMAMBA_ROOT/DataPreProcess"

# Parse splits and extra args
SPLITS=()
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    train|val|test) SPLITS+=("$arg") ;;
    --*) EXTRA_ARGS+=("$arg") ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

if [ ${#SPLITS[@]} -eq 0 ]; then
  SPLITS=(train val)
fi

# Check if --reverb is in extra args to adjust output dir
REVERB_TAG=""
for arg in "${EXTRA_ARGS[@]}"; do
  if [ "$arg" = "--reverb" ]; then
    REVERB_TAG="-reverb"
  fi
done
OUT_ROOT="${OUT_ROOT}${REVERB_TAG}"

echo "=== Step 1: Create LRS3-2mix mixtures from ClearerVoice CSV ==="
echo "  LRS3 root:   $LRS3_ROOT"
echo "  CSV:         $CSV_INPUT"
echo "  Output root: $OUT_ROOT"
echo "  Splits:      ${SPLITS[*]}"
echo "  Extra args:  ${EXTRA_ARGS[*]:-(none)}"
echo ""

python "$SPMAMBA_ROOT/DataPreProcess/LRS3/create_lrs3_mixture.py" \
  --lrs3_root "$LRS3_ROOT" \
  --csv_input "$CSV_INPUT" \
  --out_root "$OUT_ROOT" \
  --splits "${SPLITS[@]}" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "=== Step 2: Generate JSON manifests ==="
echo "  Source root: $OUT_ROOT"
echo "  JSON output: $JSON_OUT_ROOT/LRS3/"
echo ""

python "$SPMAMBA_ROOT/DataPreProcess/LRS3/create_lrs3_json.py" \
  --src_root "$OUT_ROOT" \
  --out_root "$JSON_OUT_ROOT" \
  --splits "${SPLITS[@]}"

echo ""
echo "=== Done. JSON manifests at: $JSON_OUT_ROOT/LRS3/ ==="
