#!/bin/bash
# 지정 폴더 안의 모든 bag 폴더를 SLAM 처리
#
# Usage:
#   bash scripts/batch_process_slam.sh --input data/raw --output data/processed
#   bash scripts/batch_process_slam.sh -i data/raw -o data/processed --skip-existing

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKIP_EXISTING=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i) INPUT_DIR="$2"; shift 2 ;;
        --output|-o) OUTPUT_DIR="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 --input <raw_dir> --output <processed_dir> [--skip-existing]"
    exit 1
fi

# 절대 경로 변환
INPUT_DIR="$(cd "$INPUT_DIR" 2>/dev/null && pwd)" || { echo "Error: $INPUT_DIR not found"; exit 1; }
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# metadata.yaml 있는 폴더 = bag 폴더
DEMOS=($(find "$INPUT_DIR" -maxdepth 2 -name "metadata.yaml" -exec dirname {} \; | sort -V))
TOTAL=${#DEMOS[@]}

echo "========================================"
echo " UMI Batch SLAM Processing"
echo "========================================"
echo " Input:  $INPUT_DIR"
echo " Output: $OUTPUT_DIR"
echo " Found:  $TOTAL bag(s)"
echo " Skip existing: $SKIP_EXISTING"
echo "========================================"

SUCCESS=0
FAILED=0
SKIPPED=0
FAILED_LIST=""

for idx in "${!DEMOS[@]}"; do
    BAG_PATH="${DEMOS[$idx]}"
    NAME="$(basename "$BAG_PATH")"
    OUT_PATH="$OUTPUT_DIR/$NAME"
    NUM=$((idx + 1))

    echo ""
    echo "[$NUM/$TOTAL] $NAME"

    if [ "$SKIP_EXISTING" = true ] && [ -f "$OUT_PATH/dataset.hdf5" ]; then
        echo "  SKIP (already exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    START_TIME=$(date +%s)

    if python3 "$SCRIPT_DIR/process_bag_with_slam.py" \
        --input "$BAG_PATH" \
        --output "$OUT_PATH" 2>&1 | tee "/tmp/slam_${NAME}.log"; then
        ELAPSED=$(( $(date +%s) - START_TIME ))
        echo "  OK (${ELAPSED}s)"
        SUCCESS=$((SUCCESS + 1))
    else
        ELAPSED=$(( $(date +%s) - START_TIME ))
        echo "  FAIL (${ELAPSED}s) -> /tmp/slam_${NAME}.log"
        FAILED=$((FAILED + 1))
        FAILED_LIST="$FAILED_LIST $NAME"
    fi
done

echo ""
echo "========================================"
echo " SLAM: $SUCCESS ok / $FAILED fail / $SKIPPED skip"
[ -n "$FAILED_LIST" ] && echo " Failed:$FAILED_LIST"
echo "========================================"

# Merge 단계
if [ $SUCCESS -gt 0 ]; then
    MERGED="$OUTPUT_DIR/merged.hdf5"
    echo ""
    echo "========================================"
    echo " Merging $SUCCESS episodes -> $MERGED"
    echo "========================================"
    python3 "$SCRIPT_DIR/merge_umi_hdf5.py" -d "$OUTPUT_DIR" -o "$MERGED"
    echo "========================================"
    echo " All done! Output: $MERGED"
    echo "========================================"
fi
