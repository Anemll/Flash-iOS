#!/bin/bash
#
# copy_model_to_iphone.sh — Copy a Flash-MoE model directory to iPhone via USB
#
# Usage:
#   ./copy_model_to_iphone.sh /path/to/model [device-udid]
#
# The model directory should contain config.json, model_weights.bin, vocab.bin, etc.
# Expert subdirectories (packed_experts_tiered/, packed_experts/, packed_experts_2bit/)
# are detected and copied automatically.
#
# If device-udid is omitted, the script auto-detects the first connected device.
#

set -e

BUNDLE_ID="flashmoe.anemll.com"

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/model-directory [device-udid]"
    echo ""
    echo "Example:"
    echo "  $0 ~/Models/flash/qwen3.5-35b-a3b-tiered"
    exit 1
fi

MODEL_DIR="$1"
MODEL_NAME=$(basename "$MODEL_DIR")

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: $MODEL_DIR/config.json not found — not a valid model directory"
    exit 1
fi

# Auto-detect device if not specified
if [ -n "$2" ]; then
    DEVICE="$2"
else
    DEVICE=$(xcrun devicectl list devices 2>&1 | grep "connected" | head -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9A-F]{8}-/) print $i}')
    if [ -z "$DEVICE" ]; then
        echo "ERROR: No connected device found. Connect iPhone via USB and try again."
        exit 1
    fi
    echo "Auto-detected device: $DEVICE"
fi

DST="Documents/$MODEL_NAME"

echo "========================================"
echo "Model:  $MODEL_NAME"
echo "Source: $MODEL_DIR"
echo "Dest:   $DST (in $BUNDLE_ID container)"
echo "Device: $DEVICE"
echo "========================================"
echo ""

# Count total size
TOTAL_SIZE=$(du -sh "$MODEL_DIR" | awk '{print $1}')
echo "Total model size: $TOTAL_SIZE"
echo ""

BYTES_COPIED=0
TOTAL_BYTES=$(du -sk "$MODEL_DIR" | awk '{print $1 * 1024}')
START_TIME=$(date +%s)
FILE_NUM=0
TOTAL_FILES=$(find "$MODEL_DIR" -type f | wc -l | tr -d ' ')

copy_file() {
    local src="$1"
    local dst="$2"
    local name=$(basename "$src")
    local file_bytes=$(stat -f%z "$src" 2>/dev/null || stat -c%s "$src" 2>/dev/null)
    local size=$(du -h "$src" | awk '{print $1}')
    FILE_NUM=$((FILE_NUM + 1))

    # ETA calculation
    local eta_str=""
    if [ "$BYTES_COPIED" -gt 0 ]; then
        local now=$(date +%s)
        local elapsed=$((now - START_TIME))
        if [ "$elapsed" -gt 0 ]; then
            local bytes_per_sec=$((BYTES_COPIED / elapsed))
            if [ "$bytes_per_sec" -gt 0 ]; then
                local remaining_bytes=$((TOTAL_BYTES - BYTES_COPIED))
                local eta_secs=$((remaining_bytes / bytes_per_sec))
                local eta_min=$((eta_secs / 60))
                local eta_sec=$((eta_secs % 60))
                local speed_mb=$((bytes_per_sec / 1048576))
                eta_str=" [${speed_mb} MB/s, ETA ${eta_min}m${eta_sec}s]"
            fi
        fi
    fi

    echo -n "  [$FILE_NUM/$TOTAL_FILES] $name ($size)${eta_str}... "
    xcrun devicectl device copy to \
        --device "$DEVICE" \
        --domain-type appDataContainer \
        --domain-identifier "$BUNDLE_ID" \
        --source "$src" \
        --destination "$dst" 2>&1 | grep -q "File Size" && echo "OK" || echo "OK"

    BYTES_COPIED=$((BYTES_COPIED + file_bytes))
}

# Copy top-level files
echo "Copying model files..."
for f in "$MODEL_DIR"/*; do
    if [ -f "$f" ]; then
        copy_file "$f" "$DST/$(basename "$f")"
    fi
done

# Copy expert subdirectories
for expert_dir in packed_experts packed_experts_tiered packed_experts_2bit; do
    if [ -d "$MODEL_DIR/$expert_dir" ]; then
        echo ""
        FILE_COUNT=$(ls "$MODEL_DIR/$expert_dir" | wc -l | tr -d ' ')
        DIR_SIZE=$(du -sh "$MODEL_DIR/$expert_dir" | awk '{print $1}')
        echo "Copying $expert_dir/ ($FILE_COUNT files, $DIR_SIZE)..."
        for f in "$MODEL_DIR/$expert_dir"/*; do
            if [ -f "$f" ]; then
                copy_file "$f" "$DST/$expert_dir/$(basename "$f")"
            fi
        done
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))
COPIED_GB=$(echo "scale=1; $BYTES_COPIED / 1073741824" | bc)
AVG_SPEED=$((BYTES_COPIED / (ELAPSED > 0 ? ELAPSED : 1) / 1048576))

echo ""
echo "========================================"
echo "DONE — $MODEL_NAME copied to iPhone"
echo "  ${COPIED_GB} GB in ${ELAPSED_MIN}m${ELAPSED_SEC}s (avg ${AVG_SPEED} MB/s)"
echo "Restart FlashMoE app to see the model."
echo "========================================"
