#!/bin/bash
#
# prepare_397B_ios.sh — One-shot preparation of Qwen3.5-397B for iOS
#
# Runs all steps: profile experts → generate manifest → repack tiered → extract split weights
# Interactive: asks before each step, shows progress, validates results.
#
# Usage:
#   ./prepare_397B_ios.sh [--model PATH] [--output PATH]
#

set -e

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH="${1:-$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3}"
OUTPUT_PATH="${2:-/Volumes/BK/flash-moe-397B}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
METAL_DIR="$SCRIPT_DIR/metal_infer"
SPLIT_GB=3.5
COVERAGE=0.8
NUM_PROFILE_PROMPTS=3

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ============================================================================
# Helpers
# ============================================================================

step_count=0
total_steps=6

step() {
    step_count=$((step_count + 1))
    echo ""
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  Step $step_count/$total_steps: $1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

ask() {
    echo -e "${YELLOW}$1${NC}"
    echo -n "  [Y/n] > "
    read -r answer
    case "$answer" in
        [nN]*) return 1 ;;
        *) return 0 ;;
    esac
}

info() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

warn() {
    echo -e "${YELLOW}  ⚠ $1${NC}"
}

fail() {
    echo -e "${RED}  ✗ $1${NC}"
    exit 1
}

hr() {
    echo -e "${BLUE}  ──────────────────────────────────────${NC}"
}

# ============================================================================
# Pre-flight checks
# ============================================================================

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Flash-MoE 397B iOS Preparation                 ║${NC}"
echo -e "${BOLD}║  Tiered experts + Split weights for Metal        ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Model:   $MODEL_PATH"
echo "  Output:  $OUTPUT_PATH"
echo "  Split:   ${SPLIT_GB} GB chunks (Metal 4GB buffer limit)"
echo "  Tiered:  ${COVERAGE} coverage threshold (hot=4-bit, cold=2-bit)"
echo ""

# Check model exists
if [ ! -f "$MODEL_PATH/config.json" ]; then
    fail "Model not found at $MODEL_PATH"
fi

# Check output dir / external drive
if [ ! -d "$(dirname "$OUTPUT_PATH")" ]; then
    fail "Output parent directory not found. Is the external drive mounted?"
fi

# Check free space
if [ -d "$OUTPUT_PATH" ]; then
    FREE_GB=$(df -g "$OUTPUT_PATH" | tail -1 | awk '{print $4}')
else
    FREE_GB=$(df -g "$(dirname "$OUTPUT_PATH")" | tail -1 | awk '{print $4}')
fi
echo "  Free space: ${FREE_GB} GB"

if [ "$FREE_GB" -lt 250 ]; then
    warn "Less than 250 GB free. Tiered experts need ~140 GB + split weights ~6 GB."
    if ! ask "Continue anyway?"; then
        exit 0
    fi
fi

# Check infer binary
if [ ! -f "$METAL_DIR/infer" ]; then
    warn "infer binary not found. Building..."
    (cd "$METAL_DIR" && make infer) || fail "Build failed"
fi

mkdir -p "$OUTPUT_PATH"

echo ""
if ! ask "Ready to start? This will take a while (profiling + repacking ~200GB)."; then
    echo "Aborted."
    exit 0
fi

# ============================================================================
# Step 1: Build expert index
# ============================================================================

step "Build expert index"

if [ -f "$SCRIPT_DIR/expert_index.json" ]; then
    info "expert_index.json already exists"
    if ! ask "Rebuild it?"; then
        info "Skipping"
    else
        python3 "$SCRIPT_DIR/build_expert_index.py" --model "$MODEL_PATH" || fail "Index build failed"
    fi
else
    python3 "$SCRIPT_DIR/build_expert_index.py" --model "$MODEL_PATH" || fail "Index build failed"
fi
info "Expert index ready"

# ============================================================================
# Step 2: Repack 4-bit experts (needed for profiling)
# ============================================================================

step "Repack 4-bit experts"

PACKED_DIR="$OUTPUT_PATH/packed_experts"
LAYER_COUNT=$(python3 -c "import json; print(json.load(open('$MODEL_PATH/config.json')).get('text_config',{}).get('num_hidden_layers', 60))")
EXISTING_LAYERS=$(ls "$PACKED_DIR"/*.bin 2>/dev/null | wc -l | tr -d ' ')

if [ "$EXISTING_LAYERS" -eq "$LAYER_COUNT" ]; then
    info "$EXISTING_LAYERS/$LAYER_COUNT layer files already exist"
    if ! ask "Re-repack?"; then
        info "Skipping"
    else
        python3 "$SCRIPT_DIR/repack_experts.py" --index "$SCRIPT_DIR/expert_index.json" || fail "Repack failed"
    fi
else
    echo "  Found $EXISTING_LAYERS/$LAYER_COUNT layers. Repacking..."
    # Ensure packed_experts symlink/dir points to output
    SNAP_PACKED="$MODEL_PATH/packed_experts"
    if [ ! -L "$SNAP_PACKED" ] && [ ! -d "$SNAP_PACKED" ]; then
        mkdir -p "$PACKED_DIR"
        ln -sf "$PACKED_DIR" "$SNAP_PACKED"
        info "Symlinked packed_experts → $PACKED_DIR"
    fi
    python3 "$SCRIPT_DIR/repack_experts.py" --index "$SCRIPT_DIR/expert_index.json" || fail "Repack failed"
fi

info "4-bit experts ready ($LAYER_COUNT layers)"

# ============================================================================
# Step 3: Extract split weight files
# ============================================================================

step "Extract non-expert weights (split for Metal)"

if [ -f "$OUTPUT_PATH/model_weights.bin" ] && [ -f "$OUTPUT_PATH/model_weights_1.bin" ]; then
    SIZE0=$(du -h "$OUTPUT_PATH/model_weights.bin" | cut -f1)
    SIZE1=$(du -h "$OUTPUT_PATH/model_weights_1.bin" | cut -f1)
    info "Split weight files already exist: $SIZE0 + $SIZE1"
    if ! ask "Re-extract?"; then
        info "Skipping"
    else
        python3 "$METAL_DIR/extract_weights.py" --model "$MODEL_PATH" --output "$OUTPUT_PATH" --split "$SPLIT_GB" || fail "Extract failed"
    fi
elif [ -f "$OUTPUT_PATH/model_weights.bin" ]; then
    SIZE0=$(du -h "$OUTPUT_PATH/model_weights.bin" | cut -f1)
    warn "Single weight file exists ($SIZE0) but no split. Re-extracting with --split $SPLIT_GB"
    python3 "$METAL_DIR/extract_weights.py" --model "$MODEL_PATH" --output "$OUTPUT_PATH" --split "$SPLIT_GB" || fail "Extract failed"
else
    python3 "$METAL_DIR/extract_weights.py" --model "$MODEL_PATH" --output "$OUTPUT_PATH" --split "$SPLIT_GB" || fail "Extract failed"
fi

info "Split weight files ready"

# ============================================================================
# Step 4: Copy config + tokenizer files
# ============================================================================

step "Copy config and tokenizer files"

for f in config.json tokenizer.json; do
    if [ -f "$MODEL_PATH/$f" ]; then
        cp "$MODEL_PATH/$f" "$OUTPUT_PATH/"
        info "Copied $f"
    else
        warn "$f not found in model dir"
    fi
done

for f in vocab.bin tokenizer.bin; do
    if [ -f "$OUTPUT_PATH/$f" ]; then
        info "$f already exists"
    elif [ -f "$METAL_DIR/$f" ]; then
        cp "$METAL_DIR/$f" "$OUTPUT_PATH/"
        info "Copied $f from metal_infer/"
    else
        warn "$f not found"
    fi
done

# ============================================================================
# Step 5: Profile expert usage
# ============================================================================

step "Profile expert usage for tiered quantization"

FREQ_DIR=$(mktemp -d)

echo "  Running $NUM_PROFILE_PROMPTS profiling prompts (200 tokens each)..."
echo "  This runs inference from the external drive — expect ~0.5 tok/s."
echo ""

PROMPTS=(
    "Explain the theory of general relativity and its implications for modern physics"
    "Write a Python function that implements a binary search tree with insertion and deletion"
    "What are the major causes and consequences of climate change on global food production"
)

for i in $(seq 0 $((NUM_PROFILE_PROMPTS - 1))); do
    PROMPT="${PROMPTS[$i]}"
    echo -e "  ${BLUE}[$((i+1))/$NUM_PROFILE_PROMPTS]${NC} \"${PROMPT:0:60}...\""

    "$METAL_DIR/infer" \
        --model "$OUTPUT_PATH" \
        --weights "$OUTPUT_PATH/model_weights.bin" \
        --manifest "$OUTPUT_PATH/model_weights.json" \
        --vocab "$OUTPUT_PATH/vocab.bin" \
        --prompt "$PROMPT" \
        --tokens 200 \
        --freq \
        --k 4 \
        2>&1 | tee "$FREQ_DIR/freq_$i.txt" | grep -E "FREQ_DUMP|tok/s|experts" || true

    echo ""
done

info "Profiling complete"

# ============================================================================
# Step 6: Generate hot manifest + repack tiered experts
# ============================================================================

step "Generate hot expert manifest and repack tiered"

hr
echo "  Generating hot_experts.json (coverage=${COVERAGE})..."

FREQ_FILES=""
for i in $(seq 0 $((NUM_PROFILE_PROMPTS - 1))); do
    FREQ_FILES="$FREQ_FILES $FREQ_DIR/freq_$i.txt"
done

python3 "$SCRIPT_DIR/profile_experts.py" \
    --freq-output $FREQ_FILES \
    --coverage "$COVERAGE" \
    --output "$OUTPUT_PATH/hot_experts.json" || fail "Profile analysis failed"

if [ ! -f "$OUTPUT_PATH/hot_experts.json" ]; then
    fail "hot_experts.json not generated"
fi

HOT_COUNT=$(python3 -c "import json; d=json.load(open('$OUTPUT_PATH/hot_experts.json')); print(sum(len(v) for v in d.values()))")
TOTAL_EXPERTS=$((LAYER_COUNT * 512))
COLD_COUNT=$((TOTAL_EXPERTS - HOT_COUNT))

info "Hot experts: $HOT_COUNT / $TOTAL_EXPERTS (cold: $COLD_COUNT)"
info "Expected savings: ~${COLD_COUNT} experts at 2-bit = ~34% smaller"

hr

if ! ask "Proceed with tiered repacking? (This will take a while — writing ~140 GB)"; then
    echo "Skipped tiered repacking. 4-bit experts are still available."
    echo ""
    echo -e "${BOLD}${GREEN}Done (without tiered)!${NC}"
    exit 0
fi

# Copy hot_experts.json to model dir for repack_experts_tiered.py
cp "$OUTPUT_PATH/hot_experts.json" "$MODEL_PATH/hot_experts.json" 2>/dev/null || true

python3 "$SCRIPT_DIR/repack_experts_tiered.py" --model "$MODEL_PATH" --output "$OUTPUT_PATH" || fail "Tiered repack failed"

info "Tiered experts ready"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  Preparation Complete!                           ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

echo "  Output: $OUTPUT_PATH"
echo ""
echo "  Files:"
for f in config.json model_weights.bin model_weights_1.bin model_weights.json vocab.bin tokenizer.bin tokenizer.json; do
    if [ -f "$OUTPUT_PATH/$f" ]; then
        SIZE=$(du -h "$OUTPUT_PATH/$f" | cut -f1)
        echo -e "    ${GREEN}✓${NC} $f ($SIZE)"
    else
        echo -e "    ${YELLOW}○${NC} $f (not found)"
    fi
done

for DIR in packed_experts packed_experts_tiered; do
    if [ -d "$OUTPUT_PATH/$DIR" ]; then
        COUNT=$(ls "$OUTPUT_PATH/$DIR"/*.bin 2>/dev/null | wc -l | tr -d ' ')
        SIZE=$(du -sh "$OUTPUT_PATH/$DIR" | cut -f1)
        echo -e "    ${GREEN}✓${NC} $DIR/ ($COUNT files, $SIZE)"
    fi
done

TOTAL=$(du -sh "$OUTPUT_PATH" | cut -f1)
echo ""
echo "  Total: $TOTAL"
echo ""
echo "  To test:"
echo "    cd metal_infer && ./infer --model $OUTPUT_PATH --k 4 --prompt 'Hello' --tokens 20"
echo ""
echo "  To transfer to iPhone:"
echo "    Plug iPhone via USB, open Finder → FlashMoE → drag $OUTPUT_PATH into Documents"
echo ""

# Cleanup
rm -rf "$FREQ_DIR"
