# Flash-MoE: Running Massive MoE Models on a Laptop

> **[Read the paper](paper/flash_moe.pdf)** — Full technical details, 90+ experiments, and the story of how an AI and a human built this in 24 hours.

Pure C/Metal inference engine for **Qwen3.5 Mixture-of-Experts** models on Apple Silicon. Runs models from 35B to 397B parameters on machines with as little as 24GB RAM, streaming expert weights from SSD through a custom Metal compute pipeline.

No Python runtime. No frameworks. Just C, Objective-C, and hand-tuned Metal shaders. Model architecture is auto-detected from HuggingFace `config.json` — switch models with a single `--model` flag.

## Compatible Models

Any **Qwen3.5 MoE** model with MLX quantization (`model_type: qwen3_5_moe`) is supported. Use the model manager to discover and download compatible models:

| Model | Params | Active | Quant | Disk | Min RAM |
|-------|--------|--------|-------|------|---------|
| Qwen3.5-35B-A3B | 35B | 3B | 4-bit | ~18GB | 24GB |
| Qwen3.5-35B-A3B | 35B | 3B | 8-bit | ~35GB | 48GB |
| Qwen3.5-122B-A10B | 122B | 10B | 4-bit | ~65GB | 48GB |
| Qwen3.5-397B-A17B | 397B | 17B | 4-bit | ~209GB | 48GB |
| Qwen3.5-397B-A17B | 397B | 17B | 6-bit | ~280GB | 64GB |
| Qwen3.5-397B-A17B | 397B | 17B | 8-bit | ~397GB | 96GB |

The engine auto-detects architecture, dimensions, expert counts, quantization, and layer types from `config.json`. No recompilation needed.

## Results

![Progress](progress.png)

### MacBook Pro M3 Max (48GB)

| Configuration | tok/s | Quality | Notes |
|--------------|-------|---------|-------|
| 4-bit + autoresearch optimizations | **9.7** | Excellent | +34.7% from half-precision x_shared + SIMD + FMA |
| 4-bit experts, FMA kernel | **4.36** | Excellent | Before autoresearch. Full tool calling. 209GB on disk. |
| **Tiered (hot=4bit, cold=2bit)** | **4.36+** | **Excellent** | **33% smaller on disk. Auto-detected.** |
| 2-bit experts, trust OS | 5.74 | Good* | 120GB on disk. *Breaks JSON/tool calling. |

### iPhone 17 (12GB, A19)

| Model | K | tok/s | Notes |
|-------|---|-------|-------|
| Qwen3.5-35B-A3B (4-bit) | 8 | **5.5** | 19.5GB download. Full quality. Full GPU path. |
| Qwen3.5-35B-A3B (tiered) | 8 | **5.5+** | 13.4GB download. Same quality. |
| Qwen3.5-397B-A17B (4-bit) | 4 | ~0.003* | *CPU fallback only — Metal 4GB per-buffer limit blocks GPU path. |
| Qwen3.5-397B-A17B (4-bit) | 4 | ~1-2** | **Projected with split weight files enabling GPU path. |

*2-bit quantization produces `\name\` instead of `"name"` in JSON output, making tool calling unreliable. 4-bit is the production configuration.

**Tiered mode** keeps frequently-activated experts (top ~25%) at 4-bit quality while requantizing cold experts to 2-bit — reducing disk footprint by ~34% without quality loss. Hot experts are profiled from real workloads. See [docs/tiered-expert-quantization.md](docs/tiered-expert-quantization.md) for the full experiment writeup.

## Hardware

### Development (MacBook Pro)
- **Machine**: MacBook Pro, Apple M3 Max
- **Chip**: 16-core CPU (12P + 4E), 40-core GPU, 16-core ANE
- **Memory**: 48 GB unified (~400 GB/s bandwidth)
- **SSD**: 1TB Apple Fabric, **17.5 GB/s sequential read** (measured)
- **macOS**: 26.2 (Darwin 25.2.0)

### Mobile (iPhone 17)
- **Chip**: A19, ~10-core GPU
- **Memory**: 12 GB unified
- **SSD**: ~2.5-3 GB/s NVMe
- **iOS**: 18+

## Architecture

Qwen3.5 MoE models use a hybrid attention architecture with GatedDeltaNet (linear attention) and standard full attention layers, each containing a Mixture-of-Experts MLP. Model dimensions, expert counts, and layer types vary per model and are read from `config.json` at startup. For example, the 397B model has 60 layers (45 linear + 15 full), 512 experts (K=4 active), hidden dim 4096; the 35B model has 40 layers (30 linear + 10 full), 256 experts (K=8 active), hidden dim 2048.

### Key Techniques

1. **SSD Expert Streaming** — Expert weights (209GB at 4-bit) are read from NVMe SSD on demand via parallel `pread()` with GCD dispatch groups. Only the K=4 active experts per layer are loaded (~6.75MB each). The OS page cache manages caching — no custom cache needed ("Trust the OS" principle). Inspired by Apple's "LLM in a Flash" paper.

1. **Tiered Expert Quantization** — Expert usage follows a Zipfian distribution: ~25% of experts handle ~80% of activations. Hot experts stay at 4-bit; cold experts are requantized to 2-bit (44% smaller each). This shrinks total expert disk by ~34%, improving OS page cache hit rates without quality degradation. Per-expert Metal kernel dispatch selects the right dequant shader at runtime.

2. **FMA-Optimized Dequant Kernel** — The inner loop of the 4-bit dequantized matrix-vector multiply rearranges the math from `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)`. Pre-computing `scale*x` and `bias*x` lets the GPU fused multiply-add unit do dequant+multiply in one instruction. 12% faster than the naive formulation.

3. **Metal Compute Shaders** — Hand-written Metal kernels for:
   - 4-bit and 2-bit dequantized matrix-vector multiply (tiled, SIMD-reduced, shared input cache, FMA-optimized)
   - Fused SwiGLU activation
   - RMS normalization (two-pass: sum-of-squares reduction + apply)
   - Batched GPU attention (Q@K^T, softmax, scores@V) for full attention layers
   - GPU RoPE (fused with Q deinterleave and K normalization)
   - MoE combine + residual + sigmoid gate (fused kernel)

4. **Deferred GPU Expert Compute** — CMD3 (expert forward pass) is submitted without waiting. The GPU executes it while the CPU prepares the next layer. The combine + residual + norm are also on GPU, feeding directly into the next layer's attention projections.

5. **Accelerate BLAS for Linear Attention** — The GatedDeltaNet recurrence uses `cblas_sscal`, `cblas_sgemv`, and `cblas_sger` for the 64-head × 128×128 state matrix update. 64% faster than scalar code.

6. **Trust the OS** — No custom expert cache. The OS page cache (~35GB) manages expert data caching via standard LRU. Every custom caching approach we tested (Metal LRU, malloc cache, LZ4 compressed cache) was slower due to GPU memory pressure or overhead. The page cache achieves ~71% hit rate naturally.

7. **K-Reduction for Mobile** — MoE models have a natural inference knob: activate fewer experts per token than trained. K=4 instead of K=10 on the 397B model cuts I/O by 60% per token with graceful quality degradation. The router still picks the *best* K experts from all 512 — it's not random subsampling. This makes the 397B viable on iPhone (12GB RAM, 2.5 GB/s NVMe) at ~1-2 tok/s.

8. **Half-Precision Shared Memory** — Storing the threadgroup input cache as `half` instead of `float` in Metal dequant kernels halves shared memory usage (16KB → 8KB), doubling GPU occupancy. Since inputs are already approximate from 4-bit dequantization, the precision loss is negligible. +12% tok/s on the main kernel.

9. **iOS Unity Build** — The entire 7,500-line inference engine compiles into the iOS app via `#include "infer.m"`. No fork, no separate codebase. A thin C API (`FlashMoEEngine.h`) wraps the static globals, and a Swift `@Observable` bridge provides `AsyncStream<Token>` generation with automatic memory-adaptive context sizing.

### iOS-Specific Constraints

- **Metal 4GB per-buffer limit** — iOS Metal buffers cannot exceed 4096 MB, regardless of entitlements. The 35B model (2.5GB weights) fits in a single buffer. The 397B model (5.5GB weights) does not. Attempted workarounds: two overlapping buffers (OOM), staging buffer with memcpy per dispatch (data corruption from in-flight command buffer aliasing), CPU fallback (works, 6 min/token). Solution: split `model_weights.bin` into two <4GB files at packing time.
- **K-reduction quality varies by model** — K=2 and K=4 on the 397B produce gibberish/degenerate output. The model was trained with K=10 and needs K=6+ for coherence (untested — needs GPU path). The 35B at default K=8 works perfectly.
- **Debug build overhead** — Metal API Validation adds ~2GB of `MTLDebugComputeCommandEncoder` proxies, causing OOM on iPhone. Must build Release for on-device testing.
- **Bundle ID migration** — Switching from personal to paid developer team requires a new bundle ID (Apple takes 24-48h to release old ones). Moving 300GB of model data between app containers requires `UIDocumentPickerViewController` with `.moveToService`.
- **File Provider Storage penalty** — Models accessed via Files app integration go through the file coordination layer, adding I/O latency to every `pread`. Always import models to the app's own Documents directory.
- **`isExcludedFromBackup`** — Must be set on all model files to prevent iOS from purging 200GB+ of data during storage pressure events.

See [FlashMoE-iOS/IOS_PORT.md](FlashMoE-iOS/IOS_PORT.md) for the full iOS porting story and [FlashMoE-iOS/397B_ANALYSIS.md](FlashMoE-iOS/397B_ANALYSIS.md) for the 397B memory/performance analysis.

### Pipeline Per Layer (4.28ms average at 4-bit)

```
CMD3(prev) → CMD1: attention projections + delta-net  [1.22ms GPU]
           → CPU: flush results                       [0.01ms CPU]
           → CMD2: o_proj + norm + routing + shared    [0.55ms GPU]
           → CPU: softmax + topK routing               [0.003ms]
           → I/O: parallel pread K=4 experts           [2.41ms SSD]
           → CMD3: expert forward + combine + norm     [0.04ms encode, DEFERRED]
```

### Unified Memory Constraint

On Apple Silicon, SSD DMA and GPU compute share the same memory controller and cannot be profitably overlapped. The GPU's dequant kernels are bandwidth-saturated at ~418 GiB/s. Even small background SSD DMA causes disproportionate GPU latency spikes through memory controller arbitration. The serial pipeline (GPU → SSD → GPU) is hardware-optimal.

## Model Manager

The model manager helps you find, download, and validate compatible models:

```bash
# List local models and search HuggingFace for compatible ones
python model_manager.py

# Search HuggingFace only
python model_manager.py --search

# List local models only
python model_manager.py --local

# Download a specific model
python model_manager.py --download mlx-community/Qwen3.5-35B-A3B-4bit

# Check if a local model is compatible
python model_manager.py --check /path/to/model
```

After downloading, prepare the model for inference:

```bash
MODEL=~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit

# 1. Build expert index (maps tensor names to byte offsets)
python build_expert_index.py --model $MODEL

# 2. Pack expert weights into per-layer binary files
python repack_experts.py --index expert_index.json

# 3. Extract non-expert weights into a single mmap-friendly binary
python metal_infer/extract_weights.py --model $MODEL

# 4. Run inference (auto-detects weights in model directory)
cd metal_infer && ./infer --model $MODEL --prompt "Hello" --tokens 20
```

Pre-packed models are available on HuggingFace (no repacking needed):
- `alexintosh/Qwen3.5-35B-A3B-Q4-FlashMoE` (19.5 GB, 4-bit)
- `alexintosh/Qwen3.5-35B-A3B-Q4-Tiered-FlashMoE` (13.4 GB, tiered)

### Tiered Expert Quantization (Optional)

Reduces expert disk footprint by ~34% by keeping hot experts at 4-bit and requantizing cold experts to 2-bit. Recommended for memory-constrained setups:

```bash
# 1. Profile expert usage (run a few diverse prompts)
./infer --model <MODEL> --prompt "Explain quantum computing" --tokens 200 --freq 2>&1 | tee /tmp/freq1.txt
./infer --model <MODEL> --prompt "Write a Python function" --tokens 200 --freq 2>&1 | tee /tmp/freq2.txt

# 2. Generate hot expert manifest (80% coverage threshold)
python profile_experts.py --freq-output /tmp/freq1.txt /tmp/freq2.txt --coverage 0.8

# 3. Repack experts (creates packed_experts_tiered/)
python repack_experts_tiered.py --model <MODEL>

# 4. Run with --tiered (or auto-detected if packed_experts_tiered/ exists)
cd metal_infer && ./infer --model <MODEL> --tiered --prompt "Hello" --tokens 20
```

## Quick Start

```bash
cd metal_infer
make

# Run with a specific model (auto-detects architecture from config.json)
./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --prompt "Explain quantum computing" --tokens 100

# Or set FLASH_MOE_MODEL to avoid passing --model every time
export FLASH_MOE_MODEL=~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
./infer --prompt "Explain quantum computing" --tokens 100

# 2-bit inference (faster but breaks tool calling)
./infer --prompt "Explain quantum computing" --tokens 100 --2bit

# Tiered mode (hot=4-bit, cold=2-bit, auto-detected if packed_experts_tiered/ exists)
./infer --prompt "Explain quantum computing" --tokens 100 --tiered

# Interactive chat with tool calling (start server first, then chat client)
./infer --serve &
./chat

# Per-layer timing breakdown
./infer --prompt "Hello" --tokens 20 --timing
```

## Project Structure

```
model_manager.py       # Model discovery, download, and compatibility checking
build_expert_index.py  # Build expert_index.json from safetensors (step 1 of packing)
repack_experts.py      # 4-bit expert packing from safetensors (step 2)
profile_experts.py     # Expert frequency profiling → hot_experts.json
repack_experts_tiered.py  # Tiered repacking (hot=4-bit, cold=2-bit)
progress.py            # Results visualization (Q2/Q4 tracks)
results.tsv            # Experiment log (58 experiments)

metal_infer/
  infer.m              # Complete inference engine (~7500 lines)
                       #   - ModelConfig struct + config.json parser
                       #   - Runtime model auto-detection
                       #   - Metal compute pipeline
  shaders.metal        # Metal compute kernels (~1200 lines)
  chat.m               # Interactive chat TUI with tool calling
  tokenizer.h          # C BPE tokenizer (single-header, 449 lines)
  main.m               # MoE-only benchmark
  Makefile             # Build system
  extract_weights.py   # Creates model_weights.bin from safetensors
  repack_experts_2bit.py  # 4-bit → 2-bit expert requantization
  train_predictor.py   # Expert routing prediction analysis
  model_weights.bin    # Non-expert weights (model-specific, mmap'd)
  model_weights.json   # Tensor manifest
  vocab.bin            # Vocabulary for token decoding
  tokenizer.bin        # Pre-exported BPE tokenizer data

FlashMoE-iOS/              # Native iOS app
  FlashMoEEngine/
    FlashMoEEngine.h       # C API (create/load/generate/cancel/reset/destroy)
    FlashMoEEngine.m       # Unity build wrapping infer.m for iOS
  Bridge/
    FlashMoEBridge.swift   # @Observable async Swift wrapper
  Views/
    ChatView.swift         # Streaming chat UI with thinking disclosure
    ModelListView.swift    # Model discovery + download catalog
    ModelDownloadRow.swift # Download progress with pause/resume
    ProfilerView.swift     # Resource monitoring overlay
  Services/
    DownloadManager.swift  # Background URLSession model downloads
  Models/
    ModelCatalog.swift     # HuggingFace model registry with K recommendations
  App/
    FlashMoEApp.swift      # SwiftUI app entry point
  IOS_PORT.md              # Full iOS porting documentation
  397B_ANALYSIS.md         # 397B on iPhone: memory, Metal limits, K-reduction quality
  project.yml              # XcodeGen config (iOS 18+, iPhone only)
  copy_model_to_iphone.sh  # Push models to device over USB (pymobiledevice3)

autoresearch/              # Automated experiment loop
  program.md               # Agent instructions for autonomous optimization
  benchmark.sh             # Measurement harness with quality gates
  prepare.sh               # Baseline setup
  experiments.tsv          # Experiment log
```

## What We Tried (and What Worked)

### Autoresearch Wins (Automated Experiment Loop)
| Approach | Result | Impact |
|----------|--------|--------|
| Half-precision x_shared (v3 kernel) | Halve shared mem → 2× occupancy | **+12.1% tok/s** |
| FMA 2-bit dequant kernel | fma(nibble, scale*x, bias*x) | **+6.2% tok/s** |
| Half-precision x_shared (2-bit kernel) | Same occupancy trick | **+3.3% tok/s** |
| SIMD reduction in rms_norm_qk | simd_sum replaces serial loop | **+2.1% tok/s** |

### Kept (Manual)
| Approach | Result | Impact |
|----------|--------|--------|
| FMA dequant kernel | GPU compute -12% | **+12% tok/s** |
| Trust OS page cache | Deleted Metal LRU → +38% | **Foundational** |
| GPU combine+norm in CMD3 | Eliminates CPU round-trip | **Pipeline** |
| BLAS delta-net (Accelerate) | cpu_attn 0.78→0.28ms | **+64% attn** |
| F_NOCACHE for 2-bit | +3% from avoiding page thrash | **2-bit only** |
| GPU fused attention (RoPE) | +2% for full-attn layers | **Small** |
| C BPE tokenizer | 180ms vs 3500ms startup | **20x startup** |
| Deferred CMD3 execution | GPU/CPU overlap | **Pipeline** |
| Tiered expert quant (hot=4b, cold=2b) | -34% disk, same quality | **Cache hit rate** |

### Discarded (58 experiments, highlights)
| Approach | Result | Why |
|----------|--------|-----|
| LZ4 expert compression | -13% | Decompress overhead > warm cache savings |
| F_RDADVISE prefetch | net 0% | Unified memory: SSD DMA slows GPU -73% |
| Temporal expert prediction | -18% | 25% hit rate, SSD bandwidth waste |
| MLP routing predictor | 31% accuracy | Worse than temporal baseline |
| GPU LUT dequant kernel | -2% | Indirect register access serializes |
| GPU private buffer compression | -20% pipeline | Blit cost 4×7MB > matvec savings |
| Spin-poll GPU wait | -23% | CPU thermal competes with GPU |
| Expert file clustering | 0% | NVMe ignores scatter at 7MB granularity |
| dispatch_io | -70% | dispatch_data management overhead |
| mmap expert files | -5x | Per-page fault overhead on cold data |
| Speculative early routing | -38% | Cache pollution + overhead |
| MTP speculative decoding | break-even | MoE I/O scales per-token (unlike dense) |

### Discarded (iOS-specific)
| Approach | Result | Why |
|----------|--------|-----|
| Single 5.5GB Metal buffer (397B weights) | Crash | Metal hard limit: 4096 MB per buffer, not configurable |
| Two overlapping Metal buffers (~3GB each) | OOM kill | Metal tracks ~8GB shared memory on 12GB device |
| 50MB staging buffer + memcpy per dispatch | Data corruption | In-flight command buffers alias single staging buffer; later memcpys overwrite earlier tensor data before GPU reads |
| K=2 on 397B (trained K=10) | Gibberish | Only 20% of trained expert capacity fires, output distribution collapses |
| K=4 on 397B (trained K=10) | Degenerate ("!!!!") | 40% capacity insufficient for 512-expert model |
| File Provider Storage for model access | +latency | File coordination layer adds overhead to every pread |

## Safety

The engine explicitly controls memory:
- Non-expert weights: model-dependent (e.g., 5.5GB for 397B, ~1.5GB for 35B, mmap'd read-only)
- Metal scratch buffers: ~200MB (desktop), ~500MB (397B on iPhone with reduced context)
- Expert data streams from SSD on demand — no full model load required
- No custom caches. Trust the OS page cache for expert LRU.
- iOS: adaptive context length via `os_proc_available_memory()`, KV caches sized to fit device
- Minimum RAM: 8GB iPhone (35B), 12GB iPhone (397B with K=4), 24GB Mac (35B), 48GB Mac (397B)
