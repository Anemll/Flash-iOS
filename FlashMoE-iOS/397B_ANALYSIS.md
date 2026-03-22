# Running Qwen3.5-397B on iPhone — Technical Analysis

## Status Summary

| Aspect | Status | Detail |
|--------|--------|--------|
| Model loading | Works | Config parsed, weights mmap'd, expert FDs opened |
| 35B inference (GPU) | Works | 5.5 tok/s, full quality |
| 397B inference (CPU fallback) | Works but impractical | ~6 min/token, correct output |
| 397B inference (GPU) | Blocked | 5.5GB weights exceed Metal 4GB per-buffer limit |
| K-reduction (K=2) | Gibberish | "no manager" + random tokens |
| K-reduction (K=4) | Degenerate | "!!!!" repeated output |
| K-reduction (K=6+) | Untested | Needs GPU path to evaluate at reasonable speed |

## Critical Issues (Resolved)

### 1. MAX_K=8 but 397B needs K=10

The engine hardcoded `#define MAX_K 8` in the MetalCtx struct (infer.m). The 397B model's config specifies `num_experts_per_tok=10`. Loading with default K=10 writes past the end of `buf_multi_expert_data[8]` — buffer overflow, instant crash.

**Fix applied**: Bumped MAX_K to 16 and added runtime cap `min(K, MAX_K)` as a safety net.

### 2. Weight Auto-Detection Bug

The K-reduction branch added weight auto-detection in the model directory, but the `default_weights` char array wasn't initialized when `model_path` is NULL. Reading `default_weights[0]` is undefined behavior.

**Fix applied**: Initialize `default_weights[0] = '\0'` (and same for manifest/vocab) at declaration.

## Critical Issue (Open): Metal 4GB Per-Buffer Limit

### The Problem

Metal on iOS enforces a **hard 4GB per-buffer limit**. The 397B model's non-expert weights (`model_weights.bin`) are 5.5GB.

### What We Tried

| Approach | Outcome | Root Cause |
|----------|---------|-----------|
| Single Metal buffer (5.5GB) | Metal assertion crash | `buffer must not exceed 4096 MB` — hard platform limit, not configurable |
| Two overlapping Metal buffers (~3GB each) | OOM kill | Metal internally tracks ~8GB of shared memory. On a 12GB device, iOS kills the app. |
| 50MB staging buffer (memcpy per dispatch) | **Data corruption** | Single staging buffer shared across N in-flight command buffers. Metal executes all encoders AFTER `commit`, so CPU memcpys for later tensors overwrite data before GPU reads earlier tensors. Fundamental architectural flaw — would need N staging buffers for N concurrent command buffers. |
| CPU fallback (Accelerate BLAS) | Works, 6 min/token | Weight matmuls run on CPU. Expert forward still uses GPU (small per-expert buffers). Correct output but unusable. |

### Why Entitlements Don't Help

- `extended-virtual-addressing` — expands total virtual address space, not per-buffer limit
- `increased-memory-limit` — raises the total memory ceiling for the process, not per-buffer

Neither entitlement changes the 4GB per-Metal-buffer hard cap.

### The Solution: Split Weight Files

Split `model_weights.bin` at the Python packing stage into two files:

```
model_weights_0.bin  (~3.0 GB)  — layers 0-29 tensors
model_weights_1.bin  (~2.5 GB)  — layers 30-59 tensors
```

Each file gets its own Metal buffer (both under 4GB). The tensor manifest (`model_weights.json`) gains a `file_index` field so `infer.m` selects the correct mmap base pointer per tensor.

**Engineering work required**:
1. `extract_weights.py` — emit two files with a configurable split point
2. `model_weights.json` — add file index per tensor entry
3. `infer.m` — mmap both files, index tensors to correct buffer
4. Re-upload 397B model to HuggingFace

## Memory Budget Analysis

### KV Cache

GPU_KV_SEQ was hardcoded to 8192. For the 397B with head_dim=256, num_kv_heads=2:

| Component | Formula | Size |
|-----------|---------|------|
| KV cache | 15 layers x 2 buffers x seq x 512 x 4B | 504 MB (seq=8192), **126 MB (seq=2048)** |
| Attn scores | 32 heads x seq x 4B x 15 layers | 15 MB (seq=8192), 4 MB (seq=2048) |
| Delta-net state | 45 layers x 64 x 128 x 128 x 4B | 188 MB |
| Delta-net conv | 45 layers x 3 x 8192 x 4B | 4 MB |
| Expert buffers | MAX_K x 2 x 7MB | 112 MB (MAX_K=8), 224 MB (MAX_K=16) |
| Working buffers | — | ~50 MB |

**Fix applied**: Adaptive context length reduces KV seq to 2048 on iPhone, saving ~378 MB.

### Total Memory Budget (iPhone 17, 12GB)

```
Non-expert weights (mmap'd):      5.5 GB  (virtual, paged on demand)
Metal buffers (KV=2048):          ~500 MB
iOS overhead:                     ~2.0 GB
                                  --------
Resident estimate:                ~3.0 GB
Available for expert page cache:  ~9.0 GB
```

### GQA Ratio

The 397B uses 32 query heads but only 2 KV heads (16:1 GQA ratio). The 35B uses 32:4 (8:1). The attention code handles this via `cfg.num_kv_heads` dynamically with `kv_h = h / (num_attn_heads / num_kv_heads)`. The 16:1 ratio works correctly since it's computed from config values.

## Performance Projections

### With GPU Path (After Split Weights)

| Configuration | Expert I/O | I/O Time | GPU Compute | Total | tok/s |
|--------------|-----------|----------|-------------|-------|-------|
| K=10, 4-bit | 10 x 60 x 6.75MB = 4.1 GB | ~1.6s | ~0.5s | ~2.1s | ~0.5 |
| K=4, 4-bit | 4 x 60 x 6.75MB = 1.6 GB | ~0.65s | ~0.4s | ~1.05s | **~1.0** |
| K=4, tiered | 4 x 60 x ~4.5MB = 1.1 GB | ~0.43s | ~0.4s | ~0.83s | **~1.2** |

iPhone NVMe throughput assumed at ~2.5 GB/s.

### Page Cache Effectiveness

With ~9 GB free for page cache and 208 GB of expert data:

- Cache can hold ~1,330 experts out of 30,720 total (60 layers x 512)
- Only 240 experts used per token at K=4 (4 x 60 layers)
- Expert usage follows Zipfian: ~25% of experts handle 80% of activations
- Expected page cache hit rate: **10-15%** (most reads are cache misses)
- With tiered (cold=2-bit): smaller expert files, maybe **15-20%** hit rate

The 397B on iPhone will be heavily I/O bound. Every optimization to reduce I/O (lower K, tiered, expert compression) matters more than GPU optimization.

### Storage Requirements

| Configuration | Expert Disk | Total Disk | Min iPhone Storage |
|--------------|------------|------------|-------------------|
| K=10, 4-bit | 208 GB | 214 GB | 256 GB |
| K=4, 4-bit | 208 GB | 214 GB | 256 GB |
| K=4, tiered | ~140 GB | 146 GB | 256 GB |
| K=4, 2-bit | ~104 GB | 110 GB | 128 GB |

K-reduction does NOT save storage (all 512 experts per layer remain on disk for routing). Only tiered/2-bit quantization reduces disk footprint.

## K-Reduction Quality Analysis

### Why K=2 and K=4 Fail on 397B

The 397B model was trained with K=10 active experts per token. Each expert contributes a weighted fraction of the final hidden state. Reducing K means:

- **K=2**: Only 20% of the trained expert capacity fires. The router picks the top 2, but the model's residual stream expects contributions from ~10 experts. The output distribution collapses.
- **K=4**: 40% of trained capacity. Better, but for a model this large (512 experts, K=10), each expert is specialized enough that missing 60% of them produces degenerate patterns.
- **K=6+**: Untested. The 35B model (trained K=8) works perfectly at K=8, suggesting models tolerate reduction better when the gap is smaller relative to training K.

### Testing Constraints

K-reduction quality can only be properly evaluated with the GPU path enabled (split weight files). The CPU fallback at 6 min/token makes quality evaluation impractical — you need hundreds of tokens to judge coherence, which would take hours.

## Action Items (Priority Order)

1. **Split `model_weights.bin` into two <4GB files** — enables GPU path on iOS
2. **Test K=6, K=8, K=10 with GPU path** — find minimum viable K for coherent 397B output
3. **Upload split 397B model to HuggingFace** — make it downloadable in iOS app
4. **Profile 397B tiered experts** — repack hot=4-bit cold=2-bit, reduce storage to ~140 GB
5. **Benchmark actual I/O throughput on A19** — validate 2.5 GB/s assumption
6. **Test thermal behavior** — sustained 397B inference may thermal-throttle the A19
