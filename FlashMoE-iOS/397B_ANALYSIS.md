# Running Qwen3.5-397B on iPhone — Technical Analysis

## Critical Issues (Will Crash)

### 1. MAX_K=8 but 397B needs K=10

The engine hardcodes `#define MAX_K 8` in the MetalCtx struct (infer.m:1384). The 397B model's config specifies `num_experts_per_tok=10`. If the engine loads with the default K=10, it will write past the end of `buf_multi_expert_data[8]` — buffer overflow, instant crash.

**Fix options:**
- (a) Bump `MAX_K` to 10 (or 16 for safety). Costs 2 extra expert buffer slots × 7MB × 2 (double-buffered) = ~28 MB more Metal memory.
- (b) Cap K to `min(K, MAX_K)` at runtime in FlashMoEEngine.m's load function.
- (c) Both — bump MAX_K and add the runtime cap as a safety net.

**Recommendation:** (c). Bump MAX_K to 16 and add the cap. 16 covers any future model.

### 2. Weight Auto-Detection Bug (infer.m)

The K-reduction branch added weight auto-detection in the model directory, but the `default_weights` char array isn't initialized when `model_path` is NULL. Reading `default_weights[0]` is undefined behavior.

**Fix:** Initialize `default_weights[0] = '\0'` (and same for manifest/vocab) at declaration.

## Important Issues (Correctness / Memory)

### 3. KV Cache Blows Memory Budget

GPU_KV_SEQ is hardcoded to 8192. For the 397B with head_dim=256, num_kv_heads=2:

```
KV cache = 15 layers × 2 buffers × 8192 × 512 × 4 bytes = 504 MB
Attn scores = 32 heads × 8192 × 4 bytes × 15 = 15 MB
Total attention buffers: ~520 MB
```

Plus delta-net state for 45 linear layers:
```
State: 45 × (64 × 128 × 128 × 4) = 188 MB
Conv: 45 × (3 × 8192 × 4) = 4 MB
Total delta-net: ~192 MB
```

Plus multi-expert buffers:
```
Expert data: MAX_K × 2 × 7MB = 112 MB (at MAX_K=8)
Working buffers: ~50 MB
```

**Total Metal: ~875 MB**

With 5.5 GB mmap'd weights + 875 MB Metal + ~2 GB iOS = **8.4 GB resident**. On 12 GB iPhone, only **3.6 GB** left for expert page cache.

**Fix:** Reduce GPU_KV_SEQ for iPhone. Context of 2048 tokens is plenty for a chat demo:
```
2048 instead of 8192 → KV cache drops from 504 MB to 126 MB
Saves ~378 MB → total Metal ~500 MB
Leaves ~5.5 GB for page cache (much better hit rates)
```

Wire this through `FlashMoEConfig.max_context` — already supported, just needs a sensible default for iPhone.

### 4. Verify 397B Tensor Names Actually Load

The warnings you saw ("tensor not found") were because `infer.m` was loading the wrong `model_weights.bin` (35B instead of 397B). With the weight auto-detection fix, this should resolve. But we should verify ALL tensor names match:

- The 397B is a VL model; `extract_weights.py` strips `language_model.` prefix → should produce `model.layers.X.*`
- `build_layer_cache()` looks for `model.layers.%d.self_attn.q_proj.weight` etc.
- Need to confirm: does the 397B use `self_attn.q_proj` or some other naming? (Answer: yes, same naming, verified in the manifest)

### 5. num_kv_heads=2 with GQA

The 397B uses 32 query heads but only 2 KV heads (16:1 GQA ratio). The 35B uses 32:4 (8:1). The attention code handles this via `cfg.num_kv_heads` dynamically, but the GQA mapping `kv_h = h / (num_attn_heads / num_kv_heads)` needs to work with 16:1 ratio. This should be fine since it's computed from config values, but worth a sanity check.

## Performance Considerations

### 6. Expected Performance on iPhone 17 (12 GB, A19)

With K=4 (reduced from K=10):
```
Expert I/O per token: 4 experts × 60 layers × 6.75 MB = 1,620 MB
iPhone NVMe: ~2.5-3 GB/s
I/O time: ~0.54-0.65s per token

GPU compute per token: 60 layers × (attention + routing + expert forward)
  - Attention: 32 heads × 256 dim → heavier than 35B
  - Expert forward: only 4 experts (vs 8 on 35B), but per-expert is same size
  - Estimate: ~0.3-0.5s compute per token

Total: ~0.9-1.1s per token → ~0.9-1.1 tok/s
```

With K=4 + tiered (hot=4-bit, cold=2-bit):
```
Average expert size drops ~34% → ~4.5 MB average
I/O: 4 × 60 × 4.5 = 1,080 MB → 0.36-0.43s
Total: ~0.7-0.9s → ~1.1-1.4 tok/s
```

### 7. Page Cache Effectiveness

With 5 GB free for page cache and 208 GB of expert data:
- Cache can hold ~740 experts out of 30,720 total (60 layers × 512)
- Only 240 experts are used per token (K=4 × 60 layers)
- Expert usage follows Zipfian: ~25% of experts handle 80% of activations
- Hot set size: ~60 layers × 128 hot experts = 7,680 experts × 6.75 MB = ~52 GB
- With tiered (cold=2-bit): hot set = 7,680 × 6.75 + 23,040 × 3.8 = ~139 GB
- Expected page cache hit rate: 10-15% (most reads are cache misses)
- With tiered + reduced expert sizes: maybe 15-20%

The 397B on iPhone will be heavily I/O bound. Every optimization to reduce I/O (lower K, tiered, expert compression) matters more than GPU optimization.

### 8. Storage Requirements

| Configuration | Expert Disk | Total Disk | Min iPhone |
|--------------|------------|------------|------------|
| K=10, 4-bit | 208 GB | 214 GB | 256 GB |
| K=4, 4-bit | 208 GB | 214 GB | 256 GB |
| K=4, tiered | ~140 GB | 146 GB | 256 GB |
| K=4, 2-bit | ~104 GB | 110 GB | 128 GB |

K-reduction doesn't save storage (all experts are still on disk), but tiered/2-bit does.

## Action Items (Priority Order)

1. **Bump MAX_K to 16 + runtime cap** — prevents crash, 5 min fix
2. **Fix uninitialized default_weights** — prevents UB, 1 min fix
3. **Add max_context default for iPhone** — reduce KV cache to 2048, saves 378 MB
4. **Repack 397B tiered experts** — profile hot experts, repack, reduces storage to ~140 GB
5. **Upload 397B to HuggingFace** — needs the weight auto-detection fix first
6. **Test on actual hardware** — verify end-to-end: download → load → generate on iPhone
