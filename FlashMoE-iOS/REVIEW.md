# Flash-MoE iOS: Engineering Review & Roadmap

Review of the iOS port after achieving 5.5 tok/s on iPhone 17 with Qwen3.5-35B-A3B.

## What's Working Well

### Architecture Decisions That Paid Off

**Unity build was the right call.** The `#include "infer.m"` approach meant zero fork divergence. Every iOS fix (shader loading, tokenizer paths, KV cache cap) was either a small addition to FlashMoEEngine.m or a backward-compatible change to infer.m. The engine runs identical compute paths on both platforms.

**Adaptive context via `os_proc_available_memory()`.** The power-of-2 clamping (512 → 8192) based on 25% of available RAM is elegant. Much better than the original hardcoded 2048 cap. This will automatically scale up on future iPhones with more RAM.

**`GPU_KV_SEQ` changed from `#define` to `static int`.** This was necessary — the old `#define GPU_KV_SEQ 8192` over-allocated Metal KV buffers regardless of actual context length. The iOS cap to 2048 would still waste 4× the memory on GPU-side KV caches. This fix saves ~150 MB of Metal buffer memory on a typical iPhone load.

**KV cache reuse (`flashmoe_generate_continuation`).** This is the single biggest UX feature. Without it, a 5-turn conversation would re-prefill all previous turns every message — at 5.5 tok/s, that's seconds of latency just to re-process history. The `-2` return code for "context full" is a clean signal to the Swift layer.

**The profiler overlay.** `os_proc_available_memory()`, `mach_task_basic_info`, `ProcessInfo.thermalState` — all public APIs, no entitlements. This is exactly what you need when debugging memory pressure on a phone running a 35B model.

## Concerns

### 1. Memory Safety: Silent NULL from calloc

**Severity: High**

iOS doesn't overcommit memory the way macOS does. When `calloc` fails on iPhone (because you're close to the ~6 GB app limit), it returns NULL silently. The engine has zero NULL checks after calloc/malloc in critical paths:

```c
// FlashMoEEngine.m line 266-267
ctx->hidden = calloc(cfg.hidden_dim, sizeof(float));
ctx->logits = calloc(cfg.vocab_size, sizeof(float));
// If either is NULL → immediate EXC_BAD_ACCESS in the next forward pass
```

Same in `kv_cache_new()`, `linear_attn_state_new()`, `metal_setup()` buffer allocations, and `embed_batch` malloc in the generation loop.

**Fix:** Add NULL checks after every allocation in `flashmoe_load()`. If any fail, set `ctx->last_error` and return -1. Also add a pre-flight memory check: compute total expected allocation before starting, compare to `os_proc_available_memory()`, and fail early with a clear error message.

### 2. Metal Buffer Allocation Without Error Handling

**Severity: High**

`metal_setup()` allocates ~40+ Metal buffers. On iPhone, Metal can return `nil` for buffer allocations when GPU memory is exhausted. None of these are checked:

```c
ctx->buf_input = [ctx->device newBufferWithLength:max_in options:MTLResourceStorageModeShared];
// If nil → next dispatch that uses buf_input crashes
```

The multi-expert buffers are the biggest concern — `MAX_K=8` double-buffered expert data buffers at `expert_alloc_size` each (rounded up to 2 MB alignment). For the 35B model that's `8 × 2 × ~7 MB = ~112 MB` just for expert data buffers.

**Fix:** Check every `newBufferWithLength:` return. If any critical buffer fails, log which one, free what was allocated, and return NULL from `metal_setup()`. Consider reducing MAX_K on iPhone (the 35B model uses K=8, but on a memory-constrained device, K=4 with quality degradation might be preferable to a crash).

### 3. posix_memalign in metal_setup May Fail on iOS

**Severity: Medium**

```c
posix_memalign(&aligned_data, 2*1024*1024, expert_alloc_size);
```

This 2 MB alignment is optimal for DMA on MacBook NVMe. On iPhone:
- The NVMe controller may have different alignment preferences
- `posix_memalign` can fail (return non-zero) and leave the pointer uninitialized
- 2 MB pages may not even be available under memory pressure

**Fix:** Check the return value. Consider whether 2 MB alignment matters on iPhone NVMe — it might, or the optimal alignment might be different (64 KB is iOS's large page size). Profile with and without to see if it makes a difference on A18 Pro.

### 4. Thread Safety of Static Globals

**Severity: Medium**

The engine uses ~30 static globals (`cfg`, `g_metal`, `g_deferred`, `g_use_tiered`, etc.). The `FlashMoEContext` struct wraps some state, but the actual compute still reads/writes the globals. This means:

- Two `FlashMoEContext` instances would stomp each other's state
- Calling `flashmoe_generate` from one thread while `flashmoe_get_stats` reads from another could race on `g_deferred`

This is fine today (single-instance design), but it's a landmine. The `@unchecked Sendable` on `FlashMoEEngine` in Swift acknowledges this.

**Recommendation:** Add a static mutex or `dispatch_once` guard in `flashmoe_create()` that enforces single-instance. If someone calls `flashmoe_create()` while another context exists, return NULL with an error.

### 5. Conversation State Leak on Context Full

**Severity: Medium**

When `flashmoe_generate_continuation` returns -2 (context full), the Swift bridge falls back to `flashmoe_generate` (full re-prefill). But the KV caches and delta-net state still contain the old conversation. The `flashmoe_generate` function resets everything, so this is _probably_ fine, but there's a window where:

1. Continuation returns -2
2. Swift calls `flashmoe_generate` with full chat template
3. `flashmoe_generate` resets state at line 389-395
4. New conversation starts clean

The concern is if the Swift layer doesn't handle -2 correctly and tries another continuation. Add a guard: after returning -2, set a flag that blocks further continuations until a full generate or reset.

### 6. malloc/free per Token in Generation Loop

**Severity: Low (performance)**

Every token in the generation loop does:
```c
float *normed = malloc(cfg.hidden_dim * sizeof(float));
cpu_rms_norm(ctx->hidden, ctx->final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
memcpy(ctx->hidden, normed, cfg.hidden_dim * sizeof(float));
free(normed);
```

That's a `malloc(8192)` + `free()` per token. On macOS with the magazine allocator this is ~20 ns, but on iOS under memory pressure, the allocator may need to do more work. At 5.5 tok/s this is ~180 ms per token — a 20 ns malloc is negligible. But it's easy to fix.

**Fix:** Pre-allocate `normed` once in `FlashMoEContext` alongside `hidden` and reuse it. Same for `embed_batch` — allocate max size once at load time.

### 7. Download Manager: No Integrity Verification

**Severity: Medium**

Downloaded files are used directly without checksum verification. A corrupted download (network interruption, disk error) would produce garbage inference results or crashes deep in the Metal pipeline. Users would see nonsense output and blame the model.

**Fix:** Add SHA-256 checksums to `ModelCatalog.swift` (HuggingFace provides these in the repo metadata). Verify after each file download. Re-download if mismatch.

### 8. F_RDAHEAD Disable on iOS

**Severity: Low (investigate)**

```c
fcntl(ctx->layer_fds[i], F_RDAHEAD, 0);
```

This was tested on MacBook where expert reads are random and readahead wastes SSD bandwidth. On iPhone, the SSD controller and page cache behavior may differ. The A18 Pro's NVMe controller might benefit from readahead for the sequential-within-expert reads (each expert is ~7 MB of sequential data). Worth A/B testing.

## Feature Ideas

### Priority 1: Ship Quality

1. **Memory pre-flight check.** Before loading, compute total expected allocation (weights mmap + Metal buffers + KV caches + working buffers) and compare to available memory. Show a clear error: "This model needs ~2.8 GB but only 2.1 GB is available. Close other apps or try the Tiered model (saves 34%)."

2. **Tiered as default recommendation.** The tiered model is 13.4 GB vs 19.5 GB — that's 6 GB less download and better page cache utilization on iPhone's smaller RAM. The catalog should surface this prominently. Users on iPhone 15 Pro (8 GB) should probably always use tiered.

3. **Download resume.** If the app is killed mid-download, resume from where it left off. `URLSession` background downloads support this natively — the `DownloadManager` should persist download progress and use `Range` headers.

### Priority 2: Performance

4. **Benchmark F_RDAHEAD on iPhone.** The MacBook conclusion ("disable readahead") may not hold on A18. Run 100 tokens with and without `F_RDAHEAD` and compare tok/s.

5. **Profile Metal occupancy on A18 GPU.** The half-precision `x_shared` optimization was tuned for M3 Max (40 cores). The A18 Pro has 6 GPU cores with different shared memory and occupancy characteristics. The optimal threadgroup size and shared memory usage may differ. Use Metal System Trace to check.

6. **Expert buffer count tuning.** MAX_K=8 allocates 8 double-buffered expert data slots (16 total). The 35B model uses K=8, but if memory is tight, consider K=4 with top-4 routing (lose some quality, save ~56 MB of Metal buffers).

### Priority 3: UX Polish

7. **Streaming think/reply separation.** The current `<think>` parsing works on completed text but can flicker during streaming. Consider buffering the think block until `</think>` is seen, then revealing it as a disclosure group.

8. **Token-level latency display.** The profiler shows average tok/s, but users care about _consistency_. Show a mini sparkline of per-token latency — this would reveal if certain layers have cache misses (spiky) vs warm cache (smooth).

9. **Model size warning.** Before downloading a 19.5 GB model, check available disk space and warn if it won't fit. iOS doesn't handle "disk full" gracefully.

10. **Background inference continuation.** When the app goes to background during generation, iOS suspends the process. The generation resumes when foregrounded, but the tok/s stat is wrong (counts suspended time). Detect `UIApplication.didEnterBackgroundNotification` and pause the timer.

### Priority 4: Future

11. **Smaller models.** The 35B-A3B at 19.5 GB is a hard sell for most users. If/when Qwen releases a 7B or 14B MoE variant, that would be the mass-market iOS model. The engine already auto-detects architecture from config.json.

12. **On-device expert profiling for tiered.** Run `profile_experts.py` logic on-device after the first few conversations to identify hot experts for _this user's_ workload. Requantize cold experts to 2-bit in the background. Personalized tiered quantization.

13. **Widget / Live Activity.** Show generation progress as a Live Activity on the lock screen. "Flash-MoE: generating... 142 tokens, 5.3 tok/s". Great for virality.

14. **Shortcuts integration.** Expose `flashmoe_generate` as a Siri Shortcut action. "Ask Flash-MoE: what's the weather like?" — runs inference locally, returns the answer to Shortcuts.

## Additional Concerns (from laptop-side review)

### 9. No Response to iOS Memory Warnings

The app doesn't observe `UIApplication.didReceiveMemoryWarningNotification`. When iOS sends this (which it will — we're using ~2.5GB), the app should at minimum cancel in-flight generation and log the event. Without it, iOS will jettison the app with no recovery. This is the #1 crash risk in the field.

**Fix**: In `AppDelegate.swift` or via `NotificationCenter`, observe memory warnings. Cancel generation, optionally show an alert, and consider releasing mmap'd expert data (re-openable on demand).

### 10. Thermal Throttling Goes Unhandled

The profiler shows thermal state but the engine doesn't react to it. Sustained inference at 5.5 tok/s will push iPhone 15 Pro into `.serious` thermal state within 2-3 minutes. Apple throttles both GPU and CPU at this point, creating a death spiral (slower → longer generation → more heat).

**Fix**: When `ProcessInfo.thermalState >= .serious`, insert a small delay between tokens (100-500ms) or pause with a "Phone is cooling down" overlay. This prevents the thermal runaway and gives a better user experience than silently degrading to 2 tok/s.

### 11. EOS Token Not Fed Back on Continuation

When generation stops on EOS (`<|im_end|>`), the EOS token is detected and the loop breaks — but that token's embedding was never fed through the model's forward pass. The KV cache doesn't have the EOS token's contribution. When `flashmoe_generate_continuation` resumes with `\n<|im_start|>user\n`, the model sees a conversation where the assistant turn didn't properly end.

This may or may not matter in practice (the model might be robust to it), but it's technically incorrect. Fix: after the generation loop breaks on EOS, feed the EOS token through one final forward pass (embed + all layers + discard) to properly close the turn in the KV cache.

### 12. Share Sheet for Viral Moments

Add a "Share" button on each assistant message that creates a formatted card: "Generated locally on iPhone with Flash-MoE • 5.5 tok/s • 35B parameters • No cloud". This is free viral marketing.

### 13. Free Space Check Before Download

The ModelListView shows model size (19.5 GB) but not device free space. Users start a 19.5GB download without knowing if they have room. Add `FileManager.default.attributesOfFileSystem(forPath:)` to show "X GB free" near the download button.

## Performance Notes

| Metric | MacBook Pro M3 Max | iPhone 17 (A19) | Ratio |
|--------|-------------------|-----------------|-------|
| tok/s | 9.7 | 5.5 | 57% |
| SSD bandwidth | 17.5 GB/s | ~3-4 GB/s (est.) | ~20% |
| GPU cores | 40 | 6 | 15% |
| Memory | 48 GB | 8 GB | 17% |
| Expert size (4-bit) | 18.1 GB | 18.1 GB | 100% |

The iPhone achieves 57% of MacBook speed despite having 15% of the GPU cores and 20% of the SSD bandwidth. This suggests the bottleneck on iPhone is _not_ GPU compute (the 6 cores are fast enough) but SSD streaming. The expert reads (~7 MB × K=8 × 40 layers = 2.24 GB/token) at ~3 GB/s would give ~1.3 tok/s if purely I/O bound — the 5.5 tok/s means the OS page cache is hitting ~75%+ of expert reads, similar to the MacBook's ~71% hit rate. The "Trust the OS" principle holds on iOS.

The tiered model should be even faster on iPhone — smaller expert files mean higher page cache hit rate with only 8 GB of RAM managing the cache.
