# Flash-MoE iOS Port: From MacBook to iPhone

How we took a 397B-parameter MoE inference engine written in C/Metal and made it run on an iPhone — getting 5.5 tok/s on the 35B model, with the 397B model loading but hitting hard platform limits.

## The Challenge

The desktop Flash-MoE engine was designed for a MacBook Pro with 48GB unified memory, 40-core GPU, and 17.5 GB/s NVMe. iPhones have 8-12GB RAM, ~10-core GPU, ~2.5 GB/s NVMe, and no filesystem access to `shaders.metal` at runtime. Every assumption in the 7,500-line inference engine needed to be re-examined.

## Results

| Device | Model | K | tok/s | Notes |
|--------|-------|---|-------|-------|
| iPhone 17 (12GB, A19) | Qwen3.5-35B-A3B | 8 | **5.5** | Full quality, 19.5GB download, full GPU path |
| iPhone 17 (12GB, A19) | Qwen3.5-35B-A3B (tiered) | 8 | **5.5+** | 13.4GB download, same quality |
| iPhone 17 (12GB, A19) | Qwen3.5-397B-A17B | 4 | ~0.003* | *CPU fallback only. 6 min/token. |
| MacBook Pro M3 Max (48GB) | Qwen3.5-35B-A3B | 8 | **9.7** | After autoresearch optimizations |
| MacBook Pro M3 Max (48GB) | Qwen3.5-397B-A17B | 4 | **4.4** | K-reduced from K=10 |

*397B model loads and generates correct tokens but uses CPU fallback for weight matmuls because the 5.5GB `model_weights.bin` cannot fit in Metal buffer(s) on a 12GB device. See [397B_ANALYSIS.md](397B_ANALYSIS.md) for the full breakdown.

iPhone achieves **57% of laptop speed** on the 35B model with 17% of the memory.

## What We Built

A native SwiftUI iOS app wrapping the C/Metal inference engine with:

- **Model discovery**: on-device scanning + HuggingFace download catalog
- **Interactive chat**: streaming tokens, thinking animation with disclosure triangle, text selection
- **Model management**: import from Files app, export via `moveToService`, swipe-to-delete
- **Settings**: K-reduction picker (K=2/4/6/8/10), I/O fanout picker (off/2/4/8 chunks)
- **Profiler view**: resource monitoring overlay with thermal state indicator (Cool/Warm/Hot/Critical)
- **Multi-turn chat**: KV cache reuse across chat turns, adaptive context length by device RAM
- **Background downloads**: `URLSession` downloads from HuggingFace with resume support
- **Cross-app model access**: security-scoped bookmarks for reading models from other app containers

**Architecture**: SwiftUI (UI + @Observable state) -> Swift async bridge (AsyncStream) -> Objective-C wrapper -> C inference engine (7,500 lines) -> Metal GPU shaders (1,200 lines)

## Performance Optimizations (Autoresearch)

Before the iOS port, we ran Karpathy's autoresearch pattern — an autonomous experiment loop that modifies Metal shaders, benchmarks, and keeps/discards based on tok/s. 10 experiments, 4 kept:

| Experiment | Description | Impact | Why It Works |
|-----------|-------------|--------|-------------|
| SIMD reduction | Replace serial thread-0 accumulation in `rms_norm_qk`/`gated_rms_norm` with `simd_sum` + shared memory reduction | +2.1% | Eliminates serial bottleneck in GPU reduction — 4 SIMD groups of 32 threads each contribute partial sums |
| FMA 2-bit kernel | Apply `fma(nibble, scale*x, bias*x)` pattern to 2-bit dequant (same trick as 4-bit v3) | +6.2% | GPU fused multiply-add does dequant+multiply in one instruction instead of two |
| Half-precision x_shared (v3) | Store threadgroup shared memory input cache as `half` instead of `float` | +12.1% | Halves shared memory from 16KB to 8KB, doubles GPU occupancy. Input values are already approximate from dequantization so half precision loses nothing |
| Half-precision x_shared (2-bit) | Same trick applied to 2-bit kernel | +3.3% | Same occupancy benefit, smaller because 2-bit kernel is more I/O bound |

**Combined: +15.3% theoretical, +34.7% real-world** (from 7.2 to 9.7 tok/s on MacBook Pro).

6 experiments discarded (single compute encoder -5%, FMA without shared memory -15%, extended x_shared to 8192 -7%, three others marginal).

## Problems Solved (iOS Port)

### 1. Metal Shader Loading

**Problem**: The desktop engine compiles `shaders.metal` from source at runtime via `newLibraryWithSource:`. On iOS, there is no filesystem path to the shader file.

**Fix**: Runtime fallback — try `[device newDefaultLibrary]` first (loads pre-compiled `default.metallib` from the app bundle), fall back to source compilation for macOS CLI:

```objc
ctx->library = [ctx->device newDefaultLibrary];
if (ctx->library) {
    // iOS: loaded from bundle
} else {
    // macOS: compile from source
    NSString *src = [NSString stringWithContentsOfFile:@"shaders.metal" ...];
    ctx->library = [ctx->device newLibraryWithSource:src ...];
}
```

**Xcode fix**: Moved `shaders.metal` from the **Resources** build phase to **Sources** so Xcode's Metal compiler produces `default.metallib` in the app bundle.

### 2. Memory: KV Cache OOM

**Problem**: The model's `max_position_embeddings` is 131,072 (128k context). KV cache allocation per full-attention layer: `131072 * kv_heads * head_dim * 4 bytes`. For the 35B (10 full-attn layers): ~2.5GB just for KV caches. `calloc` silently returns NULL on iPhone, causing `EXC_BAD_ACCESS` crashes.

**Fix**: Adaptive context length based on `os_proc_available_memory()`. Budget 25% of available memory for KV caches, clamp to power-of-2 sizes (512-8192). On a 12GB iPhone this yields ~2048 context, keeping KV caches under ~40MB total. Context window is limited but sufficient for chat.

### 3. Memory: Metal Debug Layer

**Problem**: Debug builds wrap every Metal object with `MTLDebugComputeCommandEncoder` validation proxies, adding ~2GB of overhead — roughly doubling GPU memory usage. The app crashes with `NSMallocException` trying to allocate debug wrappers.

**Fix**: Build in **Release** mode and disable Metal API Validation in the Xcode scheme. The debug overhead is too large for iPhone's memory budget.

### 4. Tokenizer Not Found

**Problem**: `init_tokenizer()` searches for `tokenizer.bin` at relative filesystem paths (`./tokenizer.bin`, `./metal_infer/tokenizer.bin`). These don't exist on iOS.

**Fix**: Extended the search to check the model directory (where it's downloaded) and the app bundle:

```objc
// Try model directory (downloaded with model)
snprintf(model_tok, sizeof(model_tok), "%s/tokenizer.bin", cfg.model_path);
if (access(model_tok, R_OK) == 0) { bpe_load(&g_tokenizer, model_tok); }

// Try app bundle
NSString *bundlePath = [[NSBundle mainBundle] pathForResource:@"tokenizer" ofType:@"bin"];
if (bundlePath) { bpe_load(&g_tokenizer, [bundlePath UTF8String]); }
```

### 5. Missing Info.plist

**Problem**: Xcode target configs had `INFOPLIST_KEY_*` entries but never set `GENERATE_INFOPLIST_FILE = YES`, so no `Info.plist` was produced.

**Fix**: Added `GENERATE_INFOPLIST_FILE = YES` to both Debug and Release target build settings.

### 6. Chat Template (Garbage Output)

**Problem**: The model received raw text ("Hi") instead of Qwen's chat template format. Without the `<|im_start|>` / `<|im_end|>` markers, the model treats input as a continuation of arbitrary text, producing incoherent output.

**Fix**: Added `buildChatPrompt()` in `ChatView.swift` that formats the full conversation history using Qwen's `<|im_start|>system/user/assistant<|im_end|>` template.

### 7. Special Token Leakage

**Problem**: End-of-turn tokens like `<|im_end|>` appear as visible text in the chat UI.

**Fix**: Strip special tokens from the token stream before displaying.

### 8. KV Cache Reuse Across Turns

**Problem**: Each new message re-tokenizes and re-prefills the entire conversation history. On a model generating at 5.5 tok/s, re-prefilling 200+ tokens of history adds seconds of latency per turn.

**Fix**: Added `flashmoe_generate_continuation()` — a new C API that reuses existing KV cache state. The engine tracks conversation position and only processes the new user turn tokens. Returns -2 if context is full, signaling the Swift bridge to reset and do a full re-prefill.

### 9. MAX_K Buffer Overflow

**Problem**: The engine hardcodes `#define MAX_K 8` for multi-expert buffer arrays. The 397B model needs K=10 (default). Without K-reduction (or bumping MAX_K), loading the 397B model causes a buffer overflow crash writing past `buf_multi_expert_data[8]`.

**Fix**: Bumped MAX_K to 16 and added a runtime cap `min(K, MAX_K)` as a safety net.

### 10. Race Condition in Async pread

**Problem**: The pthread-pool-based async pread had generation counter conflicts when multiple dispatch groups accessed shared state.

**Fix**: Replaced pthread pool with GCD `dispatch_group` for expert reads. Each expert read gets its own dispatch group, eliminating the shared counter race.

### 11. Tiered Validation Bug

**Problem**: `async_pread_wait` validated each chunk against a uniform 4-bit expert size, but cold experts in tiered mode are 2-bit (44% smaller). This caused silent skipping of cold experts.

**Fix**: Validate each chunk against its own actual size (resolved from the expert's quantization tier), not the uniform 4-bit size.

## Metal/GPU on iOS: The 4GB Wall

### The Hard Limit

Metal on iOS has a **hard 4GB per-buffer limit** that cannot be lifted by any entitlement. The `extended-virtual-addressing` and `increased-memory-limit` entitlements expand total addressable memory but do NOT increase the per-buffer cap.

| Scenario | Weight Size | Fits? | Result |
|----------|------------|-------|--------|
| 35B model (4-bit) | ~2.5 GB | Yes | Single Metal buffer, full GPU path, 5.5 tok/s |
| 397B model (4-bit) | ~5.5 GB | No | Exceeds 4096 MB Metal buffer limit |

### What We Tried to Get 397B Weights on GPU

| Approach | Result | Why It Failed |
|----------|--------|--------------|
| **Single Metal buffer (5.5GB)** | Metal assertion crash | Buffer must not exceed 4096 MB. Hard limit. |
| **Two overlapping Metal buffers** | OOM kill | Metal tracks ~8GB of shared memory on a 12GB device. iOS kills the app even in Release mode. |
| **Staging buffer (50MB memcpy per dispatch)** | Data corruption | One staging buffer shared across multiple in-flight command buffers. Metal executes all encoders AFTER commit, so earlier tensor data gets overwritten by later memcpys before GPU reads it. **Fundamental flaw** — would need N staging buffers for N concurrent command buffers. |
| **CPU fallback (no Metal buffers for weights)** | Works but 6 min/token | Weight matmuls run on CPU via Accelerate. Expert forward still uses GPU (separate small buffers). Correct output but unusable speed. |

### The Path Forward: Split Weight Files

The solution is to split `model_weights.bin` at the Python packing stage into two files, each under 4GB. Each file gets its own Metal buffer. This requires:

1. Modify `extract_weights.py` to emit `model_weights_0.bin` and `model_weights_1.bin` with a split point
2. Update `model_weights.json` manifest with file index per tensor
3. Update `infer.m` to mmap both files and select the correct base pointer per tensor
4. Re-upload 397B model to HuggingFace with split weights

This is the next major engineering task.

## K-Reduction: Running 397B on iPhone

### The Insight

Mixture-of-Experts gives a natural inference knob that dense models don't have: activate **fewer experts per token** at inference time, even if the model was trained with more. K=4 instead of K=10 means:

- **60% less I/O per token** (4 expert reads instead of 10, per layer)
- Storage unchanged (all 512 experts still on disk — routing decides which 4 to use)
- Quality degrades gracefully — you're still selecting the *best* 4 from 512 options

### K-Reduction Quality Findings

| Model | K | Output Quality | Notes |
|-------|---|---------------|-------|
| 397B (default K=10) | K=2 | **Gibberish** | "no manager" + random tokens |
| 397B (default K=10) | K=4 | **Degenerate** | "!!!!" repeated output |
| 397B (default K=10) | K=6 | Untested | Needs GPU path (split weights) to evaluate |
| 397B (default K=10) | K=8/K=10 | Untested | CPU path too slow (6 min/token) to evaluate quality |
| 35B (default K=8) | K=8 | Excellent | Full quality, production-ready |

**Conclusion**: K-reduction quality depends heavily on the model. The 397B was trained with K=10 and may need K=6+ for coherent output. Cannot properly evaluate until split weight files enable the GPU path. The 35B works perfectly at its default K=8.

### 397B on iPhone: Memory Analysis

```
Non-expert weights (mmap'd):     5.5 GB  (virtual, not all resident)
Metal buffers:                    ~500 MB (KV cache at 2048 ctx + delta-net + expert buffers)
iOS overhead:                     ~2 GB
                                  --------
Resident estimate:                ~3 GB
Available for page cache:         ~9 GB on 12GB iPhone

Expert I/O per token (K=4):      4 x 60 layers x 6.75 MB = 1.6 GB
iPhone NVMe throughput:           ~2.5 GB/s
I/O time per token:               ~0.65s
GPU compute per token:            ~0.3-0.5s (60 layers, 32 heads, head_dim=256)
                                  --------
Expected (with GPU path):        ~1-1.5 tok/s
```

With tiered experts (hot=4-bit, cold=2-bit): I/O drops ~34%, storage drops from 208GB to ~140GB, expected speed improves to ~1.5-2 tok/s.

## App Container / Bundle ID Saga

Getting models onto the device and keeping them across builds was harder than the actual porting work.

### The Problem Chain

1. Original app used `com.flashmoe.ios` bundle ID on a personal (free) developer team
2. Memory entitlements (`extended-virtual-addressing`, `increased-memory-limit`) require a **paid** developer account
3. Switched to paid team, but the original bundle ID was claimed by the personal team
4. Apple takes 24-48 hours to release a bundle ID after the old app is deleted from the personal team
5. Had to create a new bundle ID: `com.alexintosh.flashmoe`
6. This creates a NEW app container — the old app's Documents folder (with 300GB of model data) is inaccessible

### Moving 300GB Between App Containers

| Approach | Result | Why |
|----------|--------|-----|
| Files app direct access | Failed | `UIFileSharingEnabled` was set on old app but Files never showed its Documents. Unknown why. |
| `pymobiledevice3` (HouseArrest/AFC) | Can read both containers | But cannot cross-container move due to iOS sandbox. Would need download-to-Mac-then-upload round-trip. |
| `moveToService` picker in old app | **Worked** | `UIDocumentPickerViewController` with `.moveToService` lets iOS move files to a shared location (iCloud Drive, local "On My iPhone"). Then import picker in new app reads from that location. |

**Final workflow**: Old app exports via moveToService -> shared "On My iPhone" folder -> New app imports via document picker -> Files copied to new app's Documents.

### Developer Experience Issues

| Issue | Impact | Mitigation |
|-------|--------|-----------|
| Xcode Clean Build wipes app Documents | 300GB model data lost on every Clean Build | Avoid Clean Build. Use incremental. |
| Debug builds add ~2GB Metal validation overhead | OOM on iPhone for any model | Always build Release for on-device testing |
| USB developer disk image mount failures after OOM crashes | Cannot deploy until fixed | Restart iPhone + Xcode |
| "Persist container data on reinstall" unavailable on personal team | Reinstall = lose all models | Required paid developer team |
| `isExcludedFromBackup` needed on model files | Without it, iOS may purge 200GB+ of model data | Set on every downloaded file |

### File Provider Storage Gotcha

When a model is stored via Files app "Move to Service", the actual bytes end up in `/Shared/AppGroup/.../File Provider Storage/` which goes through the iOS file coordination layer. Every `pread` call to expert files goes through this layer, adding measurable latency per I/O. Solution: always import models into the app's own Documents directory rather than reading them in place from File Provider Storage.

## iOS App Architecture

### Engine Layer (C/Objective-C)

- **FlashMoEEngine.h** — C API: `create/load/generate/generate_continuation/cancel/reset/unload/destroy`
- **FlashMoEEngine.m** — Unity build wrapper (`#include "infer.m"`)
- **infer.m** — The full 7,500-line inference engine, shared with macOS
- **shaders.metal** — Metal compute kernels (~1,200 lines), compiled into `default.metallib`

The unity build means 100% of inference code is shared between iOS and macOS. No fork to maintain.

### Bridge Layer (Swift/ObjC Interop)

- **FlashMoEBridge.swift** — `@Observable` class wrapping the C API
  - `loadModel(at:)` -> background thread -> `flashmoe_load()` with adaptive memory config
  - `generate(prompt:)` -> `AsyncStream<GenerationToken>` via C callback bridge using `Unmanaged` pointer passing
  - `generateContinuation(userMessage:)` -> reuses KV cache for multi-turn, returns -2 on context full
  - State machine: `idle -> loading -> ready -> generating -> ready`
  - Thread safety: generation runs on dedicated `DispatchQueue`, callbacks bridge to MainActor for UI

### UI Layer (SwiftUI)

- **ChatView** — Streaming chat with thinking block disclosure triangle, text selection, braille spinner animation
- **ModelListView** — On-device models + downloadable HuggingFace catalog with auto K-reduction recommendations
- **ModelDownloadRow** — Per-model download progress with pause/resume/delete
- **ProfilerView** — Resource monitoring: memory usage, tok/s, thermal state (Cool/Warm/Hot/Critical)
- **ContentView** — Root navigation with Settings (K-reduction picker, I/O fanout picker)

### Services

- **DownloadManager.swift** — Background `URLSession` with state persistence, resume support, per-file progress
- **ModelCatalog.swift** — Static registry of pre-packed HuggingFace repos with recommended K values per device

### Entitlements

```xml
<key>com.apple.developer.kernel.extended-virtual-addressing</key>  <true/>
<key>com.apple.developer.kernel.increased-memory-limit</key>       <true/>
```

These expand total addressable memory (needed for mmap'ing 5.5GB weight files + expert reads) but do NOT lift the 4GB per-Metal-buffer limit.

## Key Design Decisions

| Decision | Why | Alternative Considered |
|----------|-----|----------------------|
| Unity build (`#include "infer.m"`) | Share 100% of inference code with macOS, no fork | Separate iOS codebase — rejected, maintenance nightmare |
| Runtime Metal library fallback | Single codepath for iOS (pre-compiled) and macOS (source) | Conditional compilation — more complex |
| Adaptive context cap via `os_proc_available_memory()` | Scale KV cache to available memory instead of hardcoded limit | Fixed 2048 — wastes memory on larger devices |
| K-reduction as user setting, not hardcoded | Quality impact varies by model, let user experiment | Auto-select — not enough data on quality thresholds |
| Pre-packed HuggingFace models | No on-device conversion needed, download and run | On-device repacking — too slow, too much temp storage |
| Background URLSession | Downloads survive app suspension (not force-quit) | Foreground-only — bad UX for 200GB downloads |
| Trust the OS page cache on iOS | Same philosophy as desktop, no custom expert cache | Custom LRU — slower on desktop, would be worse on iOS |
| `isExcludedFromBackup` on model files | Prevents iOS from purging 200GB+ model data | No flag — iOS may delete models to free space |
| Security-scoped bookmarks for external models | Read models from other app containers without copying | Copy-on-import — wastes storage for 200GB models |
| `moveToService` for export | iOS handles the file move atomically | Manual copy — error-prone, needs temp space |
| CPU fallback for >4GB weights | Correct (just slow) vs. crashing or corrupting data | Staging buffers — data corruption, fundamental flaw |
| No mmap for expert files on iOS | pread-only saves virtual address space | mmap — 5x slower due to per-page fault overhead on cold data (tested on desktop) |

## Pre-Packed Models on HuggingFace

| Repo | Model | Quant | Size | iPhone Min |
|------|-------|-------|------|-----------|
| `alexintosh/Qwen3.5-35B-A3B-Q4-FlashMoE` | 35B | 4-bit | 19.5 GB | 128 GB storage, 8 GB RAM |
| `alexintosh/Qwen3.5-35B-A3B-Q4-Tiered-FlashMoE` | 35B | Tiered | 13.4 GB | 128 GB storage, 8 GB RAM |
| `alexintosh/Qwen3.5-397B-A17B-Q4-FlashMoE` | 397B | 4-bit | ~214 GB | 256 GB storage, 12 GB RAM |

Each repo contains: `config.json`, `model_weights.bin`, `model_weights.json`, `tokenizer.bin`, `tokenizer.json`, `vocab.bin`, `packed_experts/layer_XX.bin`.

## Copying Models to iPhone

```bash
# Via USB (auto-detects connected device)
./copy_model_to_iphone.sh /path/to/model-directory

# Or specify device UDID
./copy_model_to_iphone.sh /path/to/model-directory <device-udid>
```

The script copies all model files to the app's Documents container with transfer speed and ETA display.

## What's Next

### Must Do (397B on iPhone)
1. **Split `model_weights.bin` into two <4GB files** at Python packing stage — each gets its own Metal buffer, enabling full GPU path
2. **Test K=6/8/10 with GPU path** — CPU fallback is too slow to evaluate quality; need GPU path to determine minimum viable K
3. **Upload split 397B model to HuggingFace** — needs the split weight tooling first

### Nice to Have
- **Adaptive K** — auto-select K based on device RAM, thermal state, and model size
- **Thermal throttling awareness** — monitor `ProcessInfo.ThermalState`, reduce K when throttling
- **Download resumption improvements** — handle interrupted downloads more gracefully
- **Background inference** — continue generation when app is backgrounded
- **Smaller models** — 7B/14B Qwen3.5 variants for devices with less storage
