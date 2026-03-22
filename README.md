# Flash-iOS

Few tweaks for [@alexintosh](https://github.com/Alexintosh) (macOS, iOS memory and Fanout)

Based on: https://github.com/Alexintosh/flash-moe/tree/feature/ios-app/FlashMoE-iOS

## Changes from upstream

- **macOS compatibility** — `#if os(iOS)` / `#if os(macOS)` guards for platform-specific APIs (UIKit colors, toolbar placement, memory queries, `os_proc_available_memory`)
- **iOS memory entitlements** — `extended-virtual-addressing` + `increased-memory-limit` for running large MoE models on iPhone
- **Fanout I/O** — Ported chunked pread from desktop engine: split each expert read into N page-aligned chunks for parallel SSD reads. Configurable via UI (Off / 2 / 4 / 8 chunks)
- **Race condition fix** — Async pread uses GCD `dispatch_group` (not pthread pool) to eliminate generation counter conflicts
- **Tiered validation fix** — `async_pread_wait` validates each chunk against its own size (not uniform 4-bit size), fixing silent cold expert skipping
- **No mmap on iOS** — Expert layer files use pread-only path, saving virtual address space
- **Model reload fix** — Full cleanup on unload (weight file, layer cache, tracking arrays, tensor hash table) so switching models doesn't crash
- **iOS storage protection** — Model files marked `isExcludedFromBackup` to prevent iOS from purging them
- **Models & Settings** — Menu button to return to model list / I/O settings from chat
- **Copy script** — `copy_model_to_iphone.sh` to push models to device over USB cable with auto-detect, ETA, and per-file progress

## Copy model to iPhone

Connect your iPhone via USB cable, then:

```bash
# Auto-detects connected device
./copy_model_to_iphone.sh /path/to/model-directory

# Or specify device UDID
./copy_model_to_iphone.sh /path/to/model-directory <device-udid>
```

The script copies all model files (config, weights, vocab, expert layers) file-by-file to the app's Documents container, with transfer speed and ETA display.
