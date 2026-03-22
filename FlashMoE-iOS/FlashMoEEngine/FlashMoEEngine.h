//
//  FlashMoEEngine.h
//  Flash-MoE inference engine — C API for iOS integration
//
//  Wraps the Metal compute pipeline for Qwen3.5 MoE models.
//  Expert weights stream from SSD via pread(); only K active experts
//  are loaded per layer. Works with 4-bit and tiered (4-bit/2-bit) quantization.
//

#ifndef FLASHMOE_ENGINE_H
#define FLASHMOE_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- Opaque engine handle ----
typedef struct FlashMoEContext FlashMoEContext;

// ---- Token callback ----
// Called for each generated token. Return 0 to continue, non-zero to stop.
typedef int (*FlashMoETokenCallback)(
    const char *token_text,     // Decoded token string (UTF-8)
    int token_id,               // Raw token ID
    int tokens_generated,       // Total tokens generated so far
    double tokens_per_second,   // Current tok/s
    void *user_data             // User context pointer
);

// ---- Configuration ----
typedef struct {
    const char *model_path;     // Path to model directory (contains config.json, packed_experts/, etc.)
    int max_context;            // Max sequence length (0 = use model default)
    int think_budget;           // Max thinking tokens (0 = unlimited)
    int use_tiered;             // 1 = use tiered quantization if available, 0 = auto-detect
    int active_experts_k;       // Override K (experts per token). 0 = use model default.
                                // Reducing K cuts I/O proportionally (e.g. K=4 on a K=10 model
                                // reduces expert reads by 60%). Quality degrades gracefully.
    int cache_io_split;         // >1 = split each expert pread into N page-aligned chunks (fanout), 0/1 = disabled
    int verbose;                // 1 = log to stderr, 0 = quiet
} FlashMoEConfig;

// ---- Engine stats ----
typedef struct {
    // Model info
    char model_name[256];
    int num_layers;
    int num_experts;
    int active_experts_k;
    int hidden_dim;
    int vocab_size;

    // Generation stats
    double tokens_per_second;
    int tokens_generated;
    double total_time_ms;
    double ttft_ms;             // Time to first token

    // Memory
    size_t weight_file_bytes;   // Non-expert weights (mmap'd)
    size_t expert_file_bytes;   // Total expert data on disk
    size_t metal_buffer_bytes;  // GPU buffer allocation
} FlashMoEStats;

// ---- Lifecycle ----

// Create engine context. Does NOT load the model yet.
FlashMoEContext *flashmoe_create(void);

// Load a model from the given config. Returns 0 on success, -1 on error.
// This allocates Metal resources, mmaps weight files, opens expert file descriptors.
int flashmoe_load(FlashMoEContext *ctx, const FlashMoEConfig *config);

// Unload the current model, releasing all resources.
void flashmoe_unload(FlashMoEContext *ctx);

// Destroy the engine context.
void flashmoe_destroy(FlashMoEContext *ctx);

// ---- Generation ----

// Generate tokens from a prompt. Blocks until generation completes or is cancelled.
// The callback is called for each token on the calling thread.
// Returns the number of tokens generated, or -1 on error.
int flashmoe_generate(
    FlashMoEContext *ctx,
    const char *prompt,
    int max_tokens,
    FlashMoETokenCallback callback,
    void *user_data
);

// Generate continuation — reuses KV cache from previous turns.
// Only processes the new user turn, skipping re-prefill of history.
// The user_content should be raw text (not formatted with chat template).
int flashmoe_generate_continuation(
    FlashMoEContext *ctx,
    const char *user_content,
    int max_tokens,
    FlashMoETokenCallback callback,
    void *user_data
);

// Cancel an in-progress generation. Safe to call from any thread.
void flashmoe_cancel(FlashMoEContext *ctx);

// Reset conversation state (KV cache, linear attention state, position).
void flashmoe_reset(FlashMoEContext *ctx);

// ---- Stats ----

// Get current engine stats. Fills the provided struct.
void flashmoe_get_stats(FlashMoEContext *ctx, FlashMoEStats *stats);

// ---- Utility ----

// Check if a model directory is valid (has config.json, packed_experts/, etc.)
// Returns 0 if valid, -1 if not.
int flashmoe_validate_model(const char *model_path);

// Get the current turn count (0 = no history, >0 = can use continuation).
int flashmoe_turn_count(FlashMoEContext *ctx);

// Get a human-readable error string for the last error.
const char *flashmoe_last_error(FlashMoEContext *ctx);

#ifdef __cplusplus
}
#endif

#endif // FLASHMOE_ENGINE_H
