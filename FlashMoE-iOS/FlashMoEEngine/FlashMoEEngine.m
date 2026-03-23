/*
 * FlashMoEEngine.m — iOS wrapper for the Flash-MoE inference engine
 *
 * Unity build: includes infer.m directly (with CHAT_MODE to suppress main()).
 * Provides the C API defined in FlashMoEEngine.h for Swift/SwiftUI integration.
 *
 * Single-instance design: iOS memory constraints mean only one model at a time.
 * The FlashMoEContext struct holds all state, wrapping infer.m's static globals.
 */

#define CHAT_MODE 1  // suppress main() in infer.m

// Unity build — include the entire inference engine
// This gives us access to all static functions and globals
#include "../../metal_infer/infer.m"

#include "FlashMoEEngine.h"
#include <stdatomic.h>
#include <os/proc.h>
#include <sys/utsname.h>
#if TARGET_OS_IOS
#import <UIKit/UIKit.h>
#endif

// ============================================================================
// FlashMoEContext — wraps engine state for the public C API
// ============================================================================

struct FlashMoEContext {
    // Lifecycle state
    int loaded;                    // 1 if a model is loaded
    atomic_int cancelled;          // 1 if generation should stop

    // Model resources (owned)
    WeightFile *wf;
    Vocabulary *vocab;
    int *layer_fds;                // [num_layers] file descriptors for expert layers
    int *layer_fds_cold_local;     // [num_layers] cold file descriptors
    void **layer_mmaps;            // [num_layers] mmap'd expert data
    size_t *layer_mmap_sizes;      // [num_layers] mmap sizes
    void **layer_states;           // [num_layers] linear attention state
    KVCache **kv_caches;           // [num_layers] KV caches for full attention
    float *hidden;                 // [hidden_dim] working buffer
    float *logits;                 // [vocab_size] logits buffer
    uint16_t *final_norm_w;        // pointer into wf (not owned)
    int K;                         // num experts per token

    // Conversation state (for KV cache reuse)
    int current_pos;               // sequence position for RoPE (persists across turns)
    int turn_count;                // 0 = fresh session, >0 = has history

    // Generation stats
    double tokens_per_second;
    int tokens_generated;
    double total_time_ms;
    double ttft_ms;

    // Error state
    char last_error[512];
};

// ============================================================================
// Shader loading for iOS — find shaders.metal in the app bundle
// ============================================================================

// Override the shader search path for iOS: look in the app bundle first
static NSString *flashmoe_find_shader_source(void) {
    NSError *error = nil;
    NSString *src = nil;

    // 1. Try app bundle (iOS deployment)
    NSString *bundlePath = [[NSBundle mainBundle] pathForResource:@"shaders" ofType:@"metal"];
    if (bundlePath) {
        src = [NSString stringWithContentsOfFile:bundlePath encoding:NSUTF8StringEncoding error:&error];
        if (src) return src;
    }

    // 2. Try relative paths (macOS development / testing)
    NSArray *paths = @[@"shaders.metal", @"metal_infer/shaders.metal"];
    for (NSString *p in paths) {
        src = [NSString stringWithContentsOfFile:p encoding:NSUTF8StringEncoding error:&error];
        if (src) return src;
    }

    return nil;
}

// ============================================================================
// tokenize_continuation_turn — local copy for iOS
// (The original is inside #ifndef CHAT_MODE in infer.m, excluded by our #define)
// ============================================================================

static PromptTokens *flashmoe_tokenize_continuation_turn(const char *user_content) {
    const char *prefix = "\n<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// ============================================================================
// Public API Implementation
// ============================================================================

FlashMoEContext *flashmoe_create(void) {
    FlashMoEContext *ctx = calloc(1, sizeof(FlashMoEContext));
    if (!ctx) return NULL;
    ctx->loaded = 0;
    atomic_store(&ctx->cancelled, 0);
    ctx->last_error[0] = '\0';
    return ctx;
}

int flashmoe_load(FlashMoEContext *ctx, const FlashMoEConfig *config) {
    if (!ctx || !config || !config->model_path) {
        if (ctx) snprintf(ctx->last_error, sizeof(ctx->last_error), "Invalid arguments");
        return -1;
    }

    // Unload any previously loaded model
    if (ctx->loaded) {
        flashmoe_unload(ctx);
    }

    @autoreleasepool {
        const char *model_path = config->model_path;

        // ---- Load model configuration ----
        config_init_defaults();

        // Set model path for tokenizer lookup (strdup: Swift string may be temporary)
        if (g_model_path_for_tokenizer) free((void *)g_model_path_for_tokenizer);
        g_model_path_for_tokenizer = strdup(model_path);

        // Build manifest path for config loading
        char manifest_path_buf[1024];
        snprintf(manifest_path_buf, sizeof(manifest_path_buf), "%s/model_weights.json", model_path);

        load_config_from_config_json(model_path);
        if (access(manifest_path_buf, R_OK) == 0) {
            load_config_from_manifest(manifest_path_buf);
        }

        // Note: MAX_SEQ_LEN is a compile-time constant in infer.m.
        // KV caches are allocated at MAX_SEQ_LEN. On iOS, context is
        // effectively limited by available memory and max_tokens passed
        // to flashmoe_generate(). No runtime capping needed here.
        // Suppress debug output for iOS
        g_stream_mode = 1;

        if (config->think_budget > 0) {
            g_think_budget = config->think_budget;
        }

        // Set quantization mode
        g_use_tiered = config->use_tiered;
        g_use_2bit = config->use_2bit;

        // Set cache I/O split (fanout mode): >1 = split expert preads into N chunks
        if (config->cache_io_split > 1) {
            g_cache_io_split = config->cache_io_split;
        } else {
            g_cache_io_split = 1;  // disabled by default
        }

        // KV cache sizing — allocate only what we need
        {
            int default_ctx = 8192;
#if TARGET_OS_IOS
            if (![[NSProcessInfo processInfo] isMacCatalystApp]) {
                default_ctx = 2048;  // iPhone: conserve memory
            }
#endif
            int ctx_limit = (config->max_context > 0) ? config->max_context : default_ctx;
            if (ctx_limit > MAX_SEQ_LEN) ctx_limit = MAX_SEQ_LEN;
            g_kv_seq_len = ctx_limit;
            size_t kv_per_cache = (size_t)ctx_limit * g_cfg.num_kv_heads * g_cfg.head_dim * sizeof(float);
            NSLog(@"[FlashMoE] KV cache: %d positions (%.1f MB per cache x %d layers)",
                  ctx_limit, kv_per_cache / 1e6, g_cfg.num_full_attn_layers);
        }

        // K = experts per token from config (capped to MAX_K)
        ctx->K = g_cfg.num_experts_per_tok;
        if (ctx->K > MAX_K) ctx->K = MAX_K;

        // ---- Build file paths ----
        char weights_path[1024], manifest_path[1024], vocab_path[1024];

        // On iOS, weight files are in the model directory
        snprintf(weights_path, sizeof(weights_path), "%s/model_weights.bin", model_path);
        snprintf(manifest_path, sizeof(manifest_path), "%s/model_weights.json", model_path);

        // Vocab/tokenizer: try model dir first, then app bundle
        snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.bin", model_path);
        if (access(vocab_path, R_OK) != 0) {
            // Try app bundle
            NSString *bundleVocab = [[NSBundle mainBundle] pathForResource:@"vocab" ofType:@"bin"];
            if (bundleVocab) {
                strlcpy(vocab_path, [bundleVocab UTF8String], sizeof(vocab_path));
            }
        }

        // ---- Initialize Metal ----
        g_metal = metal_setup();
        if (!g_metal) {
            snprintf(ctx->last_error, sizeof(ctx->last_error), "Metal initialization failed");
            return -1;
        }

        // ---- Initialize I/O thread pool ----
        io_pool_init();

        // ---- Load weights ----
        ctx->wf = open_weights(weights_path, manifest_path);
        if (!ctx->wf) {
            snprintf(ctx->last_error, sizeof(ctx->last_error), "Failed to load weights from %s", weights_path);
            return -1;
        }

        // Wrap weight file for Metal GPU access
        metal_set_weights(g_metal, ctx->wf->data, ctx->wf->size);

        // ---- Load vocabulary ----
        ctx->vocab = load_vocab(vocab_path);
        if (!ctx->vocab) {
            snprintf(ctx->last_error, sizeof(ctx->last_error), "Failed to load vocabulary from %s", vocab_path);
            return -1;
        }

        // ---- Initialize tokenizer ----
        init_tokenizer();

        // ---- Auto-detect/load tiered manifest ----
        if (!g_use_2bit && !g_use_tiered) {
            char probe[1024];
            snprintf(probe, sizeof(probe), "%s/packed_experts_tiered/tiered_manifest.json", model_path);
            if (access(probe, F_OK) == 0) {
                if (load_tiered_manifest(model_path)) {
                    g_use_tiered = 1;
                }
            }
        }
        if (g_use_tiered && !g_tiered_manifest) {
            if (!load_tiered_manifest(model_path)) {
                snprintf(ctx->last_error, sizeof(ctx->last_error),
                         "Tiered mode requested but no manifest found");
                return -1;
            }
        }

        // ---- Open packed expert files ----
        ctx->layer_fds = calloc(g_cfg.num_layers, sizeof(int));
        ctx->layer_fds_cold_local = calloc(g_cfg.num_layers, sizeof(int));
        ctx->layer_mmaps = calloc(g_cfg.num_layers, sizeof(void *));
        ctx->layer_mmap_sizes = calloc(g_cfg.num_layers, sizeof(size_t));

        memset(g_expert_seen, 0, sizeof(g_expert_seen));
        // Initialize per-layer quant arrays to match the global mode
        // (CLI main() does this during fd open; iOS must do it explicitly)
        memset(g_layer_is_2bit, 0, sizeof(g_layer_is_2bit));
        memset(g_layer_is_q3_hybrid, 0, sizeof(g_layer_is_q3_hybrid));
        memset(g_layer_is_q3_outlier, 0, sizeof(g_layer_is_q3_outlier));

        for (int i = 0; i < g_cfg.num_layers; i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s/layer_%02d.bin", model_path,
                     g_use_tiered ? "packed_experts_tiered" :
                     g_use_2bit ? "packed_experts_2bit" : "packed_experts", i);
            ctx->layer_fds[i] = open(path, O_RDONLY);
            // Set per-layer quant flag so fused_layer_forward uses correct expert size
            if (ctx->layer_fds[i] >= 0) {
                if (g_use_2bit) g_layer_is_2bit[i] = 1;
            }
            ctx->layer_fds_cold_local[i] = -1;
            ctx->layer_mmaps[i] = MAP_FAILED;
            ctx->layer_mmap_sizes[i] = 0;
            if (ctx->layer_fds[i] >= 0) {
                fcntl(ctx->layer_fds[i], F_RDAHEAD, 0);
                struct stat st;
                if (fstat(ctx->layer_fds[i], &st) == 0 && st.st_size > 0) {
                    ctx->layer_mmap_sizes[i] = st.st_size;
                    // Skip mmap on real iOS devices — 60 × 1.9 GB = 112 GB
                    // of mmap'd expert data causes jetsam kills.
                    // macOS (including "Designed for iPad") has plenty of address space.
                    int is_real_ios = 0;
#if TARGET_OS_IOS
                    // Runtime check: ProcessInfo.processInfo.isMacCatalystApp is false
                    // on real iOS devices but true on Mac running iPad app
                    is_real_ios = ![[NSProcessInfo processInfo] isMacCatalystApp];
#endif
                    if (!is_real_ios) {
                        ctx->layer_mmaps[i] = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE,
                                                    ctx->layer_fds[i], 0);
                        if (ctx->layer_mmaps[i] == MAP_FAILED) {
                            ctx->layer_mmaps[i] = MAP_FAILED;
                        }
                    }
                }
            }
        }

        // Log expert I/O mode
        {
            int mmap_count = 0;
            for (int i = 0; i < g_cfg.num_layers; i++) {
                if (ctx->layer_mmaps[i] != MAP_FAILED) mmap_count++;
            }
            NSLog(@"[experts] %d/%d layers opened, %d mmap'd, %d pread-only",
                  g_cfg.num_layers, g_cfg.num_layers, mmap_count, g_cfg.num_layers - mmap_count);
        }

        // Wire up global cold fds
        g_layer_fds_cold = ctx->layer_fds_cold_local;

        // ---- Deferred expert state ----
        // g_deferred.h_mid is now a static array [MAX_HIDDEN_DIM] — no allocation needed
        memset(g_deferred.h_mid, 0, sizeof(g_deferred.h_mid));

        // ---- Allocate per-layer state ----
        ctx->layer_states = calloc(g_cfg.num_layers, sizeof(void *));
        ctx->kv_caches = calloc(g_cfg.num_layers, sizeof(KVCache *));

        for (int i = 0; i < g_cfg.num_layers; i++) {
            if (((i + 1) % FULL_ATTN_INTERVAL == 0)) {
                ctx->kv_caches[i] = kv_cache_new();
                if (!ctx->kv_caches[i] || !ctx->kv_caches[i]->k_cache || !ctx->kv_caches[i]->v_cache) {
                    snprintf(ctx->last_error, sizeof(ctx->last_error),
                             "KV cache alloc failed at layer %d (seq=%d, need %.0f MB per cache). "
                             "Try reducing max context or free device memory.",
                             i, g_kv_seq_len,
                             (double)g_kv_seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(float) / 1e6);
                    NSLog(@"[FlashMoE] %s", ctx->last_error);
                    return -1;
                }
            } else {
                ctx->layer_states[i] = linear_attn_state_new();
            }
        }

        // ---- Allocate working buffers ----
        ctx->hidden = calloc(HIDDEN_DIM, sizeof(float));
        ctx->logits = calloc(VOCAB_SIZE, sizeof(float));
        ctx->final_norm_w = get_tensor_ptr(ctx->wf, "model.norm.weight");

        // ---- Build layer cache (precomputes weight pointers) ----
        build_layer_cache(ctx->wf);

        ctx->loaded = 1;
        if (config->verbose) {
            NSLog(@"[FlashMoE] Model loaded: %d layers, %d experts (K=%d), hidden=%d",
                  g_cfg.num_layers, g_cfg.num_experts, ctx->K, HIDDEN_DIM);
        }

        return 0;
    }
}

void flashmoe_unload(FlashMoEContext *ctx) {
    if (!ctx || !ctx->loaded) return;

    @autoreleasepool {
        // Wait for any in-flight GPU work
        if (g_deferred.active) {
            [g_deferred.cmd_experts waitUntilCompleted];
            g_deferred.active = 0;
            g_deferred.cmd_experts = nil;
        }

        // Reset async pread state
        g_async_pread.active = 0;

        // Shutdown I/O pool
        io_pool_shutdown();

        // Close expert files
        if (ctx->layer_fds) {
            for (int i = 0; i < g_cfg.num_layers; i++) {
                if (ctx->layer_mmaps && ctx->layer_mmaps[i] != MAP_FAILED)
                    munmap(ctx->layer_mmaps[i], ctx->layer_mmap_sizes[i]);
                if (ctx->layer_fds[i] >= 0)
                    close(ctx->layer_fds[i]);
                if (ctx->layer_fds_cold_local && ctx->layer_fds_cold_local[i] >= 0)
                    close(ctx->layer_fds_cold_local[i]);
            }
            free(ctx->layer_fds); ctx->layer_fds = NULL;
            free(ctx->layer_fds_cold_local); ctx->layer_fds_cold_local = NULL;
            free(ctx->layer_mmaps); ctx->layer_mmaps = NULL;
            free(ctx->layer_mmap_sizes); ctx->layer_mmap_sizes = NULL;
        }

        // Free per-layer state
        if (ctx->layer_states) {
            for (int i = 0; i < g_cfg.num_layers; i++) {
                if (ctx->kv_caches && ctx->kv_caches[i])
                    kv_cache_free(ctx->kv_caches[i]);
                if (ctx->layer_states[i])
                    linear_attn_state_free(ctx->layer_states[i]);
            }
            free(ctx->layer_states); ctx->layer_states = NULL;
            free(ctx->kv_caches); ctx->kv_caches = NULL;
        }

        // Free working buffers
        free(ctx->hidden); ctx->hidden = NULL;
        free(ctx->logits); ctx->logits = NULL;

        // Reset deferred state (h_mid is now a static array, no free needed)
        memset(g_deferred.h_mid, 0, sizeof(g_deferred.h_mid));

        // Free weight file (munmap + manifest)
        if (ctx->wf) {
            if (ctx->wf->data) munmap(ctx->wf->data, ctx->wf->size);
            if (ctx->wf->manifest) {
                free(ctx->wf->manifest->tensors);
                free(ctx->wf->manifest);
            }
            free(ctx->wf);
            ctx->wf = NULL;
        }
        ctx->final_norm_w = NULL;

        // Reset tensor hash table (points into freed manifest)
        memset(tensor_ht, 0, sizeof(tensor_ht));
        tensor_ht_built = 0;

        // Free vocabulary
        if (ctx->vocab) {
            free(ctx->vocab);
            ctx->vocab = NULL;
        }

        // Reset static tracking arrays (no longer dynamically allocated)
        memset(g_expert_freq, 0, sizeof(g_expert_freq));
        memset(g_expert_seen, 0, sizeof(g_expert_seen));
        memset(g_cache_seen, 0, sizeof(g_cache_seen));
        memset(g_cache_last_touch_token, 0, sizeof(g_cache_last_touch_token));
        memset(g_cache_last_evict_token, 0, sizeof(g_cache_last_evict_token));
        memset(g_pred_experts, 0, sizeof(g_pred_experts));
        memset(g_pred_count, 0, sizeof(g_pred_count));

        // Reset layer cache so it rebuilds on next load
        memset(layer_cache, 0, sizeof(layer_cache));
        layer_cache_built = 0;

        // Free tiered manifest
        if (g_tiered_manifest) {
            free(g_tiered_manifest);
            g_tiered_manifest = NULL;
            g_use_tiered = 0;
        }

        // Reset KV cache limit
        g_kv_seq_len = MAX_SEQ_LEN;

        // Reset prediction state
        g_pred_enabled = 0;
        g_pred_generating = 0;
        g_pred_valid = 0;
        g_pred_hits = 0;
        g_pred_misses = 0;
        g_pred_layers = 0;

        // Reset global flags for clean reload
        g_freq_tracking = 0;
        g_cache_telemetry_enabled = 0;

        // Release Metal context
        // MetalCtx is calloc'd but holds ARC __strong id<> objects.
        // free() alone does NOT trigger ARC release — we must nil each
        // id<> member so ARC decrements refcounts before we free the struct.
        if (g_metal) {
            // Core objects
            g_metal->device = nil;
            g_metal->queue = nil;
            g_metal->library = nil;
            // Pipeline states
            g_metal->matvec_v3 = nil;
            g_metal->matvec_v5 = nil;
            g_metal->matvec_fast = nil;
            g_metal->matvec_2bit = nil;
            g_metal->matvec_iq3_xxs = nil;
            g_metal->matvec_iq4_xs = nil;
            g_metal->matvec_q5_k = nil;
            g_metal->matvec_q8_0 = nil;
            g_metal->matvec_q6_k = nil;
            // NAX
            g_metal->nax_library = nil;
            g_metal->nax_dequant = nil;
            g_metal->nax_f32_to_half = nil;
            g_metal->nax_gemm = nil;
            g_metal->nax_extract = nil;
            g_metal->nax_w_half = nil;
            g_metal->nax_x_half = nil;
            g_metal->nax_c_buf = nil;
            // Norm/activation pipelines
            g_metal->rms_norm_sum = nil;
            g_metal->rms_norm_apply = nil;
            g_metal->rms_norm_apply_bf16 = nil;
            g_metal->residual_add = nil;
            g_metal->swiglu = nil;
            // GPU attention pipelines
            g_metal->attn_scores_pipe = nil;
            g_metal->attn_softmax_pipe = nil;
            g_metal->attn_values_pipe = nil;
            g_metal->sigmoid_gate_pipe = nil;
            // MoE combine
            g_metal->moe_combine_residual = nil;
            // Delta-net pipelines
            g_metal->delta_net_step = nil;
            g_metal->conv1d_step = nil;
            g_metal->rms_norm_qk = nil;
            g_metal->compute_decay_beta = nil;
            g_metal->gated_rms_norm = nil;
            // Reusable buffers
            g_metal->buf_input = nil;
            g_metal->buf_output = nil;
            g_metal->wf_buf = nil;
            g_metal->gguf_qkv_buf = nil;
            g_metal->gguf_full_attn_buf = nil;
            g_metal->gguf_linear_buf = nil;
            g_metal->gguf_shared_buf = nil;
            g_metal->gguf_lm_head_buf = nil;
            for (int i = 0; i < MAX_BATCH_SLOTS; i++)
                g_metal->batch_out[i] = nil;
            // Legacy single-expert buffers
            g_metal->buf_expert_data = nil;
            g_metal->buf_expert_input = nil;
            g_metal->buf_expert_gate = nil;
            g_metal->buf_expert_up = nil;
            g_metal->buf_expert_act = nil;
            g_metal->buf_expert_out = nil;
            // Multi-expert buffers
            g_metal->buf_multi_expert_input = nil;
            for (int k = 0; k < MAX_K; k++) {
                g_metal->buf_multi_expert_data[k] = nil;
                g_metal->buf_multi_expert_data_B[k] = nil;
                g_metal->buf_multi_expert_gate[k] = nil;
                g_metal->buf_multi_expert_up[k] = nil;
                g_metal->buf_multi_expert_act[k] = nil;
                g_metal->buf_multi_expert_out[k] = nil;
            }
            // Shared expert buffers
            g_metal->buf_shared_gate = nil;
            g_metal->buf_shared_up = nil;
            g_metal->buf_shared_act = nil;
            g_metal->buf_shared_out = nil;
            // Fused o_proj+norm+routing buffers
            g_metal->buf_residual = nil;
            g_metal->buf_h_mid = nil;
            g_metal->buf_sum_sq = nil;
            // GPU attention buffers
            for (int i = 0; i < 16; i++) {
                g_metal->buf_kv_k[i] = nil;
                g_metal->buf_kv_v[i] = nil;
            }
            g_metal->buf_attn_q = nil;
            g_metal->buf_attn_scores = nil;
            g_metal->buf_attn_out = nil;
            g_metal->buf_attn_gate = nil;
            // CMD3 combine buffers
            g_metal->buf_moe_hidden = nil;
            g_metal->buf_combine_params = nil;
            g_metal->buf_cmd3_sum_sq = nil;
            // Shared event
            g_metal->pipeline_event = nil;
            // Delta-net GPU state buffers
            for (int i = 0; i < 48; i++) {
                g_metal->buf_delta_state[i] = nil;
                g_metal->buf_conv_state[i] = nil;
            }
            // Delta-net scratch buffers
            g_metal->buf_delta_q = nil;
            g_metal->buf_delta_k = nil;
            g_metal->buf_delta_v = nil;
            g_metal->buf_delta_g_decay = nil;
            g_metal->buf_delta_beta = nil;
            g_metal->buf_delta_output = nil;
            g_metal->buf_conv_input = nil;
            g_metal->buf_conv_output = nil;
            free(g_metal);
            g_metal = NULL;
        }

        ctx->loaded = 0;
    }
}

void flashmoe_destroy(FlashMoEContext *ctx) {
    if (!ctx) return;
    flashmoe_unload(ctx);
    free(ctx);
}

// ============================================================================
// Generation — the core inference loop adapted for callback-based streaming
// ============================================================================

int flashmoe_generate(
    FlashMoEContext *ctx,
    const char *prompt,
    int max_tokens,
    FlashMoETokenCallback callback,
    void *user_data
) {
    if (!ctx || !ctx->loaded || !prompt) {
        if (ctx) snprintf(ctx->last_error, sizeof(ctx->last_error), "Engine not loaded or invalid arguments");
        return -1;
    }

    @autoreleasepool {
        atomic_store(&ctx->cancelled, 0);
        ctx->tokens_generated = 0;
        ctx->tokens_per_second = 0;

        double t0 = now_ms();

        // ---- Tokenize prompt ----
        PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
        if (!pt) {
            snprintf(ctx->last_error, sizeof(ctx->last_error), "Failed to tokenize prompt");
            return -1;
        }

        int K = ctx->K;

        // ---- Reset state for new generation ----
        reset_delta_net_state();
        // Reset KV cache lengths
        for (int i = 0; i < g_cfg.num_layers; i++) {
            if (ctx->kv_caches[i]) {
                ctx->kv_caches[i]->len = 0;
            }
        }

        int pos = 0;

        // ---- Batch prefill: embed all prompt tokens ----
        float *embed_batch = NULL;
        if (pt->count > 1) {
            embed_batch = malloc((size_t)pt->count * HIDDEN_DIM * sizeof(float));
            for (int i = 0; i < pt->count; i++) {
                embed_lookup(ctx->wf, pt->ids[i], embed_batch + (size_t)i * HIDDEN_DIM);
            }
        }

        // ---- Prefill intermediate tokens (discard expert output) ----
        if (pt->count > 1) {
            double prefill_start = now_ms();
            for (int token_idx = 0; token_idx < pt->count - 1; token_idx++) {
                if (atomic_load(&ctx->cancelled)) {
                    free(embed_batch);
                    free(pt->ids); free(pt);
                    return ctx->tokens_generated;
                }

                @autoreleasepool {
                memcpy(ctx->hidden, embed_batch + (size_t)token_idx * HIDDEN_DIM,
                       HIDDEN_DIM * sizeof(float));

                for (int layer = 0; layer < g_cfg.num_layers; layer++) {
                    int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                    fused_layer_forward(ctx->wf, layer, ctx->hidden,
                                        is_full ? ctx->kv_caches[layer] : NULL,
                                        is_full ? NULL : ctx->layer_states[layer],
                                        pos,
                                        ctx->layer_mmaps[layer] != MAP_FAILED ? ctx->layer_mmaps[layer] : NULL,
                                        K, ctx->layer_fds[layer]);
                }
                discard_deferred_experts();
                pos++;
                } // @autoreleasepool — drain Metal objects per prefill token

                // Report prefill progress via callback
                double prefill_elapsed = now_ms() - prefill_start;
                double prefill_tps = prefill_elapsed > 0 ? (token_idx + 1) * 1000.0 / prefill_elapsed : 0;
                ctx->tokens_per_second = prefill_tps;
                ctx->tokens_generated = -(token_idx + 1);  // negative = prefill in progress
                if (callback) {
                    char prefill_status[64];
                    snprintf(prefill_status, sizeof(prefill_status),
                             "[prefill %d/%d]", token_idx + 1, pt->count - 1);
                    callback(prefill_status, -1, -(token_idx + 1), prefill_tps, user_data);
                }
            }
            double prefill_total = now_ms() - prefill_start;
            NSLog(@"[prefill] %d tokens in %.0f ms (%.1f tok/s)",
                  pt->count - 1, prefill_total,
                  prefill_total > 0 ? (pt->count - 1) * 1000.0 / prefill_total : 0);
        }

        // ---- Last prefill token (need full hidden state) ----
        {
            if (embed_batch) {
                memcpy(ctx->hidden, embed_batch + (size_t)(pt->count - 1) * HIDDEN_DIM,
                       HIDDEN_DIM * sizeof(float));
            } else {
                embed_lookup(ctx->wf, pt->ids[0], ctx->hidden);
            }

            for (int layer = 0; layer < g_cfg.num_layers; layer++) {
                int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                fused_layer_forward(ctx->wf, layer, ctx->hidden,
                                    is_full ? ctx->kv_caches[layer] : NULL,
                                    is_full ? NULL : ctx->layer_states[layer],
                                    pos,
                                    ctx->layer_mmaps[layer] != MAP_FAILED ? ctx->layer_mmaps[layer] : NULL,
                                    K, ctx->layer_fds[layer]);
            }
            complete_deferred_experts();
            pos++;
        }

        if (embed_batch) { free(embed_batch); embed_batch = NULL; }

        // ---- Final norm + LM head + sample first token ----
        if (ctx->final_norm_w) {
            cpu_rms_norm(ctx->hidden, ctx->final_norm_w, ctx->hidden, HIDDEN_DIM, RMS_NORM_EPS);
        }

        lm_head_forward(ctx->wf, ctx->hidden, ctx->logits);
        int next_token = cpu_argmax(ctx->logits, VOCAB_SIZE);

        ctx->ttft_ms = now_ms() - t0;
        ctx->tokens_generated = 1;

        // ---- Invoke callback for first token ----
        const char *token_text = decode_token(ctx->vocab, next_token);
        NSLog(@"[gen] token %d: id=%d text=\"%s\"", ctx->tokens_generated, next_token,
              token_text ? token_text : "(null)");
        if (callback) {
            double gen_time = now_ms() - t0 - ctx->ttft_ms;
            double tps = gen_time > 0 ? 1000.0 / gen_time : 0;
            int stop = callback(token_text, next_token, ctx->tokens_generated, tps, user_data);
            if (stop) {
                free(pt->ids); free(pt);
                ctx->total_time_ms = now_ms() - t0;
                return ctx->tokens_generated;
            }
        }

        int in_think = (next_token == THINK_START_TOKEN) ? 1 : 0;
        int think_tokens = 0;

        // ---- Auto-regressive generation loop ----
        double gen_start = now_ms();

        for (int gen = 1; gen < max_tokens; gen++) {
            // Check cancellation
            if (atomic_load(&ctx->cancelled)) break;

            // Check EOS
            if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) {
                NSLog(@"[gen] EOS token %d at position %d — stopping", next_token, ctx->tokens_generated);
                break;
            }

            // Think budget enforcement
            if (next_token == THINK_START_TOKEN) in_think = 1;
            if (next_token == THINK_END_TOKEN) in_think = 0;
            if (in_think) think_tokens++;

            // Embed + forward pass (autoreleasepool drains Metal command buffers each token)
            @autoreleasepool {
            embed_lookup(ctx->wf, next_token, ctx->hidden);

            for (int layer = 0; layer < g_cfg.num_layers; layer++) {
                int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                fused_layer_forward(ctx->wf, layer, ctx->hidden,
                                    is_full ? ctx->kv_caches[layer] : NULL,
                                    is_full ? NULL : ctx->layer_states[layer],
                                    pos,
                                    ctx->layer_mmaps[layer] != MAP_FAILED ? ctx->layer_mmaps[layer] : NULL,
                                    K, ctx->layer_fds[layer]);
            }
            complete_deferred_experts();
            pos++;

            // Final norm + LM head
            double t_lm = 0;
            if (g_timing_enabled) t_lm = now_ms();

            if (ctx->final_norm_w) {
                    cpu_rms_norm(ctx->hidden, ctx->final_norm_w, ctx->hidden, HIDDEN_DIM, RMS_NORM_EPS);
            }

            lm_head_forward(ctx->wf, ctx->hidden, ctx->logits);
            next_token = cpu_argmax(ctx->logits, VOCAB_SIZE);

            if (g_timing_enabled) {
                g_timing.lm_head += now_ms() - t_lm;
                g_timing.token_count++;
            }
            } // @autoreleasepool — drains Metal command buffers

            // Think budget: force end thinking
            if (in_think && g_think_budget > 0 && think_tokens >= g_think_budget) {
                next_token = THINK_END_TOKEN;
                in_think = 0;
            }

            ctx->tokens_generated++;

            // Compute tok/s
            double elapsed_gen = now_ms() - gen_start;
            ctx->tokens_per_second = elapsed_gen > 0 ? (ctx->tokens_generated - 1) * 1000.0 / elapsed_gen : 0;

            // Invoke callback
            token_text = decode_token(ctx->vocab, next_token);
            NSLog(@"[gen] token %d: id=%d text=\"%s\" (%.1f tok/s)",
                  ctx->tokens_generated, next_token,
                  token_text ? token_text : "(null)",
                  ctx->tokens_per_second);
            if (callback) {
                int stop = callback(token_text, next_token, ctx->tokens_generated,
                                    ctx->tokens_per_second, user_data);
                if (stop) break;
            }
        }

        ctx->total_time_ms = now_ms() - t0;
        double gen_elapsed = now_ms() - gen_start;
        if (ctx->tokens_generated > 1 && gen_elapsed > 0) {
            ctx->tokens_per_second = (ctx->tokens_generated - 1) * 1000.0 / gen_elapsed;
        }

        // Persist state for KV cache reuse in next turn
        ctx->current_pos = pos;
        ctx->turn_count++;

        free(pt->ids);
        free(pt);

        return ctx->tokens_generated;
    }
}

// ============================================================================
// Continuation generation — reuses KV cache from previous turns
// ============================================================================

int flashmoe_generate_continuation(
    FlashMoEContext *ctx,
    const char *user_content,
    int max_tokens,
    FlashMoETokenCallback callback,
    void *user_data
) {
    if (!ctx || !ctx->loaded || !user_content) {
        if (ctx) snprintf(ctx->last_error, sizeof(ctx->last_error), "Engine not loaded or invalid arguments");
        return -1;
    }
    if (ctx->turn_count == 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error), "No previous turn — use flashmoe_generate first");
        return -1;
    }

    @autoreleasepool {
        atomic_store(&ctx->cancelled, 0);
        ctx->tokens_generated = 0;
        ctx->tokens_per_second = 0;

        double t0 = now_ms();

        // Tokenize only the new turn (with continuation markers)
        PromptTokens *pt = flashmoe_tokenize_continuation_turn(user_content);
        if (!pt) {
            snprintf(ctx->last_error, sizeof(ctx->last_error), "Failed to tokenize continuation turn");
            return -1;
        }

        int K = ctx->K;
        int pos = ctx->current_pos;  // Resume from where we left off

        // Check we have room in the KV cache
        if (pos + pt->count + max_tokens > MAX_SEQ_LEN) {
            NSLog(@"[FlashMoE] Context full (%d + %d + %d > %d), resetting to fresh generation",
                  pos, pt->count, max_tokens, MAX_SEQ_LEN);
            free(pt->ids); free(pt);
            // Fall back to full generation with chat template
            // Caller should handle this by using flashmoe_generate instead
            snprintf(ctx->last_error, sizeof(ctx->last_error), "Context window full, reset required");
            return -2;  // Signal to caller: context full, need reset
        }

        // NOTE: No reset_delta_net_state() — reuse KV caches and linear attention state

        // ---- Prefill continuation tokens ----
        float *embed_batch = NULL;
        if (pt->count > 1) {
            embed_batch = malloc((size_t)pt->count * HIDDEN_DIM * sizeof(float));
            for (int i = 0; i < pt->count; i++) {
                embed_lookup(ctx->wf, pt->ids[i], embed_batch + (size_t)i * HIDDEN_DIM);
            }
        }

        if (pt->count > 1) {
            for (int token_idx = 0; token_idx < pt->count - 1; token_idx++) {
                if (atomic_load(&ctx->cancelled)) {
                    free(embed_batch);
                    free(pt->ids); free(pt);
                    return ctx->tokens_generated;
                }

                @autoreleasepool {
                memcpy(ctx->hidden, embed_batch + (size_t)token_idx * HIDDEN_DIM,
                       HIDDEN_DIM * sizeof(float));

                for (int layer = 0; layer < g_cfg.num_layers; layer++) {
                    int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                    fused_layer_forward(ctx->wf, layer, ctx->hidden,
                                        is_full ? ctx->kv_caches[layer] : NULL,
                                        is_full ? NULL : ctx->layer_states[layer],
                                        pos,
                                        ctx->layer_mmaps[layer] != MAP_FAILED ? ctx->layer_mmaps[layer] : NULL,
                                        K, ctx->layer_fds[layer]);
                }
                discard_deferred_experts();
                pos++;
                } // @autoreleasepool
            }
        }

        // Last prefill token
        {
            if (embed_batch) {
                memcpy(ctx->hidden, embed_batch + (size_t)(pt->count - 1) * HIDDEN_DIM,
                       HIDDEN_DIM * sizeof(float));
            } else {
                embed_lookup(ctx->wf, pt->ids[0], ctx->hidden);
            }

            for (int layer = 0; layer < g_cfg.num_layers; layer++) {
                int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                fused_layer_forward(ctx->wf, layer, ctx->hidden,
                                    is_full ? ctx->kv_caches[layer] : NULL,
                                    is_full ? NULL : ctx->layer_states[layer],
                                    pos,
                                    ctx->layer_mmaps[layer] != MAP_FAILED ? ctx->layer_mmaps[layer] : NULL,
                                    K, ctx->layer_fds[layer]);
            }
            complete_deferred_experts();
            pos++;
        }

        if (embed_batch) { free(embed_batch); embed_batch = NULL; }

        // ---- Final norm + LM head + sample first token ----
        if (ctx->final_norm_w) {
            cpu_rms_norm(ctx->hidden, ctx->final_norm_w, ctx->hidden, HIDDEN_DIM, RMS_NORM_EPS);
        }

        lm_head_forward(ctx->wf, ctx->hidden, ctx->logits);
        int next_token = cpu_argmax(ctx->logits, VOCAB_SIZE);

        ctx->ttft_ms = now_ms() - t0;
        ctx->tokens_generated = 1;

        const char *token_text = decode_token(ctx->vocab, next_token);
        if (callback) {
            double gen_time = now_ms() - t0 - ctx->ttft_ms;
            double tps = gen_time > 0 ? 1000.0 / gen_time : 0;
            int stop = callback(token_text, next_token, ctx->tokens_generated, tps, user_data);
            if (stop) {
                free(pt->ids); free(pt);
                ctx->current_pos = pos;
                ctx->total_time_ms = now_ms() - t0;
                return ctx->tokens_generated;
            }
        }

        int in_think = (next_token == THINK_START_TOKEN) ? 1 : 0;
        int think_tokens = 0;

        // ---- Auto-regressive generation loop ----
        double gen_start = now_ms();

        for (int gen = 1; gen < max_tokens; gen++) {
            if (atomic_load(&ctx->cancelled)) break;

            if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) break;

            if (next_token == THINK_START_TOKEN) in_think = 1;
            if (next_token == THINK_END_TOKEN) in_think = 0;
            if (in_think) think_tokens++;

            @autoreleasepool {
            embed_lookup(ctx->wf, next_token, ctx->hidden);

            for (int layer = 0; layer < g_cfg.num_layers; layer++) {
                int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                fused_layer_forward(ctx->wf, layer, ctx->hidden,
                                    is_full ? ctx->kv_caches[layer] : NULL,
                                    is_full ? NULL : ctx->layer_states[layer],
                                    pos,
                                    ctx->layer_mmaps[layer] != MAP_FAILED ? ctx->layer_mmaps[layer] : NULL,
                                    K, ctx->layer_fds[layer]);
            }
            complete_deferred_experts();
            pos++;

            double t_lm2 = 0;
            if (g_timing_enabled) t_lm2 = now_ms();

            if (ctx->final_norm_w) {
                    cpu_rms_norm(ctx->hidden, ctx->final_norm_w, ctx->hidden, HIDDEN_DIM, RMS_NORM_EPS);
            }

            lm_head_forward(ctx->wf, ctx->hidden, ctx->logits);
            next_token = cpu_argmax(ctx->logits, VOCAB_SIZE);

            if (g_timing_enabled) {
                g_timing.lm_head += now_ms() - t_lm2;
                g_timing.token_count++;
            }
            } // @autoreleasepool

            if (in_think && g_think_budget > 0 && think_tokens >= g_think_budget) {
                next_token = THINK_END_TOKEN;
                in_think = 0;
            }

            ctx->tokens_generated++;
            double elapsed_gen = now_ms() - gen_start;
            ctx->tokens_per_second = elapsed_gen > 0 ? (ctx->tokens_generated - 1) * 1000.0 / elapsed_gen : 0;

            token_text = decode_token(ctx->vocab, next_token);
            if (callback) {
                int stop = callback(token_text, next_token, ctx->tokens_generated,
                                    ctx->tokens_per_second, user_data);
                if (stop) break;
            }
        }

        ctx->total_time_ms = now_ms() - t0;
        double gen_elapsed = now_ms() - gen_start;
        if (ctx->tokens_generated > 1 && gen_elapsed > 0) {
            ctx->tokens_per_second = (ctx->tokens_generated - 1) * 1000.0 / gen_elapsed;
        }

        ctx->current_pos = pos;
        ctx->turn_count++;

        free(pt->ids);
        free(pt);

        return ctx->tokens_generated;
    }
}

void flashmoe_cancel(FlashMoEContext *ctx) {
    if (!ctx) return;
    atomic_store(&ctx->cancelled, 1);
}

void flashmoe_reset(FlashMoEContext *ctx) {
    if (!ctx || !ctx->loaded) return;

    @autoreleasepool {
        // Wait for any in-flight GPU work
        if (g_deferred.active) {
            [g_deferred.cmd_experts waitUntilCompleted];
            g_deferred.active = 0;
            g_deferred.cmd_experts = nil;
        }

        // Reset delta-net state
        reset_delta_net_state();

        // Reset KV caches
        for (int i = 0; i < g_cfg.num_layers; i++) {
            if (ctx->kv_caches[i]) {
                ctx->kv_caches[i]->len = 0;
            }
        }

        // Reset conversation position
        ctx->current_pos = 0;
        ctx->turn_count = 0;

        // Reset stats
        ctx->tokens_generated = 0;
        ctx->tokens_per_second = 0;
        ctx->total_time_ms = 0;
        ctx->ttft_ms = 0;
    }
}

void flashmoe_get_stats(FlashMoEContext *ctx, FlashMoEStats *stats) {
    if (!ctx || !stats) return;

    memset(stats, 0, sizeof(FlashMoEStats));

    if (ctx->loaded) {
        snprintf(stats->model_name, sizeof(stats->model_name), "%s",
                 g_model_path_for_tokenizer ? g_model_path_for_tokenizer : "unknown");
        stats->num_layers = g_cfg.num_layers;
        stats->num_linear_layers = g_cfg.num_linear_layers;
        stats->num_full_attn_layers = g_cfg.num_full_attn_layers;
        stats->num_experts = g_cfg.num_experts;
        stats->active_experts_k = ctx->K;
        stats->hidden_dim = HIDDEN_DIM;
        stats->vocab_size = VOCAB_SIZE;
        stats->num_attn_heads = g_cfg.num_attn_heads;
        stats->num_kv_heads = g_cfg.num_kv_heads;
        stats->head_dim = g_cfg.head_dim;
        stats->moe_intermediate = g_cfg.moe_intermediate;
        stats->is_smoke_test = (g_cfg.num_experts < 512) ? 1 : 0;

        // Determine expert quantization bits
        if (g_use_2bit)          stats->expert_quant_bits = 2;
        else if (g_use_q3_experts) stats->expert_quant_bits = 3;
        else                      stats->expert_quant_bits = 4;

        // Dense weights are MLX 4-bit (group_size=64) with BF16 scales+biases
        // Effective bits/param: 4 (weight) + 16/64 (scale) + 16/64 (bias) = 4.5 bits/param
        stats->dense_quant_bits = 4;
        stats->dense_avg_bits = 4.5f;

        stats->weight_file_bytes = ctx->wf ? ctx->wf->size : 0;
        stats->expert_size_each = (size_t)active_expert_size();

        // Compute total expert file bytes
        size_t total_expert = 0;
        for (int i = 0; i < g_cfg.num_layers; i++) {
            total_expert += ctx->layer_mmap_sizes[i];
        }
        stats->expert_file_bytes = total_expert;

        // Approximate Metal buffer bytes
        stats->metal_buffer_bytes = (size_t)g_cfg.expert_size_computed * MAX_K * 2 +  // expert data (double-buffered)
                                    (size_t)HIDDEN_DIM * sizeof(float) * 20 +  // various working buffers
                                    (size_t)VOCAB_SIZE * sizeof(float);          // logits
    }

    stats->tokens_per_second = ctx->tokens_per_second;
    stats->tokens_generated = ctx->tokens_generated;
    stats->total_time_ms = ctx->total_time_ms;
    stats->ttft_ms = ctx->ttft_ms;
}

// ============================================================================
// Profiling — run short generation with timing and return report string
// ============================================================================

// Helper: get device machine identifier (e.g. "iPad16,6")
static const char *get_device_machine(void) {
    static char machine[64] = {0};
    if (machine[0]) return machine;
    struct utsname u;
    if (uname(&u) == 0) {
        strlcpy(machine, u.machine, sizeof(machine));
    } else {
        strlcpy(machine, "unknown", sizeof(machine));
    }
    return machine;
}

// Helper: map machine ID to marketing name
static const char *get_device_name(void) {
    const char *m = get_device_machine();
    // iPad Pro M4
    if (strncmp(m, "iPad16,3", 8) == 0 || strncmp(m, "iPad16,4", 8) == 0) return "iPad Pro 11\" (M4)";
    if (strncmp(m, "iPad16,5", 8) == 0 || strncmp(m, "iPad16,6", 8) == 0) return "iPad Pro 13\" (M4)";
    // iPad Air M3
    if (strncmp(m, "iPad15,3", 8) == 0 || strncmp(m, "iPad15,4", 8) == 0) return "iPad Air 11\" (M3)";
    if (strncmp(m, "iPad15,5", 8) == 0 || strncmp(m, "iPad15,6", 8) == 0) return "iPad Air 13\" (M3)";
    // iPad Pro M2
    if (strncmp(m, "iPad14,5", 8) == 0 || strncmp(m, "iPad14,6", 8) == 0) return "iPad Pro 11\" (M2)";
    if (strncmp(m, "iPad14,7", 8) == 0 || strncmp(m, "iPad14,8", 8) == 0) return "iPad Pro 12.9\" (M2)";
    // iPad Air M2
    if (strncmp(m, "iPad14,10", 9) == 0 || strncmp(m, "iPad14,11", 9) == 0) return "iPad Air 11\" (M2)";
    // iPad Pro M1
    if (strncmp(m, "iPad13,4", 8) == 0 || strncmp(m, "iPad13,5", 8) == 0) return "iPad Pro 11\" (M1)";
    if (strncmp(m, "iPad13,8", 8) == 0 || strncmp(m, "iPad13,9", 8) == 0) return "iPad Pro 12.9\" (M1)";
    // iPad Air M1
    if (strncmp(m, "iPad13,16", 9) == 0 || strncmp(m, "iPad13,17", 9) == 0) return "iPad Air (M1)";
    // iPhone 17 Pro
    if (strncmp(m, "iPhone18,1", 10) == 0) return "iPhone 17 Pro";
    if (strncmp(m, "iPhone18,2", 10) == 0) return "iPhone 17 Pro Max";
    if (strncmp(m, "iPhone18,3", 10) == 0) return "iPhone 17 Air";
    if (strncmp(m, "iPhone18,4", 10) == 0) return "iPhone 17";
    // iPhone 16 Pro
    if (strncmp(m, "iPhone17,1", 10) == 0) return "iPhone 16 Pro";
    if (strncmp(m, "iPhone17,2", 10) == 0) return "iPhone 16 Pro Max";
    if (strncmp(m, "iPhone17,3", 10) == 0) return "iPhone 16";
    // iPhone 15 Pro
    if (strncmp(m, "iPhone16,1", 10) == 0) return "iPhone 15 Pro";
    if (strncmp(m, "iPhone16,2", 10) == 0) return "iPhone 15 Pro Max";
    // Mac (running as Designed for iPad)
    if (strncmp(m, "arm64", 5) == 0) return "Mac (Apple Silicon)";
    return m;  // fallback to raw machine ID
}

// Enable timing accumulation and reset counters
void flashmoe_timing_enable(FlashMoEContext *ctx) {
    (void)ctx;
    g_timing_enabled = 1;
    memset(&g_timing, 0, sizeof(g_timing));
}

// Build timing report from accumulated data. Caller must free().
char *flashmoe_timing_report(FlashMoEContext *ctx) {
    if (!ctx) return NULL;

    g_timing_enabled = 0;

    char *buf = malloc(8192);
    if (!buf) return NULL;
    int pos = 0;
    int n = g_timing.count;
    int toks = g_timing.token_count;

    // ---- Device & Model header ----
    const char *model_path = g_model_path_for_tokenizer ? g_model_path_for_tokenizer : "unknown";
    const char *model_name = strrchr(model_path, '/');
    model_name = model_name ? model_name + 1 : model_path;

    uint64_t total_ram = [NSProcessInfo processInfo].physicalMemory;
    double avail_ram_mb = 0;
#if TARGET_OS_IOS
    avail_ram_mb = (double)os_proc_available_memory() / (1024 * 1024);
#endif

    pos += snprintf(buf + pos, 8192 - pos,
        "Device:  %s (%s)\n"
        "RAM:     %.0f GB total, %.0f MB free\n"
        "OS:      %s %s\n"
        "Model:   %s\n"
        "Quant:   %d-bit experts, K=%d\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
        get_device_name(), get_device_machine(),
        (double)total_ram / (1024.0 * 1024 * 1024), avail_ram_mb,
#if TARGET_OS_IOS
        [[[UIDevice currentDevice] systemName] UTF8String],
        [[[UIDevice currentDevice] systemVersion] UTF8String],
#else
        "macOS",
        [[[NSProcessInfo processInfo] operatingSystemVersionString] UTF8String],
#endif
        model_name,
        g_use_2bit ? 2 : (g_use_q3_experts ? 3 : 4), ctx->K);

    if (n == 0 || toks == 0) {
        pos += snprintf(buf + pos, 8192 - pos, "No timing data (%d layers timed, %d tokens)\n", n, toks);
        return buf;
    }

    // Per-token decode breakdown
    double dense_attn_ms = (g_timing.cmd1_submit + g_timing.cmd1_wait + g_timing.cpu_attn) / n * g_cfg.num_layers;
    double oproj_shared_ms = (g_timing.cmd2_encode + g_timing.cmd2_wait + g_timing.routing_cpu) / n * g_cfg.num_layers;
    double expert_io_ms = g_timing.expert_io / n * g_cfg.num_layers;
    double expert_compute_ms = (g_timing.cmd3_encode + g_timing.deferred_wait + g_timing.deferred_cpu) / n * g_cfg.num_layers;
    double lm_ms = g_timing.lm_head / toks;
    double total_ms = dense_attn_ms + oproj_shared_ms + expert_io_ms + expert_compute_ms + lm_ms;

    double linear_ms = (g_timing.count_linear > 0) ? g_timing.total_linear / g_timing.count_linear * g_cfg.num_linear_layers : 0;
    double full_ms = (g_timing.count_full > 0) ? g_timing.total_full / g_timing.count_full * g_cfg.num_full_attn_layers : 0;

    pos += snprintf(buf + pos, 8192 - pos,
        "\nDecode Breakdown (%d tokens)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", toks);
    pos += snprintf(buf + pos, 8192 - pos,
        "Dense/attn (CMD1):  %5.1f ms  %4.1f%%\n"
        "  GatedDeltaNet:    %5.1f ms  (%d layers)\n"
        "  Full attention:   %5.1f ms  (%d layers)\n",
        dense_attn_ms, 100*dense_attn_ms/total_ms,
        linear_ms, g_cfg.num_linear_layers,
        full_ms, g_cfg.num_full_attn_layers);
    pos += snprintf(buf + pos, 8192 - pos,
        "o_proj+shared (CMD2): %3.1f ms  %4.1f%%\n",
        oproj_shared_ms, 100*oproj_shared_ms/total_ms);
    pos += snprintf(buf + pos, 8192 - pos,
        "Expert I/O (SSD):   %5.1f ms  %4.1f%%\n",
        expert_io_ms, 100*expert_io_ms/total_ms);
    pos += snprintf(buf + pos, 8192 - pos,
        "Expert compute:     %5.1f ms  %4.1f%%\n",
        expert_compute_ms, 100*expert_compute_ms/total_ms);
    pos += snprintf(buf + pos, 8192 - pos,
        "LM head:            %5.1f ms  %4.1f%%\n",
        lm_ms, 100*lm_ms/total_ms);

    // Compute effective SSD throughput
    int expert_size = g_use_2bit ? EXPERT_SIZE_2BIT :
                      g_use_q3_experts ? EXPERT_SIZE_Q3_HYBRID : EXPERT_SIZE;
    double io_bytes_per_tok = (double)ctx->K * g_cfg.num_layers * expert_size;
    double io_gb_per_tok = io_bytes_per_tok / (1024.0 * 1024 * 1024);
    double ssd_gbps = (expert_io_ms > 0) ? io_gb_per_tok / (expert_io_ms / 1000.0) : 0;

    pos += snprintf(buf + pos, 8192 - pos,
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Total per token:    %5.1f ms  (%.1f tok/s)\n"
        "TTFT:               %5.0f ms\n"
        "Expert quant:       %d-bit\n"
        "Experts:            %d (K=%d)\n"
        "Expert I/O/tok:     %.2f GB\n"
        "SSD throughput:     %.1f GB/s\n",
        total_ms, 1000.0/total_ms,
        ctx->ttft_ms,
        g_use_2bit ? 2 : (g_use_q3_experts ? 3 : 4),
        g_cfg.num_experts, ctx->K,
        io_gb_per_tok, ssd_gbps);

    // Per-layer avg
    pos += snprintf(buf + pos, 8192 - pos,
        "\nPer-Layer Avg (ms):\n"
        "  deferred_wait:  %6.3f\n"
        "  cmd1 (submit):  %6.3f\n"
        "  cmd1 (wait):    %6.3f\n"
        "  cpu_attn:       %6.3f\n"
        "  cmd2 (encode):  %6.3f\n"
        "  cmd2 (wait):    %6.3f\n"
        "  routing_cpu:    %6.3f\n"
        "  expert_io:      %6.3f\n"
        "  cmd3_encode:    %6.3f\n",
        g_timing.deferred_wait / n,
        g_timing.cmd1_submit / n,
        g_timing.cmd1_wait / n,
        g_timing.cpu_attn / n,
        g_timing.cmd2_encode / n,
        g_timing.cmd2_wait / n,
        g_timing.routing_cpu / n,
        g_timing.expert_io / n,
        g_timing.cmd3_encode / n);

    NSLog(@"[profile]\n%s", buf);
    return buf;
}

// Convenience: run a self-contained timing profile (blocking)
char *flashmoe_run_profile(FlashMoEContext *ctx, int num_tokens) {
    if (!ctx || !ctx->loaded) return NULL;
    flashmoe_timing_enable(ctx);
    flashmoe_reset(ctx);
    flashmoe_generate(ctx, "What is Apple Neural Engine?", num_tokens, NULL, NULL);
    return flashmoe_timing_report(ctx);
}

int flashmoe_validate_model(const char *model_path) {
    if (!model_path) return -1;

    // Check config.json
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_path);
    if (access(path, R_OK) != 0) return -1;

    // Check model_weights.bin
    snprintf(path, sizeof(path), "%s/model_weights.bin", model_path);
    if (access(path, R_OK) != 0) return -1;

    // Check model_weights.json
    snprintf(path, sizeof(path), "%s/model_weights.json", model_path);
    if (access(path, R_OK) != 0) return -1;

    // Check for at least one expert layer file
    snprintf(path, sizeof(path), "%s/packed_experts/layer_00.bin", model_path);
    int has_4bit = (access(path, R_OK) == 0);

    snprintf(path, sizeof(path), "%s/packed_experts_tiered/layer_00.bin", model_path);
    int has_tiered = (access(path, R_OK) == 0);

    snprintf(path, sizeof(path), "%s/packed_experts_2bit/layer_00.bin", model_path);
    int has_2bit = (access(path, R_OK) == 0);

    if (!has_4bit && !has_tiered && !has_2bit) return -1;

    return 0;
}

int flashmoe_turn_count(FlashMoEContext *ctx) {
    if (!ctx) return 0;
    return ctx->turn_count;
}

const char *flashmoe_last_error(FlashMoEContext *ctx) {
    if (!ctx) return "NULL context";
    return ctx->last_error;
}

