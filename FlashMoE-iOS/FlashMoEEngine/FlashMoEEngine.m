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
        load_model_config(model_path);
        alloc_tracking_arrays();

        // Apply config overrides — cap context length for iOS memory constraints
        if (config->max_context > 0) {
            cfg.max_seq_len = config->max_context;
        }
        // iOS: adaptive context length based on available device memory
        // Must account for: weight file (mmap'd), Metal buffers, delta-net state, KV caches
        {
#if TARGET_OS_IPHONE || TARGET_OS_IOS
            size_t avail = os_proc_available_memory();
#else
            // macOS fallback: estimate available memory using host_statistics64
            mach_port_t host = mach_host_self();
            vm_size_t pageSize = 0;
            host_page_size(host, &pageSize);
            vm_statistics64_data_t vmStats = {0};
            mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
            kern_return_t kr = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vmStats, &count);
            size_t avail = 0;
            if (kr == KERN_SUCCESS) {
                uint64_t freePages = vmStats.free_count;
                uint64_t inactivePages = vmStats.inactive_count;
                uint64_t speculativePages = vmStats.speculative_count;
                avail = (size_t)((freePages + inactivePages + speculativePages) * (uint64_t)pageSize);
            }
#endif
            size_t kv_cost_per_pos = (size_t)cfg.num_kv_heads * cfg.head_dim * sizeof(float)
                                     * 2  // k + v
                                     * cfg.num_full_attn_layers
                                     * 2; // CPU + GPU mirror

            // Estimate non-KV Metal memory:
            //   delta-net: num_linear_layers * v_heads * v_dim * k_dim * 4
            //   multi-expert: MAX_K * 2 * expert_size
            //   working buffers: ~50 MB
            size_t delta_net_bytes = (size_t)cfg.num_linear_layers *
                cfg.linear_num_v_heads * cfg.linear_value_dim * cfg.linear_key_dim * sizeof(float);
            size_t expert_buf_bytes = (size_t)MAX_K * 2 * cfg.expert_size_4bit;
            size_t fixed_metal = delta_net_bytes + expert_buf_bytes + 50 * 1024 * 1024;

            // Reserve memory for: fixed Metal + OS headroom (2 GB) + expert page cache (at least 1 GB)
            size_t reserved = fixed_metal + (size_t)3 * 1024 * 1024 * 1024;
            size_t kv_budget = (avail > reserved) ? (avail - reserved) / 2 : avail / 8;

            int adaptive_max = (kv_cost_per_pos > 0) ? (int)(kv_budget / kv_cost_per_pos) : 8192;
            // Clamp to powers of 2: 512, 1024, 2048, 4096, 8192
            int capped = 512;
            for (int p = 512; p <= 8192; p *= 2) {
                if (p <= adaptive_max) capped = p;
            }
            if (cfg.max_seq_len > capped) {
                NSLog(@"[FlashMoE] Adaptive context: %d → %d (%.0f MB available, %.0f MB fixed Metal, KV %.0f bytes/pos)",
                      cfg.max_seq_len, capped, avail / 1e6, fixed_metal / 1e6, (double)kv_cost_per_pos);
                cfg.max_seq_len = capped;
            }
        }
        if (config->think_budget > 0) {
            g_think_budget = config->think_budget;
        }

        // Set tiered mode
        g_use_tiered = config->use_tiered;
        g_use_2bit = 0;

        // Set cache I/O split (fanout mode): >1 = split expert preads into N chunks
        if (config->cache_io_split > 1) {
            g_cache_io_split = config->cache_io_split;
        } else {
            g_cache_io_split = 1;  // disabled by default
        }

        // K = experts per token from config
        // K override: allow reducing active experts for memory-constrained devices.
        // Lower K = less I/O per token (linear reduction). Quality degrades gracefully
        // because the router still picks the best K experts from the full vocabulary.
        if (config->active_experts_k > 0 && config->active_experts_k <= cfg.num_experts_per_tok) {
            ctx->K = config->active_experts_k;
            if (config->verbose) {
                NSLog(@"[FlashMoE] K override: %d (model default: %d) — %.0f%% I/O reduction",
                      ctx->K, cfg.num_experts_per_tok,
                      (1.0 - (double)ctx->K / cfg.num_experts_per_tok) * 100);
            }
        } else {
            ctx->K = cfg.num_experts_per_tok;
        }

        // Safety: cap K to MAX_K to prevent buffer overflow on multi-expert buffers
        if (ctx->K > MAX_K) {
            NSLog(@"[FlashMoE] WARNING: K=%d exceeds MAX_K=%d, capping to %d", ctx->K, MAX_K, MAX_K);
            ctx->K = MAX_K;
        }

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
        ctx->layer_fds = calloc(cfg.num_layers, sizeof(int));
        ctx->layer_fds_cold_local = calloc(cfg.num_layers, sizeof(int));
        ctx->layer_mmaps = calloc(cfg.num_layers, sizeof(void *));
        ctx->layer_mmap_sizes = calloc(cfg.num_layers, sizeof(size_t));

        memset(g_expert_seen, 0, cfg.num_layers * ((cfg.num_experts + 7) / 8));

        for (int i = 0; i < cfg.num_layers; i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s/layer_%02d.bin", model_path,
                     g_use_tiered ? "packed_experts_tiered" :
                     g_use_2bit ? "packed_experts_2bit" : "packed_experts", i);
            ctx->layer_fds[i] = open(path, O_RDONLY);
            ctx->layer_fds_cold_local[i] = -1;
            ctx->layer_mmaps[i] = MAP_FAILED;
            ctx->layer_mmap_sizes[i] = 0;
            if (ctx->layer_fds[i] >= 0) {
                fcntl(ctx->layer_fds[i], F_RDAHEAD, 0);
                struct stat st;
                if (fstat(ctx->layer_fds[i], &st) == 0 && st.st_size > 0) {
                    ctx->layer_mmaps[i] = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE,
                                                ctx->layer_fds[i], 0);
                    if (ctx->layer_mmaps[i] != MAP_FAILED) {
                        ctx->layer_mmap_sizes[i] = st.st_size;
                    }
                }
            }
        }

        // Wire up global cold fds
        g_layer_fds_cold = ctx->layer_fds_cold_local;

        // ---- Allocate deferred expert state ----
        g_deferred.h_mid = calloc(cfg.hidden_dim, sizeof(float));

        // ---- Allocate per-layer state ----
        ctx->layer_states = calloc(cfg.num_layers, sizeof(void *));
        ctx->kv_caches = calloc(cfg.num_layers, sizeof(KVCache *));

        for (int i = 0; i < cfg.num_layers; i++) {
            if (cfg.is_full_attn[i]) {
                ctx->kv_caches[i] = kv_cache_new();
            } else {
                ctx->layer_states[i] = linear_attn_state_new();
            }
        }

        // ---- Allocate working buffers ----
        ctx->hidden = calloc(cfg.hidden_dim, sizeof(float));
        ctx->logits = calloc(cfg.vocab_size, sizeof(float));
        ctx->final_norm_w = get_tensor_ptr(ctx->wf, "model.norm.weight");

        // ---- Build layer cache (precomputes weight pointers) ----
        build_layer_cache(ctx->wf);

        ctx->loaded = 1;
        if (config->verbose) {
            NSLog(@"[FlashMoE] Model loaded: %d layers, %d experts (K=%d), hidden=%d",
                  cfg.num_layers, cfg.num_experts, ctx->K, cfg.hidden_dim);
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

        // Wait for any in-flight async pread
        if (g_async_pread.active) {
            dispatch_group_wait(g_async_pread.group, DISPATCH_TIME_FOREVER);
            g_async_pread.active = 0;
        }

        // Shutdown I/O pool
        io_pool_shutdown();

        // Close expert files
        if (ctx->layer_fds) {
            for (int i = 0; i < cfg.num_layers; i++) {
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
            for (int i = 0; i < cfg.num_layers; i++) {
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

        // Free deferred state
        free(g_deferred.h_mid); g_deferred.h_mid = NULL;

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

        // Free config dynamic arrays
        free(cfg.is_full_attn); cfg.is_full_attn = NULL;

        // Free tracking arrays (allocated by alloc_tracking_arrays)
        free(g_expert_freq);    g_expert_freq = NULL;
        free(g_expert_seen);    g_expert_seen = NULL;
        free(g_lz4_index);      g_lz4_index = NULL;
        free(g_cache_seen);     g_cache_seen = NULL;
        free(g_cache_last_touch_token); g_cache_last_touch_token = NULL;
        free(g_cache_last_evict_token); g_cache_last_evict_token = NULL;
        free(g_pred_experts);   g_pred_experts = NULL;
        free(g_pred_count);     g_pred_count = NULL;

        // Reset layer cache so it rebuilds on next load
        free(layer_cache);      layer_cache = NULL;
        layer_cache_built = 0;

        // Free tiered manifest
        if (g_tiered_manifest) {
            free(g_tiered_manifest);
            g_tiered_manifest = NULL;
            g_use_tiered = 0;
        }

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
        // (Note: MetalCtx uses ARC for ObjC objects, but struct is malloc'd)
        if (g_metal) {
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
        for (int i = 0; i < cfg.num_layers; i++) {
            if (ctx->kv_caches[i]) {
                ctx->kv_caches[i]->len = 0;
            }
        }

        int pos = 0;

        // ---- Batch prefill: embed all prompt tokens ----
        float *embed_batch = NULL;
        if (pt->count > 1) {
            embed_batch = malloc((size_t)pt->count * cfg.hidden_dim * sizeof(float));
            for (int i = 0; i < pt->count; i++) {
                embed_lookup(ctx->wf, pt->ids[i], embed_batch + (size_t)i * cfg.hidden_dim);
            }
        }

        // ---- Prefill intermediate tokens (discard expert output) ----
        if (pt->count > 1) {
            for (int token_idx = 0; token_idx < pt->count - 1; token_idx++) {
                if (atomic_load(&ctx->cancelled)) {
                    free(embed_batch);
                    free(pt->ids); free(pt);
                    return ctx->tokens_generated;
                }

                memcpy(ctx->hidden, embed_batch + (size_t)token_idx * cfg.hidden_dim,
                       cfg.hidden_dim * sizeof(float));

                for (int layer = 0; layer < cfg.num_layers; layer++) {
                    int is_full = cfg.is_full_attn[layer];
                    fused_layer_forward(ctx->wf, layer, ctx->hidden,
                                        is_full ? ctx->kv_caches[layer] : NULL,
                                        is_full ? NULL : ctx->layer_states[layer],
                                        pos,
                                        ctx->layer_mmaps[layer] != MAP_FAILED ? ctx->layer_mmaps[layer] : NULL,
                                        K, ctx->layer_fds[layer]);
                }
                discard_deferred_experts();
                pos++;
            }
        }

        // ---- Last prefill token (need full hidden state) ----
        {
            if (embed_batch) {
                memcpy(ctx->hidden, embed_batch + (size_t)(pt->count - 1) * cfg.hidden_dim,
                       cfg.hidden_dim * sizeof(float));
            } else {
                embed_lookup(ctx->wf, pt->ids[0], ctx->hidden);
            }

            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
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
            float *normed = malloc(cfg.hidden_dim * sizeof(float));
            cpu_rms_norm(ctx->hidden, ctx->final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
            memcpy(ctx->hidden, normed, cfg.hidden_dim * sizeof(float));
            free(normed);
        }

        lm_head_forward(ctx->wf, ctx->hidden, ctx->logits);
        int next_token = cpu_argmax(ctx->logits, cfg.vocab_size);

        ctx->ttft_ms = now_ms() - t0;
        ctx->tokens_generated = 1;

        // ---- Invoke callback for first token ----
        const char *token_text = decode_token(ctx->vocab, next_token);
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

        int in_think = (next_token == cfg.think_start_token) ? 1 : 0;
        int think_tokens = 0;

        // ---- Auto-regressive generation loop ----
        double gen_start = now_ms();

        for (int gen = 1; gen < max_tokens; gen++) {
            // Check cancellation
            if (atomic_load(&ctx->cancelled)) break;

            // Check EOS
            int is_eos = 0;
            for (int e = 0; e < cfg.num_eos_tokens; e++) {
                if (next_token == cfg.eos_token_ids[e]) { is_eos = 1; break; }
            }
            if (is_eos) break;

            // Think budget enforcement
            if (next_token == cfg.think_start_token) in_think = 1;
            if (next_token == cfg.think_end_token) in_think = 0;
            if (in_think) think_tokens++;

            // Embed + forward pass
            embed_lookup(ctx->wf, next_token, ctx->hidden);

            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
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
            if (ctx->final_norm_w) {
                float *normed = malloc(cfg.hidden_dim * sizeof(float));
                cpu_rms_norm(ctx->hidden, ctx->final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
                memcpy(ctx->hidden, normed, cfg.hidden_dim * sizeof(float));
                free(normed);
            }

            lm_head_forward(ctx->wf, ctx->hidden, ctx->logits);
            next_token = cpu_argmax(ctx->logits, cfg.vocab_size);

            // Think budget: force end thinking
            if (in_think && g_think_budget > 0 && think_tokens >= g_think_budget) {
                next_token = cfg.think_end_token;
                in_think = 0;
            }

            ctx->tokens_generated++;

            // Compute tok/s
            double elapsed_gen = now_ms() - gen_start;
            ctx->tokens_per_second = elapsed_gen > 0 ? (ctx->tokens_generated - 1) * 1000.0 / elapsed_gen : 0;

            // Invoke callback
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
        PromptTokens *pt = tokenize_continuation_turn_shared(user_content);
        if (!pt) {
            snprintf(ctx->last_error, sizeof(ctx->last_error), "Failed to tokenize continuation turn");
            return -1;
        }

        int K = ctx->K;
        int pos = ctx->current_pos;  // Resume from where we left off

        // Check we have room in the KV cache
        if (pos + pt->count + max_tokens > cfg.max_seq_len) {
            NSLog(@"[FlashMoE] Context full (%d + %d + %d > %d), resetting to fresh generation",
                  pos, pt->count, max_tokens, cfg.max_seq_len);
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
            embed_batch = malloc((size_t)pt->count * cfg.hidden_dim * sizeof(float));
            for (int i = 0; i < pt->count; i++) {
                embed_lookup(ctx->wf, pt->ids[i], embed_batch + (size_t)i * cfg.hidden_dim);
            }
        }

        if (pt->count > 1) {
            for (int token_idx = 0; token_idx < pt->count - 1; token_idx++) {
                if (atomic_load(&ctx->cancelled)) {
                    free(embed_batch);
                    free(pt->ids); free(pt);
                    return ctx->tokens_generated;
                }

                memcpy(ctx->hidden, embed_batch + (size_t)token_idx * cfg.hidden_dim,
                       cfg.hidden_dim * sizeof(float));

                for (int layer = 0; layer < cfg.num_layers; layer++) {
                    int is_full = cfg.is_full_attn[layer];
                    fused_layer_forward(ctx->wf, layer, ctx->hidden,
                                        is_full ? ctx->kv_caches[layer] : NULL,
                                        is_full ? NULL : ctx->layer_states[layer],
                                        pos,
                                        ctx->layer_mmaps[layer] != MAP_FAILED ? ctx->layer_mmaps[layer] : NULL,
                                        K, ctx->layer_fds[layer]);
                }
                discard_deferred_experts();
                pos++;
            }
        }

        // Last prefill token
        {
            if (embed_batch) {
                memcpy(ctx->hidden, embed_batch + (size_t)(pt->count - 1) * cfg.hidden_dim,
                       cfg.hidden_dim * sizeof(float));
            } else {
                embed_lookup(ctx->wf, pt->ids[0], ctx->hidden);
            }

            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
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
            float *normed = malloc(cfg.hidden_dim * sizeof(float));
            cpu_rms_norm(ctx->hidden, ctx->final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
            memcpy(ctx->hidden, normed, cfg.hidden_dim * sizeof(float));
            free(normed);
        }

        lm_head_forward(ctx->wf, ctx->hidden, ctx->logits);
        int next_token = cpu_argmax(ctx->logits, cfg.vocab_size);

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

        int in_think = (next_token == cfg.think_start_token) ? 1 : 0;
        int think_tokens = 0;

        // ---- Auto-regressive generation loop ----
        double gen_start = now_ms();

        for (int gen = 1; gen < max_tokens; gen++) {
            if (atomic_load(&ctx->cancelled)) break;

            int is_eos = 0;
            for (int e = 0; e < cfg.num_eos_tokens; e++) {
                if (next_token == cfg.eos_token_ids[e]) { is_eos = 1; break; }
            }
            if (is_eos) break;

            if (next_token == cfg.think_start_token) in_think = 1;
            if (next_token == cfg.think_end_token) in_think = 0;
            if (in_think) think_tokens++;

            embed_lookup(ctx->wf, next_token, ctx->hidden);

            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
                fused_layer_forward(ctx->wf, layer, ctx->hidden,
                                    is_full ? ctx->kv_caches[layer] : NULL,
                                    is_full ? NULL : ctx->layer_states[layer],
                                    pos,
                                    ctx->layer_mmaps[layer] != MAP_FAILED ? ctx->layer_mmaps[layer] : NULL,
                                    K, ctx->layer_fds[layer]);
            }
            complete_deferred_experts();
            pos++;

            if (ctx->final_norm_w) {
                float *normed = malloc(cfg.hidden_dim * sizeof(float));
                cpu_rms_norm(ctx->hidden, ctx->final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
                memcpy(ctx->hidden, normed, cfg.hidden_dim * sizeof(float));
                free(normed);
            }

            lm_head_forward(ctx->wf, ctx->hidden, ctx->logits);
            next_token = cpu_argmax(ctx->logits, cfg.vocab_size);

            if (in_think && g_think_budget > 0 && think_tokens >= g_think_budget) {
                next_token = cfg.think_end_token;
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
        for (int i = 0; i < cfg.num_layers; i++) {
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
        snprintf(stats->model_name, sizeof(stats->model_name), "%s", cfg.model_path);
        stats->num_layers = cfg.num_layers;
        stats->num_experts = cfg.num_experts;
        stats->active_experts_k = ctx->K;
        stats->hidden_dim = cfg.hidden_dim;
        stats->vocab_size = cfg.vocab_size;
        stats->weight_file_bytes = ctx->wf ? ctx->wf->size : 0;

        // Compute total expert file bytes
        size_t total_expert = 0;
        for (int i = 0; i < cfg.num_layers; i++) {
            total_expert += ctx->layer_mmap_sizes[i];
        }
        stats->expert_file_bytes = total_expert;

        // Approximate Metal buffer bytes
        stats->metal_buffer_bytes = (size_t)cfg.expert_size_4bit * MAX_K * 2 +  // expert data (double-buffered)
                                    (size_t)cfg.hidden_dim * sizeof(float) * 20 +  // various working buffers
                                    (size_t)cfg.vocab_size * sizeof(float);          // logits
    }

    stats->tokens_per_second = ctx->tokens_per_second;
    stats->tokens_generated = ctx->tokens_generated;
    stats->total_time_ms = ctx->total_time_ms;
    stats->ttft_ms = ctx->ttft_ms;
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

