# Runtime Model Config Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all ~54 model-specific `#define` constants in `infer.m` with a runtime `ModelConfig` struct populated from HuggingFace `config.json` + `tokenizer.json`, enabling model switching via `--model` flag without recompilation.

**Architecture:** A single `ModelConfig cfg` global struct is populated at startup by parsing JSON files using NSJSONSerialization. Expert byte offsets are computed from dimensions + quantization params. Static arrays sized by model constants become dynamically allocated after config loading.

**Tech Stack:** Objective-C (NSJSONSerialization), C (malloc/free), Metal (unchanged — already parameterized)

**Spec:** `docs/superpowers/specs/2026-03-20-runtime-model-config-design.md`

---

## File Structure

All changes are in a single file:

- **Modify:** `metal_infer/infer.m` — the entire inference engine (~7200 lines)
  - Remove lines 68-140 (model `#define`s)
  - Add `ModelConfig` struct + `load_model_config()` + `compute_expert_offsets()` + `alloc_tracking_arrays()` (~200 lines)
  - Convert ~13 static/stack arrays to dynamic allocation
  - Replace ~960 occurrences of `#define` names with `cfg.field` references

No other files need changes. `shaders.metal` is already parameterized. `chat.m` is a pure HTTP client with no model constants.

---

## Task 1: Add ModelConfig struct and config loader

**Files:**
- Modify: `metal_infer/infer.m:68-140`

This task replaces the `#define` block with the `ModelConfig` struct, adds the JSON parsing function, expert offset computation, and dynamic array allocation. The old `#define`s are kept temporarily as fallback validation.

- [ ] **Step 1.1: Add ModelConfig struct after includes (before old defines)**

Insert after line 66 (after `#include <compression.h>`), before the old defines block. The struct holds all model-specific parameters:

```c
// ============================================================================
// Runtime model configuration (populated from HuggingFace config.json)
// ============================================================================

typedef struct {
    // Core architecture
    int hidden_dim;
    int num_layers;
    int num_attn_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    float rms_norm_eps;

    // MoE
    int num_experts;
    int num_experts_per_tok;
    int moe_intermediate;
    int shared_intermediate;
    int group_size;
    int bits;

    // Linear attention (GatedDeltaNet)
    int linear_num_v_heads;
    int linear_num_k_heads;
    int linear_key_dim;
    int linear_value_dim;
    int conv_kernel_size;

    // Full attention
    float rope_theta;
    float partial_rotary;

    // Layer type map
    int num_full_attn_layers;
    int num_linear_layers;
    bool *is_full_attn;       // [num_layers]
    int *full_attn_index;     // [num_layers] — index into full-attn buffers, or -1
    int *linear_index;        // [num_layers] — index into linear-attn buffers, or -1

    // Derived: expert byte offsets (4-bit)
    size_t expert_size_4bit;
    size_t gate_w_off_4, gate_s_off_4, gate_b_off_4;
    size_t up_w_off_4, up_s_off_4, up_b_off_4;
    size_t down_w_off_4, down_s_off_4, down_b_off_4;

    // Derived: expert byte offsets (2-bit)
    size_t expert_size_2bit;
    size_t gate_w_off_2, gate_s_off_2, gate_b_off_2;
    size_t up_w_off_2, up_s_off_2, up_b_off_2;
    size_t down_w_off_2, down_s_off_2, down_b_off_2;

    // Derived dimensions
    int linear_total_key;
    int linear_total_value;
    int linear_conv_dim;
    int rotary_dim;

    // Special tokens
    int eos_token_ids[8];
    int num_eos_tokens;
    int think_start_token;
    int think_end_token;

    // Context limits
    int max_seq_len;
    int gpu_kv_seq;

    // Model path (resolved)
    char model_path[1024];
} ModelConfig;

static ModelConfig cfg;
```

- [ ] **Step 1.2: Add compute_expert_offsets()**

Place right after the struct definition:

```c
static void compute_expert_offsets(ModelConfig *c) {
    int mid = c->moe_intermediate;
    int hid = c->hidden_dim;
    int gs = c->group_size;

    for (int b = 4; b >= 2; b -= 2) {
        int vals_per_u32 = 32 / b;
        // gate_proj [mid, hid]
        size_t gw = (size_t)mid * ((hid + vals_per_u32 - 1) / vals_per_u32) * 4;
        size_t gs_sz = (size_t)mid * ((hid + gs - 1) / gs) * 2;
        size_t gb = gs_sz;
        // up_proj [mid, hid] — same shape
        size_t uw = gw, us = gs_sz, ub = gb;
        // down_proj [hid, mid]
        size_t dw = (size_t)hid * ((mid + vals_per_u32 - 1) / vals_per_u32) * 4;
        size_t ds = (size_t)hid * ((mid + gs - 1) / gs) * 2;
        size_t db = ds;

        size_t off = 0;
        if (b == 4) {
            c->gate_w_off_4 = off; off += gw;
            c->gate_s_off_4 = off; off += gs_sz;
            c->gate_b_off_4 = off; off += gb;
            c->up_w_off_4   = off; off += uw;
            c->up_s_off_4   = off; off += us;
            c->up_b_off_4   = off; off += ub;
            c->down_w_off_4 = off; off += dw;
            c->down_s_off_4 = off; off += ds;
            c->down_b_off_4 = off; off += db;
            c->expert_size_4bit = off;
        } else {
            c->gate_w_off_2 = off; off += gw;
            c->gate_s_off_2 = off; off += gs_sz;
            c->gate_b_off_2 = off; off += gb;
            c->up_w_off_2   = off; off += uw;
            c->up_s_off_2   = off; off += us;
            c->up_b_off_2   = off; off += ub;
            c->down_w_off_2 = off; off += dw;
            c->down_s_off_2 = off; off += ds;
            c->down_b_off_2 = off; off += db;
            c->expert_size_2bit = off;
        }
    }
}
```

- [ ] **Step 1.3: Add load_model_config() — the JSON parser**

This function reads `config.json` and `tokenizer.json` using NSJSONSerialization:

```c
static void load_model_config(const char *model_dir) {
    memset(&cfg, 0, sizeof(cfg));
    cfg.think_start_token = -1;
    cfg.think_end_token = -1;
    cfg.gpu_kv_seq = 8192;

    // Resolve HF snapshot directory
    NSString *base = [NSString stringWithUTF8String:model_dir];
    NSString *configPath = [base stringByAppendingPathComponent:@"config.json"];
    NSFileManager *fm = [NSFileManager defaultManager];

    if (![fm fileExistsAtPath:configPath]) {
        NSString *snapDir = [base stringByAppendingPathComponent:@"snapshots"];
        if ([fm fileExistsAtPath:snapDir]) {
            NSArray *snaps = [[fm contentsOfDirectoryAtPath:snapDir error:nil]
                              sortedArrayUsingSelector:@selector(compare:)];
            for (NSString *snap in snaps) {
                NSString *candidate = [[snapDir stringByAppendingPathComponent:snap]
                                        stringByAppendingPathComponent:@"config.json"];
                if ([fm fileExistsAtPath:candidate]) {
                    base = [snapDir stringByAppendingPathComponent:snap];
                    configPath = candidate;
                    break;
                }
            }
        }
    }

    if (![fm fileExistsAtPath:configPath]) {
        fprintf(stderr, "FATAL: config.json not found in %s\n", model_dir);
        exit(1);
    }

    strlcpy(cfg.model_path, [base UTF8String], sizeof(cfg.model_path));

    // Parse config.json
    NSData *data = [NSData dataWithContentsOfFile:configPath];
    NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    NSDictionary *tc = root[@"text_config"];
    if (!tc) { fprintf(stderr, "FATAL: config.json missing text_config\n"); exit(1); }

    cfg.hidden_dim       = [tc[@"hidden_size"] intValue];
    cfg.num_layers       = [tc[@"num_hidden_layers"] intValue];
    cfg.num_attn_heads   = [tc[@"num_attention_heads"] intValue];
    cfg.num_kv_heads     = [tc[@"num_key_value_heads"] intValue];
    cfg.head_dim         = [tc[@"head_dim"] intValue];
    cfg.vocab_size       = [tc[@"vocab_size"] intValue];
    cfg.rms_norm_eps     = [tc[@"rms_norm_eps"] floatValue];
    cfg.num_experts      = [tc[@"num_experts"] intValue];
    cfg.num_experts_per_tok = [tc[@"num_experts_per_tok"] intValue];
    cfg.moe_intermediate = [tc[@"moe_intermediate_size"] intValue];
    cfg.shared_intermediate = [tc[@"shared_expert_intermediate_size"] intValue];
    cfg.linear_num_v_heads = [tc[@"linear_num_value_heads"] intValue];
    cfg.linear_num_k_heads = [tc[@"linear_num_key_heads"] intValue];
    cfg.linear_key_dim   = [tc[@"linear_key_head_dim"] intValue];
    cfg.linear_value_dim = [tc[@"linear_value_head_dim"] intValue];
    cfg.conv_kernel_size = tc[@"linear_conv_kernel_dim"] ? [tc[@"linear_conv_kernel_dim"] intValue] : 4;
    cfg.max_seq_len      = [tc[@"max_position_embeddings"] intValue];

    // Quantization
    NSDictionary *qc = root[@"quantization_config"] ?: root[@"quantization"];
    if (qc) {
        cfg.group_size = [qc[@"group_size"] intValue];
        cfg.bits       = [qc[@"bits"] intValue];
    } else {
        cfg.group_size = 64;
        cfg.bits       = 4;
        fprintf(stderr, "[config] WARNING: no quantization_config, defaulting to 4-bit group_size=64\n");
    }

    // RoPE parameters
    NSDictionary *rope = tc[@"rope_parameters"];
    if (rope) {
        cfg.rope_theta    = [rope[@"rope_theta"] floatValue];
        cfg.partial_rotary = [rope[@"partial_rotary_factor"] floatValue];
    } else {
        cfg.rope_theta    = 10000000.0f;
        cfg.partial_rotary = 0.25f;
    }

    // Layer types
    NSArray *layerTypes = tc[@"layer_types"];
    cfg.is_full_attn    = calloc(cfg.num_layers, sizeof(bool));
    cfg.full_attn_index = malloc(cfg.num_layers * sizeof(int));
    cfg.linear_index    = malloc(cfg.num_layers * sizeof(int));

    int full_count = 0, linear_count = 0;
    if (layerTypes && [layerTypes count] == (NSUInteger)cfg.num_layers) {
        for (int i = 0; i < cfg.num_layers; i++) {
            cfg.is_full_attn[i] = [layerTypes[i] isEqualToString:@"full_attention"];
        }
    } else {
        // Fallback: use full_attn_interval pattern
        int interval = tc[@"full_attention_interval"] ? [tc[@"full_attention_interval"] intValue] : 4;
        for (int i = 0; i < cfg.num_layers; i++) {
            cfg.is_full_attn[i] = ((i + 1) % interval == 0);
        }
        fprintf(stderr, "[config] Using full_attn_interval=%d (no explicit layer_types)\n", interval);
    }

    for (int i = 0; i < cfg.num_layers; i++) {
        if (cfg.is_full_attn[i]) {
            cfg.full_attn_index[i] = full_count++;
            cfg.linear_index[i] = -1;
        } else {
            cfg.linear_index[i] = linear_count++;
            cfg.full_attn_index[i] = -1;
        }
    }
    cfg.num_full_attn_layers = full_count;
    cfg.num_linear_layers = linear_count;

    // EOS tokens (can be int or array in config.json)
    id eosVal = root[@"eos_token_id"];
    if ([eosVal isKindOfClass:[NSArray class]]) {
        NSArray *arr = (NSArray *)eosVal;
        cfg.num_eos_tokens = (int)[arr count];
        if (cfg.num_eos_tokens > 8) cfg.num_eos_tokens = 8;
        for (int i = 0; i < cfg.num_eos_tokens; i++)
            cfg.eos_token_ids[i] = [arr[i] intValue];
    } else {
        cfg.num_eos_tokens = 1;
        cfg.eos_token_ids[0] = [eosVal intValue];
    }

    // Think tokens from tokenizer.json added_tokens
    NSString *tokPath = [base stringByAppendingPathComponent:@"tokenizer.json"];
    if ([fm fileExistsAtPath:tokPath]) {
        NSData *tokData = [NSData dataWithContentsOfFile:tokPath];
        NSDictionary *tokRoot = [NSJSONSerialization JSONObjectWithData:tokData options:0 error:nil];
        NSArray *addedTokens = tokRoot[@"added_tokens"];
        if (addedTokens) {
            for (NSDictionary *tok in addedTokens) {
                NSString *content = tok[@"content"];
                int tid = [tok[@"id"] intValue];
                if ([content isEqualToString:@"<think>"]) cfg.think_start_token = tid;
                else if ([content isEqualToString:@"</think>"]) cfg.think_end_token = tid;
            }
        }
    } else {
        fprintf(stderr, "[config] WARNING: tokenizer.json not found, think tokens disabled\n");
    }

    // Derived dimensions
    cfg.linear_total_key   = cfg.linear_num_k_heads * cfg.linear_key_dim;
    cfg.linear_total_value = cfg.linear_num_v_heads * cfg.linear_value_dim;
    cfg.linear_conv_dim    = cfg.linear_total_key * 2 + cfg.linear_total_value;
    cfg.rotary_dim         = (int)(cfg.head_dim * cfg.partial_rotary);

    // Expert byte offsets
    compute_expert_offsets(&cfg);

    // Summary
    fprintf(stderr, "[config] %d layers (%d linear + %d full), hidden=%d, heads=%d, kv_heads=%d, head_dim=%d\n",
            cfg.num_layers, cfg.num_linear_layers, cfg.num_full_attn_layers,
            cfg.hidden_dim, cfg.num_attn_heads, cfg.num_kv_heads, cfg.head_dim);
    fprintf(stderr, "[config] %d experts (K=%d), moe_intermediate=%d, shared=%d\n",
            cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate, cfg.shared_intermediate);
    fprintf(stderr, "[config] %d-bit quantization, group_size=%d, expert_size=%zu bytes\n",
            cfg.bits, cfg.group_size, cfg.expert_size_4bit);
    fprintf(stderr, "[config] EOS tokens: [");
    for (int i = 0; i < cfg.num_eos_tokens; i++)
        fprintf(stderr, "%s%d", i ? ", " : "", cfg.eos_token_ids[i]);
    fprintf(stderr, "], think: %d/%d\n", cfg.think_start_token, cfg.think_end_token);
}
```

- [ ] **Step 1.4: Add alloc_tracking_arrays()**

This replaces the static arrays that were sized by `#define` constants:

```c
// Dynamic tracking arrays (allocated after config is loaded)
static int *g_expert_freq = NULL;
static uint8_t *g_expert_seen = NULL;
static LZ4IndexEntry **g_lz4_index = NULL;
static uint8_t *g_cache_seen = NULL;
static uint64_t *g_cache_last_touch_token = NULL;
static uint64_t *g_cache_last_evict_token = NULL;
static int *g_pred_experts = NULL;
static int *g_pred_count = NULL;

// Helper macros for flattened 2D access
#define FREQ(l, e)           g_expert_freq[(l) * cfg.num_experts + (e)]
#define EXPERT_SEEN_BYTE(l, e) g_expert_seen[(l) * ((cfg.num_experts + 7) / 8) + ((e) >> 3)]
#define CACHE_SEEN(l, e)     g_cache_seen[(l) * cfg.num_experts + (e)]
#define CACHE_TOUCH(l, e)    g_cache_last_touch_token[(l) * cfg.num_experts + (e)]
#define CACHE_EVICT(l, e)    g_cache_last_evict_token[(l) * cfg.num_experts + (e)]
#define PRED_EXPERT(l, k)    g_pred_experts[(l) * MAX_K + (k)]
#define PRED_COUNT(l)        g_pred_count[(l)]

static void alloc_tracking_arrays(void) {
    int nl = cfg.num_layers;
    int ne = cfg.num_experts;
    int seen_bytes_per_layer = (ne + 7) / 8;

    g_expert_freq           = calloc(nl * ne, sizeof(int));
    g_expert_seen           = calloc(nl * seen_bytes_per_layer, sizeof(uint8_t));
    g_lz4_index             = calloc(nl, sizeof(LZ4IndexEntry *));
    g_cache_seen            = calloc(nl * ne, sizeof(uint8_t));
    g_cache_last_touch_token = calloc(nl * ne, sizeof(uint64_t));
    g_cache_last_evict_token = calloc(nl * ne, sizeof(uint64_t));
    g_pred_experts          = calloc(nl * MAX_K, sizeof(int));
    g_pred_count            = calloc(nl, sizeof(int));
}
```

- [ ] **Step 1.5: Build and verify it compiles**

At this point the old `#define`s still exist (they'll be removed in Task 2). The new code coexists. Build to verify no syntax errors:

```bash
cd metal_infer && make clean && make
```

Expected: compiles successfully (old defines still in use, new code not yet called).

- [ ] **Step 1.6: Commit**

```bash
git add metal_infer/infer.m
git commit -m "feat: add ModelConfig struct and config loader (not yet wired up)"
```

---

## Task 2: Wire up config loading in main() and remove old defines

**Files:**
- Modify: `metal_infer/infer.m` — main() function (line ~6581) and defines block (lines 68-140)

- [ ] **Step 2.1: Call load_model_config() early in main()**

In `main()`, right after CLI arg parsing (after the switch block, around line 6654), add:

```c
        // ---- Load model configuration from HF config.json ----
        load_model_config(model_path);
        alloc_tracking_arrays();
```

This must come BEFORE `metal_setup()` (line 6689) since Metal buffer allocations depend on cfg values.

- [ ] **Step 2.2: Remove MODEL_PATH_DEFAULT, use cfg.model_path**

Change line 6583 from:
```c
const char *model_path = MODEL_PATH_DEFAULT;
```
to:
```c
const char *model_path = NULL;
```

And add a default path fallback in the config loader if model_path is NULL:
```c
if (!model_dir || !model_dir[0]) {
    // Try common HF cache locations
    const char *home = getenv("HOME");
    char probe[1024];
    snprintf(probe, sizeof(probe), "%s/.cache/huggingface/hub", home);
    // ... or just require --model
    fprintf(stderr, "FATAL: --model path required\n");
    exit(1);
}
```

Actually, keep a sensible default: if `--model` not provided, search `~/.cache/huggingface/hub/` for any `models--*Qwen*` directory. If exactly one found, use it. Otherwise, print error asking for `--model`.

- [ ] **Step 2.3: Remove the old #define block (lines 72-140)**

Delete the entire block of `#define`s from `HIDDEN_DIM` through `MODEL_PATH_DEFAULT`. Also remove:
- `#define NUM_FULL_ATTN_LAYERS 10` (line ~1011)
- `#define NUM_LINEAR_LAYERS 30` (line ~1033)

- [ ] **Step 2.4: Remove old static array declarations**

Remove these declarations (they're now in `alloc_tracking_arrays()`):
- `static int g_expert_freq[NUM_LAYERS][NUM_EXPERTS];` (line 206)
- `static uint8_t g_expert_seen[NUM_LAYERS][NUM_EXPERTS / 8];` (line 214)
- `static LZ4IndexEntry *g_lz4_index[NUM_LAYERS];` (line 198)
- `static uint8_t g_cache_seen[NUM_LAYERS][NUM_EXPERTS];` (line 254)
- `static uint64_t g_cache_last_touch_token[NUM_LAYERS][NUM_EXPERTS];` (line 255)
- `static uint64_t g_cache_last_evict_token[NUM_LAYERS][NUM_EXPERTS];` (line 256)
- `static int g_pred_experts[60][MAX_K];` (line 3298)
- `static int g_pred_count[60];` (line 3299)

- [ ] **Step 2.5: Update active_expert_size() to use cfg**

Change:
```c
static inline size_t active_expert_size(void) {
    return g_use_2bit ? EXPERT_SIZE_2BIT : EXPERT_SIZE;
}
```
to:
```c
static inline size_t active_expert_size(void) {
    return g_use_2bit ? cfg.expert_size_2bit : cfg.expert_size_4bit;
}
```

- [ ] **Step 2.6: Attempt build — expect ~960 errors from removed defines**

```bash
cd metal_infer && make 2>&1 | head -50
```

This confirms the scope of replacements needed. Do NOT try to fix yet — just verify the errors are all "use of undeclared identifier" for the removed defines.

- [ ] **Step 2.7: Commit (broken state, WIP)**

```bash
git add metal_infer/infer.m
git commit -m "wip: remove old defines, wire up config loader (broken — refs not yet updated)"
```

---

## Task 3: Bulk replace #define references with cfg.field

**Files:**
- Modify: `metal_infer/infer.m` — ~960 occurrences across the entire file

This is the largest task. Use find-and-replace for each define→cfg mapping. Order matters: replace longer names first to avoid partial matches (e.g., `LINEAR_TOTAL_KEY` before `LINEAR_KEY_DIM`).

- [ ] **Step 3.1: Replace core model dimension defines**

Apply these replacements throughout the file (use replace-all):

| Old | New |
|-----|-----|
| `HIDDEN_DIM` | `cfg.hidden_dim` |
| `NUM_LAYERS` | `cfg.num_layers` |
| `NUM_ATTN_HEADS` | `cfg.num_attn_heads` |
| `NUM_KV_HEADS` | `cfg.num_kv_heads` |
| `HEAD_DIM` | `cfg.head_dim` |
| `VOCAB_SIZE` | `cfg.vocab_size` |
| `RMS_NORM_EPS` | `cfg.rms_norm_eps` |

**CAUTION:** Be careful with `HEAD_DIM` — it must not match inside longer names. Verify no `*_HEAD_DIM` defines exist that would be corrupted.

- [ ] **Step 3.2: Replace MoE defines**

| Old | New |
|-----|-----|
| `NUM_EXPERTS_PER_TOK` | `cfg.num_experts_per_tok` |
| `NUM_EXPERTS` | `cfg.num_experts` |
| `MOE_INTERMEDIATE` | `cfg.moe_intermediate` |
| `SHARED_INTERMEDIATE` | `cfg.shared_intermediate` |
| `GROUP_SIZE` | `cfg.group_size` |
| `BITS` | `cfg.bits` |

**CAUTION:** `NUM_EXPERTS` must not match `NUM_EXPERTS_PER_TOK`. Replace `NUM_EXPERTS_PER_TOK` first, then `NUM_EXPERTS`. Similarly, `BITS` is short — verify it doesn't appear in other contexts.

- [ ] **Step 3.3: Replace linear attention defines**

Replace in this order (longest first):

| Old | New |
|-----|-----|
| `LINEAR_TOTAL_VALUE` | `cfg.linear_total_value` |
| `LINEAR_TOTAL_KEY` | `cfg.linear_total_key` |
| `LINEAR_CONV_DIM` | `cfg.linear_conv_dim` |
| `LINEAR_NUM_V_HEADS` | `cfg.linear_num_v_heads` |
| `LINEAR_NUM_K_HEADS` | `cfg.linear_num_k_heads` |
| `LINEAR_KEY_DIM` | `cfg.linear_key_dim` |
| `LINEAR_VALUE_DIM` | `cfg.linear_value_dim` |
| `CONV_KERNEL_SIZE` | `cfg.conv_kernel_size` |

- [ ] **Step 3.4: Replace RoPE and full attention defines**

| Old | New |
|-----|-----|
| `FULL_ATTN_INTERVAL` | — (see Step 3.8 for formula replacement) |
| `ROPE_THETA` | `cfg.rope_theta` |
| `PARTIAL_ROTARY` | `cfg.partial_rotary` |
| `ROTARY_DIM` | `cfg.rotary_dim` |

- [ ] **Step 3.5: Replace expert offset defines**

4-bit offsets:

| Old | New |
|-----|-----|
| `EXPERT_SIZE` (but NOT `EXPERT_SIZE_2BIT`) | `cfg.expert_size_4bit` |
| `GATE_W_OFF_4` | `cfg.gate_w_off_4` |
| `GATE_S_OFF_4` | `cfg.gate_s_off_4` |
| `GATE_B_OFF_4` | `cfg.gate_b_off_4` |
| `UP_W_OFF_4` | `cfg.up_w_off_4` |
| `UP_S_OFF_4` | `cfg.up_s_off_4` |
| `UP_B_OFF_4` | `cfg.up_b_off_4` |
| `DOWN_W_OFF_4` | `cfg.down_w_off_4` |
| `DOWN_S_OFF_4` | `cfg.down_s_off_4` |
| `DOWN_B_OFF_4` | `cfg.down_b_off_4` |

2-bit offsets:

| Old | New |
|-----|-----|
| `EXPERT_SIZE_2BIT` | `cfg.expert_size_2bit` |
| `GATE_W_OFF_2` | `cfg.gate_w_off_2` |
| ... (same pattern for all 2-bit offsets) |

- [ ] **Step 3.6: Replace special token defines**

| Old | New |
|-----|-----|
| `EOS_TOKEN_1` | `cfg.eos_token_ids[0]` |
| `EOS_TOKEN_2` | `cfg.eos_token_ids[1]` |
| `THINK_START_TOKEN` | `cfg.think_start_token` |
| `THINK_END_TOKEN` | `cfg.think_end_token` |
| `MAX_SEQ_LEN` | `cfg.max_seq_len` |

- [ ] **Step 3.7: Replace struct-local defines**

| Old | New |
|-----|-----|
| `NUM_FULL_ATTN_LAYERS` | `cfg.num_full_attn_layers` |
| `NUM_LINEAR_LAYERS` | `cfg.num_linear_layers` |

- [ ] **Step 3.8: Replace FULL_ATTN_INTERVAL formula patterns**

Find all occurrences of the formula pattern `(i + 1) % FULL_ATTN_INTERVAL == 0` (or variants with different variable names like `layer_idx`) and replace with `cfg.is_full_attn[i]`.

Also replace index computation formulas:
- `(layer_idx + 1) / FULL_ATTN_INTERVAL - 1` → `cfg.full_attn_index[layer_idx]`
- `layer_idx - (layer_idx + 1) / FULL_ATTN_INTERVAL` → `cfg.linear_index[layer_idx]`

- [ ] **Step 3.9: Build and fix any remaining errors**

```bash
cd metal_infer && make 2>&1 | head -100
```

Fix any remaining compilation errors from the replacements. Common issues:
- `sizeof()` on old arrays that are now pointers (need explicit size)
- `memset()` on old arrays (need explicit size calculation)
- Places where `EXPERT_SIZE` was used generically (should use `active_expert_size()`)

- [ ] **Step 3.10: Commit**

```bash
git add metal_infer/infer.m
git commit -m "feat: replace all model #defines with cfg.* struct fields (~960 occurrences)"
```

---

## Task 4: Convert MetalCtx fixed arrays to dynamic allocation

**Files:**
- Modify: `metal_infer/infer.m` — MetalCtx struct (~line 1011) and metal_setup() (~line 1049)

- [ ] **Step 4.1: Change MetalCtx arrays to pointers**

In the `MetalCtx` struct, change:
```c
id<MTLBuffer> buf_kv_k[NUM_FULL_ATTN_LAYERS];
id<MTLBuffer> buf_kv_v[NUM_FULL_ATTN_LAYERS];
```
to:
```c
id<MTLBuffer> *buf_kv_k;
id<MTLBuffer> *buf_kv_v;
```

And:
```c
id<MTLBuffer> buf_delta_state[NUM_LINEAR_LAYERS];
id<MTLBuffer> buf_conv_state[NUM_LINEAR_LAYERS];
```
to:
```c
id<MTLBuffer> *buf_delta_state;
id<MTLBuffer> *buf_conv_state;
```

- [ ] **Step 4.2: Allocate in metal_setup()**

In `metal_setup()`, after creating the Metal device but before buffer allocation, add:

```c
ctx->buf_kv_k       = calloc(cfg.num_full_attn_layers, sizeof(id<MTLBuffer>));
ctx->buf_kv_v       = calloc(cfg.num_full_attn_layers, sizeof(id<MTLBuffer>));
ctx->buf_delta_state = calloc(cfg.num_linear_layers, sizeof(id<MTLBuffer>));
ctx->buf_conv_state  = calloc(cfg.num_linear_layers, sizeof(id<MTLBuffer>));
```

- [ ] **Step 4.3: Convert stack VLAs in serve loop to malloc**

In the serve function (~line 6160), change:
```c
void *gpu_delta_snapshots[NUM_LINEAR_LAYERS];
void *gpu_conv_snapshots[NUM_LINEAR_LAYERS];
```
to:
```c
void **gpu_delta_snapshots = calloc(cfg.num_linear_layers, sizeof(void *));
void **gpu_conv_snapshots  = calloc(cfg.num_linear_layers, sizeof(void *));
```

Add `free(gpu_delta_snapshots); free(gpu_conv_snapshots);` at function exit/cleanup.

- [ ] **Step 4.4: Convert main() stack arrays to use cfg**

In main() (~line 6801), change:
```c
int layer_fds[NUM_LAYERS];
int layer_fds_cold[NUM_LAYERS];
void *layer_mmaps[NUM_LAYERS];
size_t layer_mmap_sizes[NUM_LAYERS];
```
to VLAs using `cfg.num_layers` (C99 VLAs are fine here since main() runs after config load):
```c
int layer_fds[cfg.num_layers];
int layer_fds_cold[cfg.num_layers];
void *layer_mmaps[cfg.num_layers];
size_t layer_mmap_sizes[cfg.num_layers];
```

Or use malloc if compiler doesn't support VLAs in ObjC.

- [ ] **Step 4.5: Update LayerWeightCache array**

Change:
```c
static LayerWeightCache layer_cache[NUM_LAYERS];
```
to:
```c
static LayerWeightCache *layer_cache = NULL;
```

And in `alloc_tracking_arrays()` add:
```c
layer_cache = calloc(cfg.num_layers, sizeof(LayerWeightCache));
```

- [ ] **Step 4.6: Fix memset/sizeof on converted arrays**

Find all `memset(g_expert_seen, 0, sizeof(g_expert_seen))` and similar calls that relied on compile-time sizeof. Replace with explicit size:
```c
memset(g_expert_seen, 0, cfg.num_layers * ((cfg.num_experts + 7) / 8));
```

Same for `g_cache_seen`, `g_cache_last_touch_token`, `g_cache_last_evict_token`.

- [ ] **Step 4.7: Update expert_is_seen / expert_mark_seen helpers**

Change from direct 2D array access to flattened access using the helper macros:
```c
static inline int expert_is_seen(int layer, int expert) {
    return (EXPERT_SEEN_BYTE(layer, expert) >> (expert & 7)) & 1;
}
static inline void expert_mark_seen(int layer, int expert) {
    EXPERT_SEEN_BYTE(layer, expert) |= (1 << (expert & 7));
}
```

- [ ] **Step 4.8: Update g_pred_experts / g_pred_count access**

Replace all `g_pred_experts[layer_idx][k]` with `PRED_EXPERT(layer_idx, k)` and `g_pred_count[layer_idx]` with `PRED_COUNT(layer_idx)`.

- [ ] **Step 4.9: Update all g_expert_freq, g_cache_seen access patterns**

Replace all `g_expert_freq[layer][expert]` with `FREQ(layer, expert)` etc. throughout the file.

- [ ] **Step 4.10: Build and verify**

```bash
cd metal_infer && make clean && make
```

Expected: compiles with 0 errors, 0 warnings.

- [ ] **Step 4.11: Commit**

```bash
git add metal_infer/infer.m
git commit -m "feat: convert fixed-size arrays to dynamic allocation"
```

---

## Task 5: Validate with current model

**Files:** None (testing only)

- [ ] **Step 5.1: Run inference with 35B model**

```bash
cd metal_infer && ./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit --prompt "What is 2+2?" --tokens 20
```

Verify:
1. Config summary prints correctly to stderr
2. No NaN values
3. Coherent output
4. Same token/s as before (~4.7 tok/s)

- [ ] **Step 5.2: Run server mode**

```bash
cd metal_infer && ./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit --serve 8000
```

In another terminal:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

Verify: no bus error, coherent response.

- [ ] **Step 5.3: Run with --timing flag**

```bash
cd metal_infer && ./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit --prompt "Hello" --tokens 10 --timing
```

Verify: timing breakdown looks normal, no regressions.

- [ ] **Step 5.4: Commit validation**

```bash
git add metal_infer/infer.m
git commit -m "feat: runtime model config — validated with Qwen3.5-35B-A3B"
```

---

## Task 6: Clean up and update header comment

**Files:**
- Modify: `metal_infer/infer.m:1-43` (header comment)

- [ ] **Step 6.1: Update file header comment**

Replace the hardcoded model description in the header (lines 1-43) with a generic description that mentions runtime config loading:

```c
/*
 * infer.m — Qwen3.5 MoE inference engine using Metal
 *
 * Full forward pass: embedding -> N transformer layers -> norm -> lm_head -> sample
 * Model architecture loaded at runtime from HuggingFace config.json (--model flag).
 * Non-expert weights loaded from model_weights.bin (mmap'd at startup)
 * Expert weights loaded from packed_experts/ per layer per token (pread)
 *
 * Supported models: Qwen3.5-35B-A3B, Qwen3.5-397B-A17B, and compatible MoE variants
 * ...
```

- [ ] **Step 6.2: Update the startup banner in main()**

Change `printf("=== Qwen3.5-35B-A3B Metal Inference Engine ===\n")` to dynamically show the model info:

```c
printf("=== Flash-MoE Metal Inference Engine ===\n");
printf("Config:   %s/config.json\n", cfg.model_path);
```

- [ ] **Step 6.3: Final build and test**

```bash
cd metal_infer && make clean && make && ./infer --prompt "Hello" --tokens 5
```

- [ ] **Step 6.4: Commit**

```bash
git add metal_infer/infer.m
git commit -m "chore: update header and banner for runtime model config"
```
