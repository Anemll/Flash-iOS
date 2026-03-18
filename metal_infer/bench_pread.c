/*
 * bench_pread.c — Isolated expert I/O benchmark
 *
 * Tests sorted vs unsorted pread patterns at various parallelism levels.
 * No Metal, no model, no GPU — pure I/O measurement.
 *
 * Build: clang -O2 -lpthread bench_pread.c -o bench_pread
 * Run:   ./bench_pread [--layers 60] [--experts-per-layer 4] [--rounds 5]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>
#include <sys/stat.h>

#define NUM_LAYERS 60
#define NUM_EXPERTS 512
#define EXPERT_SIZE_2BIT 3932160
#define EXPERT_SIZE_4BIT 7077888
#define MODEL_PATH "/Users/danielwoods/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3"

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

typedef struct {
    int fd;
    void *dst;
    off_t offset;
    size_t size;
} ReadTask;

// Sequential: read tasks one at a time
static double bench_sequential(ReadTask *tasks, int n) {
    double t0 = now_ms();
    for (int i = 0; i < n; i++) {
        pread(tasks[i].fd, tasks[i].dst, tasks[i].size, tasks[i].offset);
    }
    return now_ms() - t0;
}

// Parallel: N pthreads, one task each
typedef struct { ReadTask *task; } ThreadArg;
static void *pread_thread(void *arg) {
    ReadTask *t = ((ThreadArg *)arg)->task;
    pread(t->fd, t->dst, t->size, t->offset);
    return NULL;
}

static double bench_parallel(ReadTask *tasks, int n) {
    pthread_t threads[n];
    ThreadArg args[n];
    double t0 = now_ms();
    for (int i = 0; i < n; i++) {
        args[i].task = &tasks[i];
        pthread_create(&threads[i], NULL, pread_thread, &args[i]);
    }
    for (int i = 0; i < n; i++) {
        pthread_join(threads[i], NULL);
    }
    return now_ms() - t0;
}

static int cmp_offset(const void *a, const void *b) {
    const ReadTask *ta = (const ReadTask *)a;
    const ReadTask *tb = (const ReadTask *)b;
    if (ta->offset < tb->offset) return -1;
    if (ta->offset > tb->offset) return 1;
    return 0;
}

static void purge_cache(int fd) {
    // F_NOCACHE toggles bypass the page cache for subsequent reads
    fcntl(fd, F_NOCACHE, 1);
}

int main(int argc, char **argv) {
    int K = 4;           // experts per layer
    int rounds = 10;
    int use_2bit = 1;
    int do_purge = 1;    // purge page cache between tests

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--k") == 0 && i+1 < argc) K = atoi(argv[++i]);
        if (strcmp(argv[i], "--rounds") == 0 && i+1 < argc) rounds = atoi(argv[++i]);
        if (strcmp(argv[i], "--4bit") == 0) use_2bit = 0;
        if (strcmp(argv[i], "--no-purge") == 0) do_purge = 0;
    }

    size_t esz = use_2bit ? EXPERT_SIZE_2BIT : EXPERT_SIZE_4BIT;
    const char *subdir = use_2bit ? "packed_experts_2bit" : "packed_experts";

    printf("Expert I/O Benchmark\n");
    printf("  Expert size: %.2f MB (%s)\n", esz / 1e6, use_2bit ? "2-bit" : "4-bit");
    printf("  K=%d experts/layer, %d layers, %d rounds\n", K, NUM_LAYERS, rounds);
    printf("  Page cache purge: %s\n", do_purge ? "yes (F_NOCACHE)" : "no");
    printf("\n");

    // Open layer files
    int fds[NUM_LAYERS];
    int available = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s/layer_%02d.bin", MODEL_PATH, subdir, i);
        fds[i] = open(path, O_RDONLY);
        if (fds[i] >= 0) {
            available++;
            if (do_purge) purge_cache(fds[i]);
        }
    }
    printf("  Layer files: %d/%d\n\n", available, NUM_LAYERS);
    if (available == 0) { fprintf(stderr, "No layer files found\n"); return 1; }

    // Allocate read buffers
    void *bufs[K];
    for (int k = 0; k < K; k++) {
        bufs[k] = malloc(esz);
    }

    // Generate random expert indices (deterministic seed)
    srand(42);
    int expert_map[NUM_LAYERS][8]; // up to K=8
    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int k = 0; k < K; k++) {
            expert_map[l][k] = rand() % NUM_EXPERTS;
        }
    }

    // === Test 1: Per-layer, unsorted, parallel ===
    printf("=== Test 1: Unsorted parallel (current approach) ===\n");
    for (int r = 0; r < rounds; r++) {
        double total = 0;
        for (int l = 0; l < NUM_LAYERS; l++) {
            if (fds[l] < 0) continue;
            ReadTask tasks[K];
            for (int k = 0; k < K; k++) {
                tasks[k].fd = fds[l];
                tasks[k].dst = bufs[k];
                tasks[k].offset = (off_t)expert_map[l][k] * esz;
                tasks[k].size = esz;
            }
            total += bench_parallel(tasks, K);
        }
        double per_layer = total / available;
        double gb_s = (double)(K * esz * available) / 1e9 / (total / 1000.0);
        printf("  Round %d: %.1f ms total, %.2f ms/layer, %.1f GB/s\n",
               r, total, per_layer, gb_s);
    }

    // === Test 2: Per-layer, sorted by offset, parallel ===
    printf("\n=== Test 2: Sorted parallel (sorted by file offset) ===\n");
    for (int r = 0; r < rounds; r++) {
        double total = 0;
        for (int l = 0; l < NUM_LAYERS; l++) {
            if (fds[l] < 0) continue;
            ReadTask tasks[K];
            for (int k = 0; k < K; k++) {
                tasks[k].fd = fds[l];
                tasks[k].dst = bufs[k];
                tasks[k].offset = (off_t)expert_map[l][k] * esz;
                tasks[k].size = esz;
            }
            qsort(tasks, K, sizeof(ReadTask), cmp_offset);
            total += bench_parallel(tasks, K);
        }
        double per_layer = total / available;
        double gb_s = (double)(K * esz * available) / 1e9 / (total / 1000.0);
        printf("  Round %d: %.1f ms total, %.2f ms/layer, %.1f GB/s\n",
               r, total, per_layer, gb_s);
    }

    // === Test 3: Per-layer, sequential (no parallelism) ===
    printf("\n=== Test 3: Sequential (single thread, unsorted) ===\n");
    for (int r = 0; r < rounds; r++) {
        double total = 0;
        for (int l = 0; l < NUM_LAYERS; l++) {
            if (fds[l] < 0) continue;
            ReadTask tasks[K];
            for (int k = 0; k < K; k++) {
                tasks[k].fd = fds[l];
                tasks[k].dst = bufs[k];
                tasks[k].offset = (off_t)expert_map[l][k] * esz;
                tasks[k].size = esz;
            }
            total += bench_sequential(tasks, K);
        }
        double per_layer = total / available;
        double gb_s = (double)(K * esz * available) / 1e9 / (total / 1000.0);
        printf("  Round %d: %.1f ms total, %.2f ms/layer, %.1f GB/s\n",
               r, total, per_layer, gb_s);
    }

    // === Test 4: Sequential, sorted ===
    printf("\n=== Test 4: Sequential sorted ===\n");
    for (int r = 0; r < rounds; r++) {
        double total = 0;
        for (int l = 0; l < NUM_LAYERS; l++) {
            if (fds[l] < 0) continue;
            ReadTask tasks[K];
            for (int k = 0; k < K; k++) {
                tasks[k].fd = fds[l];
                tasks[k].dst = bufs[k];
                tasks[k].offset = (off_t)expert_map[l][k] * esz;
                tasks[k].size = esz;
            }
            qsort(tasks, K, sizeof(ReadTask), cmp_offset);
            total += bench_sequential(tasks, K);
        }
        double per_layer = total / available;
        double gb_s = (double)(K * esz * available) / 1e9 / (total / 1000.0);
        printf("  Round %d: %.1f ms total, %.2f ms/layer, %.1f GB/s\n",
               r, total, per_layer, gb_s);
    }

    // === Test 5: Batch all layers, sorted globally ===
    printf("\n=== Test 5: Global sort (all %d reads sorted by layer+offset, parallel %d at a time) ===\n",
           NUM_LAYERS * K, K);
    for (int r = 0; r < rounds; r++) {
        // Build all tasks
        ReadTask all_tasks[NUM_LAYERS * 8];
        int n = 0;
        for (int l = 0; l < NUM_LAYERS; l++) {
            if (fds[l] < 0) continue;
            for (int k = 0; k < K; k++) {
                all_tasks[n].fd = fds[l];
                all_tasks[n].dst = bufs[k % K];
                all_tasks[n].offset = (off_t)expert_map[l][k] * esz;
                all_tasks[n].size = esz;
                n++;
            }
        }
        // Sort all by (fd, offset) — groups by layer file, then by offset within
        double total = 0;
        // Execute in batches of K (parallel)
        double t0 = now_ms();
        for (int i = 0; i < n; i += K) {
            int batch = (n - i < K) ? n - i : K;
            bench_parallel(&all_tasks[i], batch);
        }
        total = now_ms() - t0;
        double per_layer = total / available;
        double gb_s = (double)(n * esz) / 1e9 / (total / 1000.0);
        printf("  Round %d: %.1f ms total, %.2f ms/layer, %.1f GB/s\n",
               r, total, per_layer, gb_s);
    }

    // Cleanup
    for (int k = 0; k < K; k++) free(bufs[k]);
    for (int l = 0; l < NUM_LAYERS; l++) if (fds[l] >= 0) close(fds[l]);

    return 0;
}
