# ENSC 453/894 — Lab 5: CUDA Image Blur Optimization Report

**GPU:** Nvidia Ampere A4000 (48 SMs, 128 cores/SM, 164 KB shared mem/SM, 2048 threads/SM)
**Image:** 3840 × 2160 (grayscale), **BLUR_SIZE = 21** (43×43 window = 1,849 neighbours/pixel)
**Kernel repeats:** 10 per run (timing includes data transfer)

---

## 1. Shared Memory Optimization

### 1.1 Strategy — Separable Row/Column Sum Reuse

The baseline kernel performs **O((2R+1)²) = O(1,849)** global memory reads per output pixel.
The optimized kernel exploits the fact that vertically adjacent output pixels share
(2R+1)−1 = 42 out of 43 rows of their blur window. The optimization proceeds in three
phases, all operating on shared memory:

| Phase | Work | Purpose |
|-------|------|---------|
| **Phase 1 — Tile Load** | Each block cooperatively loads a tile of (OUTPUT\_DIM + 2·BLUR\_SIZE)² pixels from global memory into shared memory, zero-padding out-of-bounds pixels. | Converts expensive global reads to fast shared-memory reads. |
| **Phase 2 — Row Sums** | For every tile row and every output-column position, accumulate (2·BLUR\_SIZE+1) = 43 consecutive elements horizontally and store the result. | Each row sum is reused by up to 43 output pixels that share the same row. |
| **Phase 3 — Column Accumulation** | For each output pixel, sum 43 row sums vertically and divide by an analytically computed neighbour count. | Final blur value with O(2·(2R+1)) = O(86) work per pixel instead of O(1,849). |

The neighbour count for boundary pixels is computed analytically as
`(x_end − x_start + 1) × (y_end − y_start + 1)`, avoiding a second shared-memory count array.

### 1.2 Tile Size Analysis

The tile multiplier **I** (TILE\_I) controls how many output pixels each thread block produces.
Each block of 16×16 = 256 threads outputs an (I·16)² region; each thread handles I² pixels.

| TILE\_I | Output Region | Tile Size | Shared Mem / Block | Max Blocks/SM | Occupancy | Avg Time (s) | Speedup vs Baseline |
|---------|--------------|-----------|-------------------|---------------|-----------|-------------|-------------------|
| 1 | 16 × 16 | 58 × 58 | 17,168 B (16.8 KB) | 8 (thread-limited) | 100% | 0.0373 | 33.4× |
| 2 | 32 × 32 | 74 × 74 | 31,376 B (30.6 KB) | 5 (smem-limited) | 62.5% | 0.0317 | 39.3× |
| **3** | **48 × 48** | **90 × 90** | **49,680 B (48.5 KB)** | **3 (smem-limited)** | **37.5%** | **0.0300** | **41.5×** |
| 4 | 64 × 64 | 106 × 106 | 72,080 B (70.4 KB) | 2 (smem-limited) | 25% | 0.0308 | 40.5× |

**Best tile size: TILE\_I = 3** (0.0300 s). Despite having only 37.5% occupancy, the
larger output region amortizes the halo overhead (the ratio of useful output pixels
to total tile pixels is 48²/90² = 28.4%, vs 16²/58² = 7.6% for I=1). At TILE\_I = 4,
the halo ratio improves further (36.5%) but occupancy drops to 25%, and the performance
benefit plateaus — the SM can no longer hide memory latency with so few active warps.

---

## 2. Thread Block Size Analysis

Tested on the naive kernel (global memory, no tiling) with pinned memory transfers and `-O3 -use_fast_math`:

| Block Size | Threads/Block | Blocks/SM (thread limit) | Full Occupancy? | Avg Time (s) |
|-----------|---------------|--------------------------|-----------------|-------------|
| 8 × 8 | 64 | 32 (max-block limited) | 2,048/2,048 = 100% | 0.2184 |
| **16 × 16** | **256** | **8** | **2,048/2,048 = 100%** | **0.2152** |
| 32 × 32 | 1,024 | 2 | 2,048/2,048 = 100% | 0.2185 |
| 32 × 8 | 256 | 8 | 2,048/2,048 = 100% | 0.2155 |
| 16 × 8 | 128 | 16 | 2,048/2,048 = 100% | 0.2160 |

All configurations achieve 100% occupancy on the A4000. The **16×16** block is fastest
because: (1) the square shape maximises 2D spatial locality for the blur window,
(2) 256 threads per block provides enough warps (8) for good latency hiding while
leaving headroom for register/shared-memory usage, and (3) it avoids the excessive
register pressure of 1,024-thread blocks (32×32).

---

## 3. CPU–GPU Data Transfer Optimization

### Changes from Baseline
1. **`cudaFree(0)` warmup** — forces CUDA context initialization before timing begins
   (baseline includes ~1 s of context init + `cudaMalloc` in the timed section).
2. **Pinned host memory** — `cudaMallocHost` replaces pageable `malloc` buffers,
   enabling direct DMA transfers between CPU and GPU over PCIe.
3. **Compiler flags** — `-O3 -use_fast_math` added to `nvcc` (baseline had no optimization flags).

### Results

| Configuration | Avg Time (s) | Speedup vs Baseline |
|--------------|-------------|-------------------|
| **Baseline** (pageable mem, no warmup, no `-O3`) | 1.246 | 1.00× |
| **Data Transfer Optimized** (pinned mem, warmup, `-O3`) | 0.215 | 5.79× |

The 5.79× speedup comes primarily from removing CUDA context initialization overhead
from the timed region (~1 s) and from faster DMA transfers with pinned memory.
The kernel itself is unchanged; the ~0.215 s is dominated by 10 runs of the naive
kernel (each touching 2160 × 3840 × 1,849 ≈ 15.3 billion global memory reads).

---

## 4. Control Divergence Analysis

### 4.1 Naive Kernel (baseline / data\_transfer-OPUS)

**Grid:** 135 × 240 = 32,400 blocks (each 16×16), **259,200 warps** total.

| Divergence Source | Where | Impact |
|-------------------|-------|--------|
| Outer boundary check `if (x >= width \|\| y >= height)` | Grid perfectly tiles the image (135×16 = 2,160; 240×16 = 3,840) | **0 divergent warps** |
| Inner-loop clamping `if (nx >= 0 && nx < width && ny >= 0 && ny < height)` | Threads within 21 px of any edge clamp differently than interior threads | See below |

**Inner-loop divergence calculation:**

- **Divergent block columns:** 0–1 (left edge, x < 21) and 133–134 (right edge, x > 2138) = 4 columns
- **Divergent block rows:** 0–1 (top) and 238–239 (bottom) = 4 rows
- Interior (non-divergent) blocks: (135 − 4) × (240 − 4) = 30,916
- **Divergent blocks:** 32,400 − 30,916 = **1,484**
- **Upper-bound divergent warps:** 1,484 × 8 = **11,872 / 259,200 = 4.58%**

### 4.2 Shared Memory Kernel (shared\_memory-OPUS, TILE\_I = 2)

**Grid:** 68 × 120 = 8,160 blocks, **65,280 warps** total.

| Phase | Conditionals | Divergence |
|-------|-------------|------------|
| Phase 1 — Tile Load | Ternary `(inBounds) ? pixel : 0.0f` at edges | ~372 edge blocks / 8,160 = **~4.6%** (predicated, low cost) |
| Phase 2 — Row Sums | Fixed loop `k = 0 .. 42`, no conditionals | **0%** |
| Phase 3 — Column Sums | Output boundary check at rightmost column only; 16-pixel alignment ensures entire warps take the same path | **~0%** |

**Effective computation divergence ≈ 0%.** The separable design eliminates all
data-dependent branches from the inner computation loops. Only tile loading at
image edges incurs minor predicated divergence, which has negligible performance impact.

---

## 5. Overall Performance Summary

| Branch | Kernel | Data Transfer | Avg Time (s) | Speedup vs Baseline |
|--------|--------|--------------|-------------|-------------------|
| `baseline` | Naive (global mem) | Pageable | 1.246 | 1.00× |
| `data_transfer-OPUS` | Naive (global mem) | Pinned + warmup | 0.215 | 5.79× |
| `shared_memory-OPUS` (TILE\_I=3) | Shared mem + separable reuse | Pinned + warmup | 0.030 | 41.5× |

### Speedup Breakdown

```
Baseline ──(5.79×)──▶ Data Transfer ──(7.17×)──▶ Shared Memory
           pinned mem + warmup + -O3      separable computation reuse
                                          in shared memory

Total: 41.5× speedup over baseline
```

The shared memory kernel's 7.17× speedup over the naive kernel (both with pinned
memory) comes from two factors:

1. **Arithmetic reduction:** O(86) vs O(1,849) operations per pixel — a 21.5× reduction
   in computation, the dominant contributor.
2. **Memory hierarchy:** All intermediate reads from shared memory (~19 TB/s bandwidth)
   vs global memory (~448 GB/s) — a 42× bandwidth advantage.

The best configuration is **TILE\_I = 3, BLOCK\_SIZE = 16×16**, achieving **0.030 s**
total execution time including data transfer on the Nvidia A4000.
