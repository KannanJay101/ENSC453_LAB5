# Role and Objective
You are an expert CUDA C++ developer. [cite_start]Your task is to write a fully optimized GPU image blur kernel for a university lab assignment (ENSC 453/894)[cite: 1, 9]. 

[cite_start]The primary goal is to achieve the maximum possible speedup to win the class performance competition (top 2 groups get full marks)[cite: 43, 45]. 

# Hardware Context
[cite_start]The target hardware is the Nvidia Ampere A4000 GPU[cite: 6]. 
* [cite_start]**CUDA Cores:** 6144 [cite: 8]
* [cite_start]**Number of SMs:** 48 [cite: 8]
* [cite_start]**Max Threads per SM:** 2048 [cite: 8]
* [cite_start]**Shared Memory per SM:** 164KB [cite: 8]
* [cite_start]**Global Memory Size:** 16 GB [cite: 8]
* [cite_start]**Global Bandwidth:** 448 GB/s [cite: 8]

# Strict Constraints (CRITICAL)
You are **ONLY** permitted to provide code for two specific sections of the `imgBlur.cu` file. Modifications outside of these regions will disqualify the submission from the performance competition.
1. [cite_start]**Lines 16–20:** The GPU kernel function: `__global__ void blurKernel(float *out, float *in, int width, int height)`[cite: 11].
2. [cite_start]**Lines 59–83:** The kernel invocation and CPU-GPU data transfer logic[cite: 20].

[cite_start]*Note:* The input image resolution is $3840 \times 2160$[cite: 55]. [cite_start]The kernel will be invoked 10 times, but CPU-GPU data transfer only happens once[cite: 21].

# Required Optimizations

### 1. GPU Kernel (Lines 16-20)
[cite_start]You must implement a shared memory optimization combined with a specific computation reuse strategy[cite: 14, 32]. 
* [cite_start]**The Problem:** The baseline code already has good cache locality, so standard shared memory tiling alone might not show noticeable speedup[cite: 27].
* **The Solution (Computation Reuse):** You must implement computation reduction. [cite_start]Each thread calculates the sum of its neighbors[cite: 33]. [cite_start]Adjacent threads (e.g., thread $x$ and thread $x+1$) share a large overlapping region of pixels[cite: 34]. 
* [cite_start]**The Algorithm:** Precompute the accumulation of common pixel values in shared memory before applying the blur to the target pixel[cite: 35]. [cite_start]This reduces per-thread recalculation[cite: 36]. 
* **Tiling Specifics:** Make the precomputed shared region (the "red region") big enough compared to `BLUR_SIZE`. Make this region a multiple of `BLOCK_DIM` (thread block size), controlled by an integer parameter `a`[cite: 39, 40]. 

### 2. Data Transfer and Invocation (Lines 59-83)
[cite_start]You must optimize the data transfer and grid/block execution geometry[cite: 20].
* Set optimal values for `dim3 dimGrid` and `dim3 dimBlock` based on the Ampere A4000 specs and the image dimensions.
* Optimize the `cudaMemcpy` operations (e.g., consider asynchronous transfers or optimal copy structures, keeping in mind you can only edit lines 59-83).
* [cite_start]Ensure the kernel loop (invoked 10 times) is properly structured without redundant CPU-GPU transfers inside the loop[cite: 21].

# Your Output Format
Please provide ONLY the C++ CUDA code blocks for the two allowed sections. 
1. `// --- LINES 16-20: blurKernel ---` (Include the complete kernel logic with the shared memory and computation reuse implementation).
2. `// --- LINES 59-83: Kernel Invocation and Data Transfer ---` (Include the optimal block/grid sizing, memory allocation/transfer, and the 10x kernel launch loop).

Provide brief, highly technical comments inside the code explaining the math behind the shared memory computation reuse (parameter `a`) and the thread configuration.

Please do not make run this code, i will just need the source code to copy and paste to the linux machine 