// MTT S4000 (Moore Threads MUSA) smoke verification for the project's
// three-stage parallel f32 semigroup scan kernels (verbatim from
// crates/cuda/kernels/scan_parallel.cu, compiled by mcc instead of NVRTC).
//
// Host uses the MUSA runtime API (musaMalloc / musaMemcpy / musaLaunchKernel),
// the MUSA equivalent of the CUDA runtime. Numerics are checked against an
// independent scalar reference with the same tolerance contract as the project
// (abs 1e-3 / rel 1e-5 for the 16K/64-block case, per P*eps*||B|| bound).
#include <musa_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define LLM_SCAN_MAXN 256
#define LLM_SCAN_BLOCKDIM 256

// --- project kernels, verbatim from crates/cuda/kernels/scan_parallel.cu ---
extern "C" __global__ void scan_agg(const float* input, const float* a,
                                    const float* b, const float* c,
                                    const float* skip, const float* init,
                                    float* block_total, unsigned n, unsigned d,
                                    unsigned N, unsigned chunk, unsigned blocks) {
    const unsigned block = blockIdx.x;
    const unsigned i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= d) {
        return;
    }
    const unsigned t0 = block * chunk;
    const unsigned t1 = (t0 + chunk < n) ? (t0 + chunk) : n;
    const unsigned dN = d * N;
    const unsigned base = block * (2u * dN);
    for (unsigned j = 0; j < N; ++j) {
        const unsigned idx = i * N + j;
        float A = 1.0f;
        float B = 0.0f;
        const float aj = a[idx];
        const float bj = b[idx];
        for (unsigned t = t0; t < t1; ++t) {
            const float x = input[t * d + i];
            B = aj * B + bj * x;
            A = aj * A;
        }
        block_total[base + idx] = A;
        block_total[base + dN + idx] = B;
    }
}

extern "C" __global__ void scan_prefix(const float* block_total,
                                       const float* init, float* state,
                                       float* block_prefix, unsigned n,
                                       unsigned d, unsigned N, unsigned chunk,
                                       unsigned blocks) {
    const unsigned dN = d * N;
    for (unsigned j = threadIdx.x; j < dN; j += blockDim.x) {
        block_prefix[j] = 1.0f;
        block_prefix[dN + j] = 0.0f;
        float A = 1.0f;
        float B = 0.0f;
        for (unsigned q = 0; q < blocks; ++q) {
            const unsigned base = q * (2u * dN);
            const float aq = block_total[base + j];
            const float bq = block_total[base + dN + j];
            block_prefix[(q + 1u) * (2u * dN) + j] = A * aq;
            block_prefix[(q + 1u) * (2u * dN) + dN + j] = fmaf(aq, B, bq);
            B = fmaf(aq, B, bq);
            A = A * aq;
        }
        state[j] = A * init[j] + B;
    }
}

extern "C" __global__ void scan_apply(const float* input, const float* a,
                                      const float* b, const float* c,
                                      const float* skip, const float* init,
                                      const float* block_prefix, float* out,
                                      unsigned n, unsigned d, unsigned N,
                                      unsigned chunk, unsigned blocks) {
    const unsigned block = blockIdx.x;
    const unsigned i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= d) {
        return;
    }
    const unsigned t0 = block * chunk;
    const unsigned t1 = (t0 + chunk < n) ? (t0 + chunk) : n;
    const unsigned dN = d * N;
    const unsigned base = block * (2u * dN);
    float h[LLM_SCAN_MAXN];
    for (unsigned j = 0; j < N; ++j) {
        const unsigned idx = i * N + j;
        h[j] = block_prefix[base + idx] * init[idx]
            + block_prefix[base + dN + idx];
    }
    for (unsigned t = t0; t < t1; ++t) {
        const float x = input[t * d + i];
        float acc = 0.0f;
        for (unsigned j = 0; j < N; ++j) {
            const unsigned idx = i * N + j;
            h[j] = a[idx] * h[j] + b[idx] * x;
            acc += c[idx] * h[j];
        }
        out[t * d + i] = acc + skip[i] * x;
    }
}

// --- deterministic LCG fill (fixed seed, same as project test convention) ---
static float* fill_f32(unsigned n, unsigned seed, float lo, float hi) {
    float* v = (float*)malloc(n * sizeof(float));
    unsigned s = seed;
    for (unsigned k = 0; k < n; ++k) {
        s = s * 1103515245u + 12345u;
        const double u = (double)(s >> 8) / 16777216.0;
        v[k] = (float)(lo + u * (hi - lo));
    }
    return v;
}

static int check(const char* err) {
    musaError_t e = musaGetLastError();
    if (e != musaSuccess) {
        fprintf(stderr, "MUSA error %d at %s: %s\n", (int)e, err, musaGetErrorString(e));
        return 1;
    }
    return 0;
}

int main(void) {
    int dev_count = 0;
    musaGetDeviceCount(&dev_count);
    printf("MUSA devices: %d\n", dev_count);
    if (dev_count < 1) { printf("FAIL: no MUSA device\n"); return 1; }
    musaDeviceProp prop;
    musaGetDeviceProperties(&prop, 0);
    printf("device[0]: %s  cc=%d.%d  mem=%.1f GB\n", prop.name,
           prop.major, prop.minor, (double)prop.totalGlobalMem / 1073741824.0);

    // cases: (n, d, N, chunk) — mirrors project cudacontract scan_parallel cases
    const unsigned cases[][4] = {
        {1000u, 3u, 4u, 256u},     // non-divisible, 4 blocks
        {16384u, 256u, 64u, 256u}, // large sequence, 64 blocks
        {128u, 8u, 8u, 64u},       // small, chunk 64 -> 2 blocks
    };
    const unsigned ncases = sizeof(cases) / sizeof(cases[0]);
    int failed = 0;

    for (unsigned ci = 0; ci < ncases; ++ci) {
        const unsigned n = cases[ci][0], d = cases[ci][1], N = cases[ci][2], chunk = cases[ci][3];
        const unsigned blocks = (n + chunk - 1) / chunk;
        const unsigned dN = d * N;
        const unsigned nn = n * d;
        printf("\ncase %u: n=%u d=%u N=%u chunk=%u blocks=%u\n", ci, n, d, N, chunk, blocks);

        float* input = fill_f32(nn, 1000u + ci, -1.5f, 1.5f);
        float* a = fill_f32(dN, 2000u + ci, 0.80f, 0.99f);
        float* b = fill_f32(dN, 3000u + ci, -0.5f, 0.5f);
        float* c = fill_f32(dN, 4000u + ci, -0.5f, 0.5f);
        float* skip = fill_f32(d, 5000u + ci, -1.0f, 1.0f);
        float* init = fill_f32(dN, 6000u + ci, -0.5f, 0.5f);

        // scalar reference (independent implementation, serial order)
        float* ref_out = (float*)malloc(nn * sizeof(float));
        float* ref_state = (float*)malloc(dN * sizeof(float));
        for (unsigned i = 0; i < d; ++i) {
            float h[LLM_SCAN_MAXN];
            for (unsigned j = 0; j < N; ++j) { h[j] = init[i * N + j]; }
            for (unsigned t = 0; t < n; ++t) {
                const float x = input[t * d + i];
                float acc = 0.0f;
                for (unsigned j = 0; j < N; ++j) {
                    const unsigned idx = i * N + j;
                    h[j] = a[idx] * h[j] + b[idx] * x;
                    acc += c[idx] * h[j];
                }
                ref_out[t * d + i] = acc + skip[i] * x;
            }
            for (unsigned j = 0; j < N; ++j) { ref_state[i * N + j] = h[j]; }
        }

        // device buffers
        float *d_input, *d_a, *d_b, *d_c, *d_skip, *d_init, *d_bt, *d_bp, *d_state, *d_out;
        const size_t sz_input = nn * sizeof(float);
        const size_t sz_dN = dN * sizeof(float);
        const size_t sz_bt = blocks * 2u * dN * sizeof(float);
        const size_t sz_bp = (blocks + 1u) * 2u * dN * sizeof(float);
        musaMalloc(&d_input, sz_input);
        musaMalloc(&d_a, sz_dN);
        musaMalloc(&d_b, sz_dN);
        musaMalloc(&d_c, sz_dN);
        musaMalloc(&d_skip, d * sizeof(float));
        musaMalloc(&d_init, sz_dN);
        musaMalloc(&d_bt, sz_bt);
        musaMalloc(&d_bp, sz_bp);
        musaMalloc(&d_state, sz_dN);
        musaMalloc(&d_out, sz_input);
        musaMemcpy(d_input, input, sz_input, musaMemcpyHostToDevice);
        musaMemcpy(d_a, a, sz_dN, musaMemcpyHostToDevice);
        musaMemcpy(d_b, b, sz_dN, musaMemcpyHostToDevice);
        musaMemcpy(d_c, c, sz_dN, musaMemcpyHostToDevice);
        musaMemcpy(d_skip, skip, d * sizeof(float), musaMemcpyHostToDevice);
        musaMemcpy(d_init, init, sz_dN, musaMemcpyHostToDevice);
        int ok_case = 1;
        double ms = 0.0;
        if (check("alloc/copy")) { failed = 1; ok_case = 0; }

        // launches (same grid/block geometry as crates/cuda/src/ffi.rs)
        const unsigned grid_y = (d + LLM_SCAN_BLOCKDIM - 1) / LLM_SCAN_BLOCKDIM;
        void* args_agg[] = { &d_input, &d_a, &d_b, &d_c, &d_skip, &d_init, &d_bt,
                             (void*)&n, (void*)&d, (void*)&N, (void*)&chunk, (void*)&blocks };
        if (ok_case) {
            musaLaunchKernel((const void*)scan_agg, dim3(blocks, grid_y, 1),
                             dim3(LLM_SCAN_BLOCKDIM, 1, 1), args_agg, 0, NULL);
            if (check("scan_agg")) { failed = 1; ok_case = 0; }
        }

        const unsigned prefix_threads = dN < 1024 ? dN : 1024;
        void* args_prefix[] = { &d_bt, &d_init, &d_state, &d_bp, (void*)&n, (void*)&d,
                               (void*)&N, (void*)&chunk, (void*)&blocks };
        if (ok_case) {
            musaLaunchKernel((const void*)scan_prefix, dim3(1, 1, 1),
                             dim3(prefix_threads, 1, 1), args_prefix, 0, NULL);
            if (check("scan_prefix")) { failed = 1; ok_case = 0; }
        }

        void* args_apply[] = { &d_input, &d_a, &d_b, &d_c, &d_skip, &d_init, &d_bp,
                               &d_out, (void*)&n, (void*)&d, (void*)&N, (void*)&chunk,
                               (void*)&blocks };
        if (ok_case) {
            musaLaunchKernel((const void*)scan_apply, dim3(blocks, grid_y, 1),
                             dim3(LLM_SCAN_BLOCKDIM, 1, 1), args_apply, 0, NULL);
            if (check("scan_apply")) { failed = 1; ok_case = 0; }
            struct timespec ts0, ts1;
            clock_gettime(CLOCK_MONOTONIC, &ts0);
            musaDeviceSynchronize();
            if (check("sync")) { failed = 1; ok_case = 0; }
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            ms = (double)(ts1.tv_sec - ts0.tv_sec) * 1e3
               + (double)(ts1.tv_nsec - ts0.tv_nsec) / 1e6;
        }

        float* out = (float*)malloc(sz_input);
        float* state = (float*)malloc(sz_dN);
        if (ok_case) {
            musaMemcpy(out, d_out, sz_input, musaMemcpyDeviceToHost);
            musaMemcpy(state, d_state, sz_dN, musaMemcpyDeviceToHost);
            if (check("D2H")) { failed = 1; ok_case = 0; }
        }

        // tolerance contract (same as project 16K case): abs 1e-3 / rel 1e-5
        const float tol_abs = 1e-3f, tol_rel = 1e-5f;
        double max_abs = 0.0, max_rel = 0.0;
        unsigned bad_out = 0, bad_state = 0;
        for (unsigned k = 0; k < nn; ++k) {
            const double ae = fabs((double)out[k] - ref_out[k]);
            const double re = ae / (fabs((double)ref_out[k]) + 1e-12);
            if (ae > max_abs) max_abs = ae;
            if (re > max_rel) max_rel = re;
            if (ae > tol_abs && re > tol_rel) bad_out++;
        }
        for (unsigned k = 0; k < dN; ++k) {
            const double ae = fabs((double)state[k] - ref_state[k]);
            const double re = ae / (fabs((double)ref_state[k]) + 1e-12);
            if (ae > tol_abs && re > tol_rel) bad_state++;
        }
        const int ok = (bad_out == 0 && bad_state == 0);
        printf("  gpu time (3 passes incl. sync): %.2f ms\n", ms);
        printf("  out:  max|abs|=%.3e max|rel|=%.3e bad=%u\n", max_abs, max_rel, bad_out);
        printf("  state:max|abs|=%.3e max|rel|=%.3e bad=%u\n", max_abs, max_rel, bad_state);
        printf("  => %s\n", ok ? "PASS" : "FAIL");
        if (!ok) failed = 1;

        free(out); free(state);
        musaFree(d_input); musaFree(d_a); musaFree(d_b); musaFree(d_c);
        musaFree(d_skip); musaFree(d_init); musaFree(d_bt); musaFree(d_bp);
        musaFree(d_state); musaFree(d_out);
        free(input); free(a); free(b); free(c); free(skip); free(init);
        free(ref_out); free(ref_state);
    }
    printf("\n%s\n", failed ? "SMOKE FAILED" : "SMOKE PASSED (MTT S4000)");
    return failed;
}
