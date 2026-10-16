// MTT S4000 (Moore Threads MUSA) per-dispatch cost decomposition probe.
//
// The crate's dispatch (crates/musa/src/ffi.rs) is run-to-completion: per
// dispatch it allocates fresh device buffers, uploads the host operands,
// launches one kernel, synchronises, downloads the outputs, and returns the
// buffers to the pool (or frees them). This probe times each phase in
// isolation so a real-machine run can say which phase dominates the scan
// benchmark's ~9 ms per dispatch, and therefore whether the device-buffer pool
// (Layer 2, already landed) or a device-resident pass-through (Layer 1) is the
// higher-value follow-up.
//
// It drives the MUSA *runtime* API (musaMalloc/musaMemcpy/musaLaunchKernel/
// musaDeviceSynchronize/musaFree), the thin layer over the driver the crate
// reaches through the `mu*` ABI. The plan's launch probe measured the two
// layers within a few microseconds of each other (docs/musa-backend-plan.md
// §4.x), so the numbers are a valid proxy for the crate's driver calls.
//
// Compile and run on the MTT host (see scripts/musa/README.md):
//   /usr/local/musa-3.1.0/bin/mcc -mtgpu --offload-arch=mp_22 \
//     -I/usr/local/musa-3.1.0/include -L/usr/local/musa-3.1.0/lib -lmusart \
//     -o alloc_probe alloc_probe.cu
//   LD_LIBRARY_PATH=/usr/local/musa-3.1.0/lib ./alloc_probe
#include <musa_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// One word of work, enough to launch without being optimised away, with a run
// time negligible next to the launch+sync being measured.
extern "C" __global__ void noop(unsigned* flag) { *flag = threadIdx.x; }

#define ITERS 2000

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static int check(const char* where) {
    musaError_t e = musaGetLastError();
    if (e != musaSuccess) {
        fprintf(stderr, "MUSA error %d at %s: %s\n", (int)e, where, musaGetErrorString(e));
        return 1;
    }
    return 0;
}

static void time_alloc_free(size_t bytes) {
    double t0 = now_us();
    for (int i = 0; i < ITERS; ++i) {
        void* p = NULL;
        musaMalloc(&p, bytes);
        musaFree(p);
    }
    if (check("alloc/free")) return;
    printf("alloc+free  %6zu B   %8.2f us/round\n", bytes, (now_us() - t0) / ITERS);
}

static void time_htod_dtoh(size_t bytes) {
    void* h = malloc(bytes);
    void* d = NULL;
    musaMalloc(&d, bytes);
    if (check("alloc for copy")) return;
    double t0 = now_us();
    for (int i = 0; i < ITERS; ++i) musaMemcpy(d, h, bytes, musaMemcpyHostToDevice);
    double htod = (now_us() - t0) / ITERS;
    t0 = now_us();
    for (int i = 0; i < ITERS; ++i) musaMemcpy(h, d, bytes, musaMemcpyDeviceToHost);
    double dtoh = (now_us() - t0) / ITERS;
    if (check("copy")) return;
    printf("HtoD        %6zu B   %8.2f us   |  DtoH  %6zu B   %8.2f us\n",
           bytes, htod, bytes, dtoh);
    musaFree(d);
    free(h);
}

static void time_launch_sync(void) {
    void* d_flag = NULL;
    musaMalloc(&d_flag, sizeof(unsigned));
    void* args[] = { &d_flag };
    double t0 = now_us();
    for (int i = 0; i < ITERS; ++i) {
        musaLaunchKernel((const void*)noop, dim3(1, 1, 1), dim3(1, 1, 1), args, 0, NULL);
        musaDeviceSynchronize();
    }
    if (check("launch/sync")) return;
    printf("launch+sync (1 thread)          %8.2f us\n", (now_us() - t0) / ITERS);
    musaFree(d_flag);
}

int main(void) {
    int dev_count = 0;
    musaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        printf("FAIL: no MUSA device\n");
        return 1;
    }
    // Warm the allocator once so the first-call cost is not counted.
    {
        void* p = NULL;
        musaMalloc(&p, 1 << 20);
        musaFree(p);
    }
    time_launch_sync();
    time_alloc_free(1 << 16);  // scan input (n=256, d=64, f32)
    time_alloc_free(1 << 10);  // a/b/c/init (d*statewidth f32)
    time_alloc_free(1 << 12);  // dtotal/dprefix scale
    time_htod_dtoh(1 << 16);
    time_htod_dtoh(1 << 10);
    return 0;
}
