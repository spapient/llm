// f32 semigroup scan kernel, shared by the CUDA and MUSA backends. One thread
// per channel; each thread runs the serial recurrence over the sequence axis for
// its channel's state vector, so the result is bit-for-bit the serial reference.
// The state buffer is initialized to init by the host and updated in place.
// n, d, N are three unsigned values; the pointer arguments are input, a, b, c,
// skip, state (in/out), out.
//
// MUSA's recorded path keeps the caller's state slot downloadable, so it takes
// a second output for the final state; the definition is compiled in only for
// the `mcc` pipeline (`-DMUSA_OUT_STATE`). NVRTC compiles without it and the
// kernel is byte-for-byte the CUDA definition.
// Symbol: scan
extern "C" __global__ void scan(const float* input, const float* a,
                                const float* b, const float* c, const float* skip,
                                float* state, float* out
#ifdef MUSA_OUT_STATE
                                , float* out_state
#endif
                                , unsigned n, unsigned d, unsigned N) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned long long)d) {
        return;
    }
    for (unsigned t = 0; t < n; ++t) {
        float x = input[t * d + i];
        float acc = 0.0f;
        for (unsigned j = 0; j < N; ++j) {
            unsigned idx = i * N + j;
            float h = a[idx] * state[idx] + b[idx] * x;
            state[idx] = h;
            acc += c[idx] * h;
        }
        out[t * d + i] = acc + skip[i] * x;
    }
#ifdef MUSA_OUT_STATE
    for (unsigned j = 0; j < N; ++j) {
        out_state[i * N + j] = state[i * N + j];
    }
#endif
}
