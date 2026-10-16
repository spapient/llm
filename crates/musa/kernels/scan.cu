// f32 semigroup scan kernel, compiled at run time through NVRTC. One thread per
// channel; each thread runs the serial recurrence over the sequence axis for its
// channel's state vector, so the result is bit-for-bit the serial reference. The
// state buffer is initialized to init by the host and updated in place. n, d, N
// are three unsigned values; the pointer arguments are input, a, b, c, skip,
// state (in/out), out. Symbol: scan
extern "C" __global__ void scan(const float* input, const float* a,
                                const float* b, const float* c, const float* skip,
                                float* state, float* out, float* out_state,
                                unsigned n, unsigned d, unsigned N) {
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
    for (unsigned j = 0; j < N; ++j) {
        out_state[i * N + j] = state[i * N + j];
    }
}
