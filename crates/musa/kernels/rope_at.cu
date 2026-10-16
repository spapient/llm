// Absolute-position rotary embedding of a single token, compiled at run time
// through NVRTC. Position is a scalar parameter, not row % seqaxis.
// Symbol: rope_at
extern "C" __global__ void rope_at(const float* x, float* y, unsigned heads,
                                   unsigned headdim, unsigned position, float base) {
    unsigned half = headdim / 2;
    unsigned long long total = (unsigned long long)heads * half;
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    unsigned i = (unsigned)(index % half);
    unsigned h = (unsigned)(index / half);
    float angle = (float)position * powf(base, -2.0f * (float)i / (float)headdim);
    float s = sinf(angle);
    float c = cosf(angle);
    unsigned long long offset = ((unsigned long long)h) * headdim + 2u * i;
    float even = x[offset];
    float odd = x[offset + 1u];
    y[offset] = even * c - odd * s;
    y[offset + 1u] = even * s + odd * c;
}
