// f32 rotary-position-embedding kernel, shared by the CUDA and MUSA backends.
// One thread per (row, head, pair); x and y are [rows, heads * headdim]. The
// position of a row is row % seqaxis, and the angle of pair i is
// position * base^(-2i / headdim). seqaxis == 0 places every row at position 0.
// Symbol: rope
extern "C" __global__ void rope(const float* x, float* y, unsigned rows,
                                unsigned seqaxis, unsigned heads, unsigned headdim,
                                float base) {
    unsigned half = headdim / 2;
    unsigned long long total = (unsigned long long)rows * heads * half;
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    unsigned i = (unsigned)(index % half);
    unsigned h = (unsigned)((index / half) % heads);
    unsigned row = (unsigned)(index / ((unsigned long long)half * heads));
    unsigned position = (seqaxis == 0u) ? 0u : (row % seqaxis);
    float angle = (float)position * powf(base, -2.0f * (float)i / (float)headdim);
    float s = sinf(angle);
    float c = cosf(angle);
    unsigned long long offset = (((unsigned long long)row * heads) + h) * headdim + 2u * i;
    float even = x[offset];
    float odd = x[offset + 1u];
    y[offset] = even * c - odd * s;
    y[offset + 1u] = even * s + odd * c;
}
