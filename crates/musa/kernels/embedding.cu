// f32 embedding-gather kernel, compiled at run time through NVRTC.
// table is [vocab, width], indices holds rows row numbers validated on the host,
// out is [rows, width]. One thread per output element; fully coalesced reads.
// Symbol: embedding
extern "C" __global__ void embedding(const float* table,
                                     const unsigned long long* indices,
                                     float* out, unsigned long long rows,
                                     unsigned long long width) {
    unsigned long long total = rows * width;
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    unsigned long long row = index / width;
    out[index] = table[indices[row] * width + (index % width)];
}
