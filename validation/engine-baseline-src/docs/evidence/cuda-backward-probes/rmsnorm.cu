#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define N 96
#define BLOCK 256
#define EPS 1e-6

__global__ void rms_reduce(const float* __restrict__ x, const float* __restrict__ g,
    const float* __restrict__ dout, float* __restrict__ scratch) {
    __shared__ float sx[BLOCK], sw[BLOCK];
    int i = threadIdx.x;
    sx[i] = (i < N) ? x[i] * x[i] : 0.0f;
    sw[i] = (i < N) ? dout[i] * g[i] * x[i] : 0.0f;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (i < s) { sx[i] += sx[i + s]; sw[i] += sw[i + s]; }
        __syncthreads();
    }
    if (i == 0) { scratch[0] = sx[0]; scratch[1] = sw[0]; }
}

__global__ void rms_map(const float* __restrict__ x, const float* __restrict__ g,
    const float* __restrict__ dout, const float* __restrict__ sc,
    float* __restrict__ dx, float* __restrict__ dg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float rms = sqrtf(sc[0] / (float)N + EPS);
    float w = sc[1];
    float r3 = (float)N * rms * rms * rms;
    dx[i] = dout[i] * g[i] / rms - x[i] * w / r3;
    dg[i] = dout[i] * x[i] / rms;
}

static double fwd_row(const double* x, const double* g, int i) {
    double ms = 0;
    for (int k = 0; k < N; k++) ms += x[k] * x[k];
    ms /= N;
    double rms = sqrt(ms + EPS);
    return g[i] * x[i] / rms;
}
static double loss(const double* x, const double* g, const double* dout) {
    double L = 0;
    for (int i = 0; i < N; i++) L += dout[i] * fwd_row(x, g, i);
    return L;
}

int main() {
    double hx[N], hg[N], hd[N];
    for (int i = 0; i < N; i++) {
        hx[i] = -1.0 + 0.03 * i;
        hg[i] = 0.8 + 0.02 * i;
        hd[i] = 0.4 + 0.01 * i;
    }
    double refx[N], refg[N];
    double h = 1e-6;
    for (int i = 0; i < N; i++) {
        double xp[N], xm[N], gp[N], gm[N];
        for (int k = 0; k < N; k++) { xp[k]=hx[k]; xm[k]=hx[k]; gp[k]=hg[k]; gm[k]=hg[k]; }
        xp[i]+=h; xm[i]-=h; gp[i]+=h; gm[i]-=h;
        refx[i]=(loss(xp,hg,hd)-loss(xm,hg,hd))/(2*h);
        refg[i]=(loss(hx,gp,hd)-loss(hx,gm,hd))/(2*h);
    }
    float *dx, *dg, *hx2, *hg2, *hd2, *sc;
    cudaMalloc(&dx, N*4); cudaMalloc(&dg, N*4);
    cudaMalloc(&hx2, N*4); cudaMalloc(&hg2, N*4); cudaMalloc(&hd2, N*4); cudaMalloc(&sc, 8);
    float tmx[N], tmg[N], tmd[N];
    for (int i=0;i<N;i++){tmx[i]=(float)hx[i];tmg[i]=(float)hg[i];tmd[i]=(float)hd[i];}
    cudaMemcpy(hx2,tmx,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(hg2,tmg,N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(hd2,tmd,N*4,cudaMemcpyHostToDevice);
    rms_reduce<<<1,BLOCK>>>(hx2,hg2,hd2,sc);
    rms_map<<<(N+BLOCK-1)/BLOCK,BLOCK>>>(hx2,hg2,hd2,sc,dx,dg);
    float trx[N], trg[N];
    cudaMemcpy(trx,dx,N*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(trg,dg,N*4,cudaMemcpyDeviceToHost);
    double mrx=0,mrg=0; int wx=-1,wg=-1;
    for (int i=0;i<N;i++){
        double dxp=trx[i], denomx=refx[i]>1e-3?refx[i]:1e-3; double rx=fabs(dxp-refx[i])/denomx; if(rx>mrx){mrx=rx;wx=i;}
        double dgp=trg[i], denomg=refg[i]>1e-3?refg[i]:1e-3; double rg=fabs(dgp-refg[i])/denomg; if(rg>mrg){mrg=rg;wg=i;}
    }
    printf("rmsnorm dx maxrel=%.5f worst=%d %s\n", mrx, wx, mrx<5e-2?"PASS":"FAIL");
    printf("rmsnorm dgamma maxrel=%.5f worst=%d %s\n", mrg, wg, mrg<5e-2?"PASS":"FAIL");
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0); printf("device=%s\n", p.name);
    return 0;
}
