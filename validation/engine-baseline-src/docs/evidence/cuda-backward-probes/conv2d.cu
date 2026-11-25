#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define H 4
#define W 4
#define CIN 2
#define COUT 3
#define K 3
#define S 1
#define P 1
#define OH 4
#define OW 4
#define PATCH (K*K*CIN)
#define ROWS (OH*OW)

// col[ROWS*PATCH], weight[PATCH*COUT], bias[COUT], dy[ROWS*COUT]
__global__ void conv_dw(const float* __restrict__ col, const float* __restrict__ dy,
    float* __restrict__ dw) {
    int p = blockIdx.x, o = threadIdx.x;
    if (p >= PATCH || o >= COUT) return;
    float acc = 0;
    for (int r = 0; r < ROWS; r++) acc += col[r * PATCH + p] * dy[r * COUT + o];
    dw[p * COUT + o] = acc;
}
__global__ void conv_dbias(const float* __restrict__ dy, float* __restrict__ dbias) {
    int o = threadIdx.x;
    if (o >= COUT) return;
    float acc = 0;
    for (int r = 0; r < ROWS; r++) acc += dy[r * COUT + o];
    dbias[o] = acc;
}
__global__ void conv_dcol(const float* __restrict__ dy, const float* __restrict__ w,
    float* __restrict__ dcol) {
    int r = blockIdx.x, p = threadIdx.x;
    if (r >= ROWS || p >= PATCH) return;
    float acc = 0;
    for (int o = 0; o < COUT; o++) acc += dy[r * COUT + o] * w[p * COUT + o];
    dcol[r * PATCH + p] = acc;
}
__global__ void conv_dx(const float* __restrict__ dcol, float* __restrict__ dx) {
    int r = blockIdx.x, p = threadIdx.x;
    if (r >= ROWS || p >= PATCH) return;
    int ki = p / (K * CIN);
    int rem = p % (K * CIN);
    int kj = rem / CIN;
    int ci = rem % CIN;
    int ni = r / (OH * OW);
    int rr = r % (OH * OW);
    int oi = rr / OW, oj = rr % OW;
    int ii = oi * S + ki - P, jj = oj * S + kj - P;
    if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
        int dst = ((ni * H + ii) * W + jj) * CIN + ci;
        atomicAdd(dx + dst, dcol[r * PATCH + p]);
    }
}

static double maxrel(const float* got, const double* ref, int len) {
    double mr = 0;
    for (int i = 0; i < len; i++) {
        double den = ref[i] > 1e-3 ? ref[i] : 1e-3;
        double rr = fabs((double)got[i] - ref[i]) / den;
        if (rr > mr) mr = rr;
    }
    return mr;
}

int main() {
    // host f64 reference
    double x[H*W*CIN], wt[PATCH*COUT], bs[COUT], dy[ROWS*COUT];
    for (int i=0;i<H*W*CIN;i++) x[i]=0.1*(i%5)-0.2;
    for (int i=0;i<PATCH*COUT;i++) wt[i]=0.05+0.03*(i%7);
    for (int o=0;o<COUT;o++) bs[o]=0.4;
    for (int i=0;i<ROWS*COUT;i++) dy[i]=0.1*(i%6)-0.25;
    double col[ROWS*PATCH];
    for (int r=0;r<ROWS;r++){
        int ni=r/(OH*OW), rr=r%(OH*OW), oi=rr/OW, oj=rr%OW;
        for (int ki=0;ki<K;ki++) for(int kj=0;kj<K;kj++){
            int ii=oi*S+ki-P, jj=oj*S+kj-P;
            for(int ci=0;ci<CIN;ci++){
                int p=(ki*K+kj)*CIN+ci;
                col[r*PATCH+p] = (ii>=0&&ii<H&&jj>=0&&jj<W) ? x[((ni*H+ii)*W+jj)*CIN+ci] : 0.0;
            }
        }
    }
    double rDW[PATCH*COUT], rDB[COUT], rDC[ROWS*PATCH], rDX[H*W*CIN];
    for(int i=0;i<H*W*CIN;i++) rDX[i]=0;
    for(int o=0;o<COUT;o++){ double acc=0; for(int r=0;r<ROWS;r++) acc+=dy[r*COUT+o]; rDB[o]=acc; }
    for(int p=0;p<PATCH;p++)for(int o=0;o<COUT;o++){double acc=0;for(int r=0;r<ROWS;r++)acc+=col[r*PATCH+p]*dy[r*COUT+o];rDW[p*COUT+o]=acc;}
    for(int r=0;r<ROWS;r++)for(int p=0;p<PATCH;p++){double acc=0;for(int o=0;o<COUT;o++)acc+=dy[r*COUT+o]*wt[p*COUT+o];rDC[r*PATCH+p]=acc;}
    for(int r=0;r<ROWS;r++)for(int ki=0;ki<K;ki++)for(int kj=0;kj<K;kj++)for(int ci=0;ci<CIN;ci++){
        int ii=(r%(OH*OW))/OW*S+ki-P, jj=(r%(OH*OW))%OW*S+kj-P;
        if(ii>=0&&ii<H&&jj>=0&&jj<W){int p=(ki*K+kj)*CIN+ci;rDX[((ii)*W+jj)*CIN+ci]+=rDC[r*PATCH+p];}
    }

    float *dc,*dw,*db,*dx,*pw,*pdy,*pcol;
    cudaMalloc(&dw,PATCH*COUT*4);cudaMalloc(&db,COUT*4);cudaMalloc(&dc,ROWS*PATCH*4);cudaMalloc(&dx,H*W*CIN*4);
    cudaMalloc(&pw,PATCH*COUT*4);cudaMalloc(&pdy,ROWS*COUT*4);cudaMalloc(&pcol,ROWS*PATCH*4);
    float fw[PATCH*COUT],fdy[ROWS*COUT],fcol[ROWS*PATCH];
    for(int i=0;i<PATCH*COUT;i++)fw[i]=(float)wt[i];
    for(int i=0;i<ROWS*COUT;i++)fdy[i]=(float)dy[i];
    for(int i=0;i<ROWS*PATCH;i++)fcol[i]=(float)col[i];
    cudaMemcpy(pw,fw,PATCH*COUT*4,cudaMemcpyHostToDevice);
    cudaMemcpy(pdy,fdy,ROWS*COUT*4,cudaMemcpyHostToDevice);
    cudaMemcpy(pcol,fcol,ROWS*PATCH*4,cudaMemcpyHostToDevice);
    cudaMemset(dx,0,H*W*CIN*4);
    conv_dw<<<PATCH,COUT>>>(pcol,pdy,dw);
    conv_dbias<<<1,COUT>>>(pdy,db);
    conv_dcol<<<ROWS,PATCH>>>(pdy,pw,dc);
    conv_dx<<<ROWS,PATCH>>>(dc,dx);
    float gDW[PATCH*COUT],gDB[COUT],gDC[ROWS*PATCH],gDX[H*W*CIN];
    cudaMemcpy(gDW,dw,PATCH*COUT*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(gDB,db,COUT*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(gDC,dc,ROWS*PATCH*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(gDX,dx,H*W*CIN*4,cudaMemcpyDeviceToHost);
    printf("conv dW maxrel=%.5f %s\n", maxrel(gDW,rDW,PATCH*COUT), maxrel(gDW,rDW,PATCH*COUT)<5e-2?"PASS":"FAIL");
    printf("conv dbias maxrel=%.5f %s\n", maxrel(gDB,rDB,COUT), maxrel(gDB,rDB,COUT)<5e-2?"PASS":"FAIL");
    printf("conv dcol maxrel=%.5f %s\n", maxrel(gDC,rDC,ROWS*PATCH), maxrel(gDC,rDC,ROWS*PATCH)<5e-2?"PASS":"FAIL");
    printf("conv dx maxrel=%.5f %s\n", maxrel(gDX,rDX,H*W*CIN), maxrel(gDX,rDX,H*W*CIN)<5e-2?"PASS":"FAIL");
    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);printf("device=%s\n",p.name);
    return 0;
}
