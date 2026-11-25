#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// gemv: y[R] = W[R,C] * x[C]
#define R 6
#define C 8
// gemm: Y[M,N] = A[M,K] * B[K,N]
#define M 4
#define K 5
#define N 6

__global__ void gemv_dw(const float* __restrict__ w, const float* __restrict__ x,
    const float* __restrict__ dout, float* __restrict__ dw) {
    int n = blockIdx.x, k = threadIdx.x;
    if (n >= R || k >= C) return;
    dw[n * C + k] = dout[n] * x[k];
}
__global__ void gemv_dx(const float* __restrict__ w, const float* __restrict__ dout,
    float* __restrict__ dx) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= C) return;
    float acc = 0;
    for (int n = 0; n < R; n++) acc += dout[n] * w[n * C + k];
    dx[k] = acc;
}
__global__ void gemm_da(const float* __restrict__ b, const float* __restrict__ g,
    float* __restrict__ da) {
    int m = blockIdx.x, k = threadIdx.x;
    if (m >= M || k >= K) return;
    float acc = 0;
    for (int n = 0; n < N; n++) acc += g[m * N + n] * b[k * N + n];
    da[m * K + k] = acc;
}
__global__ void gemm_db(const float* __restrict__ a, const float* __restrict__ g,
    float* __restrict__ db) {
    int k = blockIdx.x, n = threadIdx.x;
    if (k >= K || n >= N) return;
    float acc = 0;
    for (int m = 0; m < M; m++) acc += a[m * K + k] * g[m * N + n];
    db[k * N + n] = acc;
}

static double maxrel(const float* got, const double* ref, int len) {
    double mr = 0;
    for (int i = 0; i < len; i++) {
        double den = ref[i] > 1e-3 ? ref[i] : 1e-3;
        double r = fabs((double)got[i] - ref[i]) / den;
        if (r > mr) mr = r;
    }
    return mr;
}

int main() {
    double W[R*C], x[C], y[R], dout[R];
    for (int n=0;n<R;n++) for(int k=0;k<C;k++) W[n*C+k]=0.1*((n+k)%5)-0.2;
    for (int k=0;k<C;k++) x[k]=0.2*k-0.5;
    for (int n=0;n<R;n++){ y[n]=0; for(int k=0;k<C;k++) y[n]+=W[n*C+k]*x[k]; dout[n]=0.3*n-0.4; }
    // reference
    double rDW[R*C], rDX[C];
    for (int n=0;n<R;n++) for(int k=0;k<C;k++) rDW[n*C+k]=dout[n]*x[k];
    for (int k=0;k<C;k++){double acc=0;for(int n=0;n<R;n++)acc+=dout[n]*W[n*C+k];rDX[k]=acc;}

    float *dW,*dX,*pW,*pX,*pD;
    cudaMalloc(&dW,R*C*4);cudaMalloc(&dX,C*4);cudaMalloc(&pW,R*C*4);cudaMalloc(&pX,C*4);cudaMalloc(&pD,R*4);
    float tW[R*C],tX[C],tD[R];
    for(int i=0;i<R*C;i++)tW[i]=(float)W[i]; for(int i=0;i<C;i++)tX[i]=(float)x[i]; for(int i=0;i<R;i++)tD[i]=(float)dout[i];
    cudaMemcpy(pW,tW,R*C*4,cudaMemcpyHostToDevice);cudaMemcpy(pX,tX,C*4,cudaMemcpyHostToDevice);cudaMemcpy(pD,tD,R*4,cudaMemcpyHostToDevice);
    gemv_dw<<<R,C>>>(pW,pX,pD,dW);
    gemv_dx<<<1,C>>>(pW,pD,dX);
    float gDW[R*C],gDX[C];
    cudaMemcpy(gDW,dW,R*C*4,cudaMemcpyDeviceToHost);cudaMemcpy(gDX,dX,C*4,cudaMemcpyDeviceToHost);
    printf("gemv dW maxrel=%.5f %s\n", maxrel(gDW,rDW,R*C), maxrel(gDW,rDW,R*C)<5e-2?"PASS":"FAIL");
    printf("gemv dx maxrel=%.5f %s\n", maxrel(gDX,rDX,C), maxrel(gDX,rDX,C)<5e-2?"PASS":"FAIL");

    // gemm
    double A[M*K],B[K*N],G[M*N];
    for(int m=0;m<M;m++)for(int k=0;k<K;k++)A[m*K+k]=0.15*((m+k)%4)-0.3;
    for(int k=0;k<K;k++)for(int n=0;n<N;n++)B[k*N+n]=0.12*((k+n)%3)-0.15;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++)G[m*N+n]=0.2*m-0.1*n;
    double rDA[M*K],rDB[K*N];
    for(int m=0;m<M;m++)for(int k=0;k<K;k++){double acc=0;for(int n=0;n<N;n++)acc+=G[m*N+n]*B[k*N+n];rDA[m*K+k]=acc;}
    for(int k=0;k<K;k++)for(int n=0;n<N;n++){double acc=0;for(int m=0;m<M;m++)acc+=A[m*K+k]*G[m*N+n];rDB[k*N+n]=acc;}
    float *dA,*dB,*pA,*pB,*pG;
    cudaMalloc(&dA,M*K*4);cudaMalloc(&dB,K*N*4);cudaMalloc(&pA,M*K*4);cudaMalloc(&pB,K*N*4);cudaMalloc(&pG,M*N*4);
    float tA[M*K],tB[K*N],tG[M*N];
    for(int i=0;i<M*K;i++)tA[i]=(float)A[i];for(int i=0;i<K*N;i++)tB[i]=(float)B[i];for(int i=0;i<M*N;i++)tG[i]=(float)G[i];
    cudaMemcpy(pA,tA,M*K*4,cudaMemcpyHostToDevice);cudaMemcpy(pB,tB,K*N*4,cudaMemcpyHostToDevice);cudaMemcpy(pG,tG,M*N*4,cudaMemcpyHostToDevice);
    gemm_da<<<M,K>>>(pB,pG,dA);
    gemm_db<<<K,N>>>(pA,pG,dB);
    float gDA[M*K],gDB[K*N];
    cudaMemcpy(gDA,dA,M*K*4,cudaMemcpyDeviceToHost);cudaMemcpy(gDB,dB,K*N*4,cudaMemcpyDeviceToHost);
    printf("gemm dA maxrel=%.5f %s\n", maxrel(gDA,rDA,M*K), maxrel(gDA,rDA,M*K)<5e-2?"PASS":"FAIL");
    printf("gemm dB maxrel=%.5f %s\n", maxrel(gDB,rDB,K*N), maxrel(gDB,rDB,K*N)<5e-2?"PASS":"FAIL");
    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);printf("device=%s\n",p.name);
    return 0;
}
