#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <math.h>

#define QR 8
#define HST 8
#define D 8
#define SCALE (1.0f/sqrtf((float)D))

__global__ void att_row(const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    float* __restrict__ prob, float* __restrict__ ds) {
    int i = blockIdx.x, j = threadIdx.x;
    __shared__ float sc[HST], pp[HST], dp[HST], dsv[HST];
    float dotk = 0, dotv = 0;
    for (int h = 0; h < D; h++) {
        dotk += Q[i*D+h] * K[j*D+h];
        dotv += dO[i*D+h] * V[j*D+h];
    }
    sc[j] = dotk * SCALE;
    dp[j] = dotv;
    __syncthreads();
    float mx = sc[0];
    for (int t=1;t<HST;t++) if(sc[t]>mx) mx=sc[t];
    __syncthreads();
    float e = expf(sc[j]-mx);
    pp[j]=e;
    __syncthreads();
    float sum=0; for(int t=0;t<HST;t++) sum+=pp[t];
    __syncthreads();
    pp[j]=pp[j]/sum;
    float weighted=0; for(int t=0;t<HST;t++) weighted+=pp[t]*dp[t];
    __syncthreads();
    dsv[j]=pp[j]*(dp[j]-weighted);
    prob[i*HST+j]=pp[j];
    ds[i*HST+j]=dsv[j];
}

__global__ void att_scatter(const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ dO,
    const float* __restrict__ prob, const float* __restrict__ ds,
    float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dv) {
    int i = blockIdx.y, j = blockIdx.x, h = threadIdx.x;
    if (h >= D) return;
    float p = prob[i*HST+j];
    float s = ds[i*HST+j] * SCALE;
    atomicAdd(dv + j*D+h, p * dO[i*D+h]);
    atomicAdd(dq + i*D+h, s * K[j*D+h]);
    atomicAdd(dk + j*D+h, s * Q[i*D+h]);
}

static double maxrel(const float* got, const double* ref, int len) {
    double mr=0;
    for(int i=0;i<len;i++){double den=ref[i]>1e-3?ref[i]:1e-3;double r=fabs((double)got[i]-ref[i])/den;if(r>mr)mr=r;}
    return mr;
}

int main() {
    double Q[QR*D],K[HST*D],V[HST*D],dO[QR*D];
    for(int i=0;i<QR*D;i++) Q[i]=0.12*(i%7)-0.3;
    for(int i=0;i<HST*D;i++) K[i]=0.1*(i%5)-0.2;
    for(int i=0;i<HST*D;i++) V[i]=0.08*(i%4)-0.12;
    for(int i=0;i<QR*D;i++) dO[i]=0.06*(i%6)-0.15;
    double rDQ[QR*D]={0},rDK[HST*D]={0},rDV[HST*D]={0};
    double prob[QR*HST], dS[QR*HST];
    for(int i=0;i<QR;i++){
        double sc[HST], pp[HST], dp[HST];
        double mx=-1e30;
        for(int j=0;j<HST;j++){double a=0;for(int h=0;h<D;h++)a+=Q[i*D+h]*K[j*D+h];sc[j]=a*SCALE;if(sc[j]>mx)mx=sc[j];}
        double sum=0;for(int j=0;j<HST;j++){pp[j]=exp(sc[j]-mx);sum+=pp[j];}
        for(int j=0;j<HST;j++){pp[j]/=sum;double a=0;for(int h=0;h<D;h++)a+=dO[i*D+h]*V[j*D+h];dp[j]=a;}
        double w=0;for(int j=0;j<HST;j++)w+=pp[j]*dp[j];
        for(int j=0;j<HST;j++){double dsj=pp[j]*(dp[j]-w);prob[i*HST+j]=pp[j];dS[i*HST+j]=dsj;
            for(int h=0;h<D;h++){
                rDV[j*D+h]+=pp[j]*dO[i*D+h];
                rDQ[i*D+h]+=dsj*SCALE*K[j*D+h];
                rDK[j*D+h]+=dsj*SCALE*Q[i*D+h];
            }}
    }
    float *pQ,*pK,*pV,*pDO,*pp,*pds,*pdq,*pdk,*pdv;
    cudaMalloc(&pQ,QR*D*4);cudaMalloc(&pK,HST*D*4);cudaMalloc(&pV,HST*D*4);cudaMalloc(&pDO,QR*D*4);
    cudaMalloc(&pp,QR*HST*4);cudaMalloc(&pds,QR*HST*4);
    cudaMalloc(&pdq,QR*D*4);cudaMalloc(&pdk,HST*D*4);cudaMalloc(&pdv,HST*D*4);
    float fQ[QR*D],fK[HST*D],fV[HST*D],fDO[QR*D];
    for(int i=0;i<QR*D;i++)fQ[i]=(float)Q[i];for(int i=0;i<HST*D;i++)fK[i]=(float)K[i];
    for(int i=0;i<HST*D;i++)fV[i]=(float)V[i];for(int i=0;i<QR*D;i++)fDO[i]=(float)dO[i];
    cudaMemcpy(pQ,fQ,QR*D*4,cudaMemcpyHostToDevice);cudaMemcpy(pK,fK,HST*D*4,cudaMemcpyHostToDevice);
    cudaMemcpy(pV,fV,HST*D*4,cudaMemcpyHostToDevice);cudaMemcpy(pDO,fDO,QR*D*4,cudaMemcpyHostToDevice);
    cudaMemset(pdq,0,QR*D*4);cudaMemset(pdk,0,HST*D*4);cudaMemset(pdv,0,HST*D*4);
    att_row<<<QR,HST>>>(pQ,pK,pV,pDO,pp,pds);
    dim3 g(HST,QR);
    att_scatter<<<g,D>>>(pQ,pK,pV,pDO,pp,pds,pdq,pdk,pdv);
    float gDQ[QR*D],gDK[HST*D],gDV[HST*D];
    cudaMemcpy(gDQ,pdq,QR*D*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(gDK,pdk,HST*D*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(gDV,pdv,HST*D*4,cudaMemcpyDeviceToHost);
    printf("attn dQ maxrel=%.5f %s\n", maxrel(gDQ,rDQ,QR*D), maxrel(gDQ,rDQ,QR*D)<5e-2?"PASS":"FAIL");
    printf("attn dK maxrel=%.5f %s\n", maxrel(gDK,rDK,HST*D), maxrel(gDK,rDK,HST*D)<5e-2?"PASS":"FAIL");
    printf("attn dV maxrel=%.5f %s\n", maxrel(gDV,rDV,HST*D), maxrel(gDV,rDV,HST*D)<5e-2?"PASS":"FAIL");
    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);printf("device=%s\n",p.name);
    return 0;
}
