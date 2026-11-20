// CUDA runtime → MUSA 垫片:让 cutile/engine 生成的 CUDA 程序在 mcc 下编译。
// 仅覆盖 harness 用到的 API;类型与常量同名映射。
#pragma once
#include <musa_runtime.h>

#define cudaError_t musaError_t
#define cudaSuccess musaSuccess
#define cudaMalloc musaMalloc
#define cudaFree musaFree
#define cudaMemcpy musaMemcpy
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaDeviceSynchronize musaDeviceSynchronize
#define cudaGetErrorString musaGetErrorString
