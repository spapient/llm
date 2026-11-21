// CUDA runtime → MUSA 垫片：让 `cutile` / `engine` 生成的 CUDA 程序在 mcc 下编译。
//
// 生成的程序只写 `cuda*` 名字；MUSA 5.2 的工具链提供同名能力、但类型与函数叫 `musa*`，
// 因此这里只做宏映射，不做任何语义包装。**只列生成程序与计时注入真正用到的符号**——
// 多列一个就等于承诺了 harness 从没用过的东西。
#pragma once
#include <musa_runtime.h>

#define cudaError_t musaError_t
#define cudaSuccess musaSuccess
#define cudaMalloc musaMalloc
#define cudaFree musaFree
#define cudaMemset musaMemset
#define cudaMemcpy musaMemcpy
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaDeviceSynchronize musaDeviceSynchronize
#define cudaGetErrorString musaGetErrorString

// 计时注入（`musa::timed`）把发射到回读之间的链路包进事件循环，用到的是这四个事件 API。
#define cudaEvent_t musaEvent_t
#define cudaEventCreate musaEventCreate
#define cudaEventRecord musaEventRecord
#define cudaEventSynchronize musaEventSynchronize
#define cudaEventElapsedTime musaEventElapsedTime
