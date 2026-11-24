@echo off
call "D:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d C:\Users\fei\cutile-nvidia-cublas-validation
nvcc -std=c++17 -O3 -arch=sm_86 -lineinfo -Xcompiler /O2 matmul_validation.cu -o matmul_validation.exe -lcublas > build.log 2>&1
if errorlevel 1 goto failure
nvcc -std=c++17 -O3 -arch=sm_86 -lineinfo -Xcompiler /O2 cublas_benchmark.cu -o cublas_benchmark.exe -lcublas > build_cublas.log 2>&1
if errorlevel 1 goto failure
type build.log
type build_cublas.log
exit /b 0
:failure
type build.log
if exist build_cublas.log type build_cublas.log
exit /b 1
