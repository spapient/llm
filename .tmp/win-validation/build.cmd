@echo off
call "D:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d C:\Users\fei\cutile-gpu-validation
nvcc -std=c++17 -O3 -arch=sm_86 -lineinfo -Xcompiler /O2 matmul_validation.cu -o matmul_validation.exe > build.log 2>&1
set STATUS=%ERRORLEVEL%
type build.log
exit /b %STATUS%
