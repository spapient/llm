@echo off
call "D:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d C:\Users\fei\a3000-device-query
nvcc -std=c++17 -O2 -arch=sm_86 device_query.cu -o device_query.exe
if errorlevel 1 exit /b 1
device_query.exe
