#!/usr/bin/env bash
# S4000 测量 harness 构建:mcc 编译微核与 muBLAS 对照。
set -e
MCC=/usr/local/musa/bin/mcc
LIB=/usr/local/musa/lib
INC=/usr/local/musa/include
mkdir -p obj
$MCC -O3 --use_fast_math -I$INC t_musa.cu -o obj/t_musa -L$LIB -lmusart
$MCC -O3 -I$INC t_mublas.cu -o obj/t_mublas -L$LIB -lmusart -lmublas
echo "build ok: obj/t_musa obj/t_mublas"
