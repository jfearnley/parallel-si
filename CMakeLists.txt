cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)

set(
    CUDA_NVCC_FLAGS
    "-O3 -gencode arch=compute_35,code=sm_35 --std=c++11"
    #"-g -O3 -lineinfo -gencode arch=compute_35,code=sm_35 --std=c++11"
    )

set( 
    CMAKE_CXX_FLAGS
    #"-Og -g --std=c++11"
    "-O3 -march=native --std=c++11"
    )

list (APPEND CUDA_NVCC_FLAGS --compiler-bindir /usr/bin/gcc-4.8)



cuda_add_executable(
    parity
    main.cpp 

    game.cpp 
    game.h

    gpu_game.cpp
    gpu_game.h

    cpu_parity.cpp 
    cpu_parity.h

    cpu_bv.cpp 
    cpu_bv.h

    cpu_list_ranking.cpp
    cpu_list_ranking.h

    cpu_vj.cpp
    cpu_vj.h

    gpu_util.cpp
    gpu_util.h

    gpu_parity.cu
    gpu_parity.h

    gpu_list_ranking.cu
    gpu_list_ranking.h

    cpu_bf.cpp
    cpu_bf.h

    valuation.h
    valuation.cpp

    si.h
    )

target_link_libraries(parity pthread)
