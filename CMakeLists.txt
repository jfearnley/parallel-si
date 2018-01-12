cmake_minimum_required(VERSION 3.0)
project(parsi VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

add_executable(parsi
    main.cpp 
    game.cpp 
    game.h
    cpu_parity.cpp 
    cpu_parity.h
    cpu_bv.cpp 
    cpu_bv.h
    cpu_list_ranking.cpp
    cpu_list_ranking.h
    cpu_vj.cpp
    cpu_vj.h
    cpu_bf.cpp
    cpu_bf.h
    valuation.h
    valuation.cpp
    si.h
)
set(CMAKE_CXX_FLAGS "-O3 -march=native")
target_link_libraries(parsi pthread boost_iostreams)

option(WITH_CUDA "Also build psi_gpu requiring CUDA" OFF)

if(WITH_CUDA)
    set(CUDA_HOST_COMPILER /usr/bin/gcc-4.9)
    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_FLAGS "-O3 -gencode arch=compute_35,code=sm_35 --std=c++11 -Xcompiler -fPIC")

    cuda_add_executable(
        gpuparsi
        gpumain.cpp 
        game.cpp 
        game.h
        gpu_game.cpp
        gpu_game.h
        gpu_util.cpp
        gpu_util.h
        gpu_parity.cu
        gpu_parity.h
        gpu_list_ranking.cu
        gpu_list_ranking.h
        valuation.h
        valuation.cpp
        si.h
    )
    set_property(TARGET gpuparsi PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(gpuparsi pthread boost_iostreams)
endif()
