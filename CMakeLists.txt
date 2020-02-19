cmake_minimum_required(VERSION 3.15)
project(blocked_ijk_blas C)

set(CMAKE_C_STANDARD 11)

# THIS IS ONLY WORKS ON MY SYSTEM, PLEASE FIGURE OUT YOUR PATHS YOURSELF
include_directories(
        /usr/local/opt/openblas/include
        /usr/local/opt/libomp/include
)
link_directories(
        /usr/local/opt/openblas/lib
        /usr/local/opt/libomp/lib
)

add_executable(blocked_ijk_blas main.c)
target_link_libraries(blocked_ijk_blas /usr/local/opt/openblas/lib/libopenblas.a /usr/local/opt/libomp/lib/libomp.a)
