cmake_minimum_required(VERSION 3.15)
project(ijk C)
project(blocked_ijk_blas C)
project(blocked_kij_atlas C)

set(CMAKE_C_STANDARD 11)
find_package (Threads)
add_executable(ijk ijk.c)
add_executable(blocked_ijk_blas blocked_ijk_blas.c)
add_executable(blocked_kij_atlas blocked_kij_atlas.c)
if("${CMAKE_SYSTEM}" MATCHES "Linux")
    include_directories(
            /usr/include/x86_64-linux-gnu
    )
    link_directories(
            /usr/lib/x86_64-linux-gnu/openblas-serial
            /usr/lib/x86_64-linux-gnu/atlas
    )
    target_link_libraries(blocked_ijk_blas /usr/lib/x86_64-linux-gnu/openblas-serial/libblas.a)
    target_link_libraries(blocked_kij_atlas /usr/lib/x86_64-linux-gnu/atlas/libblas.a)
else()
    # THIS IS ONLY WORKS ON MY SYSTEM, PLEASE FIGURE OUT YOUR PATHS YOURSELF
    include_directories(
            /usr/local/opt/openblas/include
            /usr/local/opt/libomp/include
    )
    link_directories(
            /usr/local/opt/openblas/lib
            /usr/local/opt/libomp/lib
    )
    target_link_libraries(blocked_ijk_blas /usr/local/opt/openblas/lib/libopenblas.a /usr/local/opt/libomp/lib/libomp.a)
    target_link_libraries(blocked_kij_atlas /usr/local/opt/openblas/lib/libopenblas.a /usr/local/opt/libomp/lib/libomp.a)
endif()
target_link_libraries (ijk ${CMAKE_THREAD_LIBS_INIT} m)
target_link_libraries (blocked_ijk_blas ${CMAKE_THREAD_LIBS_INIT} m)
target_link_libraries (blocked_kij_atlas ${CMAKE_THREAD_LIBS_INIT} m)