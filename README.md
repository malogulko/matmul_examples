# What?

This code implement blocked(tiled) IJK matrix multiplication algorithm backed by cblas calls for an actual matrix multiplication.

The general idea of this particular implementation is to arrange the memory into stripes in the way that the block is always represented as a consecutive memory blocks.

# How to use

## Building 

This is a regular CMake project, just follow the steps below:

```
~$ cmake .
-- Configuring done
-- Generating done
-- Build files have been written to: ~/blocked_ijk_blas
~$ make
 Scanning dependencies of target blocked_ijk_blas
 [ 50%] Building C object CMakeFiles/blocked_ijk_blas.dir/main.c.o
 [100%] Linking C executable blocked_ijk_blas
 [100%] Built target blocked_ijk_blas
```

## Running

The binary expects two arguments:
1. Size of the matrix
1. Size of the block

Where the division of "Size of the matrix" by "Size of the block" must leave no remainder.

See an example run:

```
~$ ./blocked_ijk_blas 100 10
100;10;496
```

Where we try to multiply two square 100x100 matrices with block size 10.
The output shows that this operation took 496 milliseconds.