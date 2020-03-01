//
// Created by Malogulko, Alexey on 01/03/2020.
//

#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include "utils.c"

/**
 * Classic IJK matrix multiplication
 * @param matrix_a - row-wise addressed matrix_a
 * @param matrix_b - column-wise addressed matrix_b
 * @param matrix_c - row-wise addressed matrix_c
 * @param matrix_size - size of the matrix
 */
void ijk(double *matrix_a, double *matrix_b, double *matrix_c, int matrix_size) {
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            double *c_sum = matrix_c + i * matrix_size + j;
            for (int k = 0; k < matrix_size; k++) {
                *(c_sum) += *(matrix_a + i * matrix_size + k) * *(matrix_b + j * matrix_size + k);
            }
        }
    }
}

/**
 * 4x4 matrix with block size 2 represented in memory as:
 *
 * 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16
 *
 * Matrices A and C are stored in row-wise format:
 *
 * 01 02 03 04
 * 05 06 07 08
 * 09 10 11 12
 * 13 14 15 16
 *
 * At the same time, matrix B blocks stored in column-wise format:
 *
 * 01 05 09 13
 * 02 06 10 14
 * 03 07 11 15
 * 04 08 12 16
 *
 */
int main(int argc, char *argv[]) {
    int size;
    pthread_t thread_ids[2];
    struct timespec start, end;
    //srand(time(0));
    parse_matrix_size(argc, argv, &size);
    double *matrix_a = matrix_malloc(size);
    double *matrix_b = matrix_malloc(size);
    // Initialize info objs
    struct matrixInfo matrix_a_info = {.size = size, .mxPtr = matrix_a};
    struct matrixInfo matrix_b_info = {.size = size, .mxPtr = matrix_b};
    // Fill matrices with random values
    pthread_create(&thread_ids[0], NULL, random_matrix, &matrix_a_info);
    pthread_create(&thread_ids[1], NULL, random_matrix, &matrix_b_info);
    double *matrix_c = matrix_malloc(size); // result matrix
    for (int tn = 0; tn < 2; tn++) {
        (void) pthread_join(thread_ids[tn], 0);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    ijk(matrix_a, matrix_b, matrix_c, size);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000; // microseconds
    printf("%d;%llu\n", size, delta_us);
    return 0;
}