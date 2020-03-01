//
// Created by Malogulko, Alexey on 01/03/2020.
// This is just shared code for other matrix multipliers
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int SQUARE = 2;
const double NUM_MAX = 10.0;

struct matrixInfo {
    double *mxPtr;
    int size;
};

// Prints row-wise blocked matrix
void print_matrix_blocked_rows(double *matrix, int size, int block_size) {
    for (int i = 0; i < (size / block_size); i++) { // Block row
        for (int bi = 0; bi < block_size; bi++) { // In-Block row
            printf("[ ");
            for (int j = 0; j < (size / block_size); j++) { // Block column
                for (int bj = 0; bj < block_size; bj++) { // In-Block column
                    printf("%f ",
                           *(matrix + j * block_size * block_size + i * size * block_size + bi * block_size + bj));
                }
            }
            printf("]\n");
        }
    }
}

// Prints col-wise blocked matrix
void print_matrix_blocked_cols(double *matrix, int size, int block_size) {
    for (int i = 0; i < (size / block_size); i++) {
        for (int bi = 0; bi < block_size; bi++) {
            printf("[ ");
            for (int j = 0; j < (size / block_size); j++) {
                for (int bj = 0; bj < block_size; bj++) { // In-Block column
                    printf("%f ",
                           *(matrix + bj * block_size + j * size * block_size + bi + i * block_size * block_size));
                }
            }
            printf("]\n");
        }
    }
}


// Prints col-wise blocked matrix
void print_matrix_blocked_cols_in_rows(double *matrix, int size, int block_size) {
    for (int i = 0; i < (size / block_size); i++) { // Block row
        for (int bi = 0; bi < block_size; bi++) { // In-Block row
            printf("[ ");
            for (int j = 0; j < (size / block_size); j++) { // Block column
                for (int bj = 0; bj < block_size; bj++) { // In-Block column
                    printf("%f ",
                           *(matrix + bj * block_size + i * size * block_size + bi  + j * block_size * block_size));
                }
            }
            printf("]\n");
        }
    }
}

void parse_matrix_size(int argc, char *argv[], int *size) {
    if (argc >= 2) {
        if (sscanf(argv[1], "%i", size) != 1) {
            fprintf(stderr, "error - not an integer");
            exit(1);
        }
    } else {
        printf("Please use <matrix size>");
        exit(1);
    }
}

// Parses args
void parse_args(int argc, char *argv[], int *size, int *block_size) {
    if (argc >= 3) {
        if (sscanf(argv[1], "%i", size) != 1) {
            fprintf(stderr, "error - not an integer");
            exit(1);
        }
        if (sscanf(argv[2], "%i", block_size) != 1) {
            fprintf(stderr, "error - not an integer");
            exit(1);
        }
    } else {
        printf("Please use <matrix size> <block size> args");
        exit(1);
    }
}

// Allocates matrix memory
double *matrix_malloc(int size) {
    return (double *) malloc((int) pow(size, SQUARE) * sizeof(double));
}

// Generates SQUARE matrix of the size x size
void *random_matrix(void *input) {
    double *mx;
    int size;
    struct matrixInfo *info = (struct matrixInfo *) input;
    mx = info->mxPtr;
    size = info->size;
    for (int i = 0; i < size * size; i++) {
        *(mx + i) = (double) rand() / RAND_MAX * (NUM_MAX * 2) - NUM_MAX;
    }
    return NULL;
}