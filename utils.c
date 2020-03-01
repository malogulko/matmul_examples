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