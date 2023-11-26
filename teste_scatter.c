#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <omp.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    // Read command line arguments
    int n = atoi(argv[1]);
    int seed = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    int num_points = n * n;

    srand(seed);
    omp_set_nested(1);

    // Initialize MPI
    int num_nodes, rank;
    MPI_Status status;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // printf("%d %d\n", num_points, num_nodes);
    int num_points_per_node = num_points / num_nodes;

    // Global variables
    int local_min_manhattan = INT_MAX, sum_min_manhattan = 0;
    int local_max_manhattan = INT_MIN, sum_max_manhattan = 0;
    double local_min_euclidean = DBL_MAX, sum_min_euclidean = 0.0;
    double local_max_euclidean = DBL_MIN, sum_max_euclidean = 0.0;

    // Allocate memory for x, y, z in all processes
    int *x = (int *)malloc(sizeof(int) * num_points_per_node);
    int *y = (int *)malloc(sizeof(int) * num_points_per_node);
    int *z = (int *)malloc(sizeof(int) * num_points_per_node);

    // Allocate memory for x, y, z in master process
    int *x_master, *y_master, *z_master;
    if (rank == 0) {
        x_master = (int *)malloc(sizeof(int) * num_points);
        y_master = (int *)malloc(sizeof(int) * num_points);
        z_master = (int *)malloc(sizeof(int) * num_points);

        // Generate the X, Y, Z coordinates of the points
        for (int i = 0; i < num_points; i++) {
            x_master[i] = rand() % 100;
            y_master[i] = rand() % 100;
            z_master[i] = rand() % 100;
            printf("%d ",x_master[i]);
            if(i%num_points_per_node == 0 && i != 0){
                printf("\n");
            }
        }
        printf("\n\n");

    }
    // // Scatter the generated points to all processes
    // MPI_Scatter(x_master, num_points, MPI_INT, x, num_points_per_node, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Scatter(y_master, num_points, MPI_INT, y, num_points_per_node, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Scatter(z_master, num_points, MPI_INT, z, num_points_per_node, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(x_master, num_points_per_node, MPI_INT, x, num_points_per_node, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(y_master, num_points_per_node, MPI_INT, y, num_points_per_node, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(z_master, num_points_per_node, MPI_INT, z, num_points_per_node, MPI_INT, 0, MPI_COMM_WORLD);


    for(int i = 0 ; i < num_points_per_node ; i++){
        printf("%d ",x[i]);
    }
    printf("\n");

    // Rest of the computation ...

    // Free memory
    free(x);
    free(y);
    free(z);
    if (rank == 0) {
        free(x_master);
        free(y_master);
        free(z_master);
    }

    MPI_Finalize();
    return 0;
}