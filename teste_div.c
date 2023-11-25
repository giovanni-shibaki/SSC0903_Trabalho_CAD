/*
 * Trabalho CAD 1 - Grupo 5
 * Integrantes:
 *   - Giovanni Shibaki Camargo - 11796444
 *   - Matheus Giraldi Alvarenga - 12543669
 *   - Pedro Dias Batista - 10769809
 *   - Pedro Kenzo Muramatsu Carmo - 11796451
 *   - Rafael Sartori Vantin - 12543353
 *
 * Como compilar e rodar:
 *   - mpicc main.c -o main -lm
 *   - mpirun -np 4 ./main 1000 1234 4
 *
 * Exemplo:
 *   - Entrada:
 *       - n = 1000
 *       - seed = 1234
 *       - num_threads = 4
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <omp.h>
#include "mpi.h"

#define abs2(x) ((x) < 0 ? -(x) : (x))

int main(int argc, char *argv[])
{
    // Read command line arguments

    int n = atoi(argv[1]);
    int seed = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    int num_points = n * n;

    //printf("n: %d | seed: %d | num_threads: %d\n", n, seed, num_threads);

    srand(seed);
    omp_set_nested(1);

    // Initialize MPI
    int num_nodes, rank, src, dest, tag = 0, ret;
    MPI_Status status;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        printf("Aviso: O ambiente MPI não suporta multithreading\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int num_points_per_node = num_points / num_nodes;

    int calcs_per_node = num_points_per_node * (num_points_per_node - 1) / 2;

    int total_distance_pairs = calcs_per_node * 2;

    int *distance_pairs = (int *)malloc(sizeof(int) * total_distance_pairs);

    for(int i = 0; i < num_nodes; i++)
    {
        for(int j = i+1; j < num_nodes; j++)
        {
            distance_pairs[2*i] = i;
            distance_pairs[2*i + 1] = j;
        }
    }

    // Global variables
    int local_min_manhattan = INT_MAX, sum_min_manhattan = 0;
    int local_max_manhattan = INT_MIN, sum_max_manhattan = 0;
    double local_min_euclidean = DBL_MAX, sum_min_euclidean = 0.0;
    double local_max_euclidean = DBL_MIN, sum_max_euclidean = 0.0;

    //printf("Each process will have vector of size %d\n", (num_points_per_node));

    // Node 0 generates x,y and z and distribute them equally to all nodes
    int *x = (int *)malloc(sizeof(int) * num_points_per_node);
    int *y = (int *)malloc(sizeof(int) * num_points_per_node);
    int *z = (int *)malloc(sizeof(int) * num_points_per_node);

    // Create aux vars and fill then with current values of x, y and z
    int *x_aux = (int *)malloc(sizeof(int) * num_points_per_node);
    int *y_aux = (int *)malloc(sizeof(int) * num_points_per_node);
    int *z_aux = (int *)malloc(sizeof(int) * num_points_per_node);

    // Rank 0 generate and distribute the numbers between each process
    if (rank == 0)
    {
        // Generate the X coordinates of the points and send them to the other nodes
        // First generate X for himself and then for the other nodes
        for (int i = 0; i < (num_points_per_node); i++)
        {
            x[i] = rand() % 100;
        }

        for (int i = 1; i < num_nodes; i++)
        {
            for (int j = 0; j < (num_points_per_node); j++)
            {
                x_aux[j] = rand() % 100;
            }
            MPI_Send(x_aux, (num_points_per_node), MPI_INT, i, tag, MPI_COMM_WORLD);
        }

        // Generate the Y coordinates of the points and send them to the other nodes
        // First generate Y for himself and then for the other nodes
        for (int i = 0; i < (num_points_per_node); i++)
        {
            y[i] = rand() % 100;
        }

        for (int i = 1; i < num_nodes; i++)
        {
            for (int j = 0; j < (num_points_per_node); j++)
            {
                y_aux[j] = rand() % 100;
            }
            MPI_Send(y_aux, (num_points_per_node), MPI_INT, i, tag, MPI_COMM_WORLD);
        }

        // Generate the Z coordinates of the points and send them to the other nodes
        // First generate Z for himself and then for the other nodes
        for (int i = 0; i < (num_points_per_node); i++)
        {
            z[i] = rand() % 100;
        }

        for (int i = 1; i < num_nodes; i++)
        {
            for (int j = 0; j < (num_points_per_node); j++)
            {
                z_aux[j] = rand() % 100;
            }
            MPI_Send(z_aux, (num_points_per_node), MPI_INT, i, tag, MPI_COMM_WORLD);
        }

        // Descomentar para ver se os pontos foram distribuídos corretamente
        /*printf("\n\nRank %d received values:\n", rank);

        printf("\nVector X of rank %d:\n", rank);
        for (int i = 0; i < num_points_per_node; i++)
        {
            printf("%d[X%d] ", x[i], rank);
        }

        printf("\nVector Y of rank %d:\n", rank);
        for (int i = 0; i < num_points_per_node; i++)
        {
            printf("%d[Y%d] ", y[i], rank);
        }

        printf("\nVector Z of rank %d:\n", rank);
        for (int i = 0; i < num_points_per_node; i++)
        {
            printf("%d[Z%d] ", z[i], rank);
        }*/
    }
    else
    {
        MPI_Recv(x, num_points_per_node, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        fflush(0);
        MPI_Recv(y, num_points_per_node, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        fflush(0);
        MPI_Recv(z, num_points_per_node, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        fflush(0);

        // Descomentar para ver se os pontos foram distribuídos corretamente
        /*printf("\n\nRank %d received values:\n", rank);

        printf("\nVector X of rank %d:\n", rank);
        for (int i = 0; i < num_points_per_node; i++)
        {
            printf("%d[X%d] ", x[i], rank);
        }

        printf("\nVector Y of rank %d:\n", rank);
        for (int i = 0; i < num_points_per_node; i++)
        {
            printf("%d[Y%d] ", y[i], rank);
        }

        printf("\nVector Z of rank %d:\n", rank);
        for (int i = 0; i < num_points_per_node; i++)
        {
            printf("%d[Z%d] ", z[i], rank);
        }*/
    }

    free(x_aux);
    x_aux = NULL;
    free(y_aux);
    y_aux = NULL;
    free(z_aux);
    z_aux = NULL;

    int *local_min_manhattan_per_point = (int *)malloc(sizeof(int) * num_points_per_node);
    int *local_max_manhattan_per_point = (int *)malloc(sizeof(int) * num_points_per_node);
    double *local_min_euclidean_per_point = (double *)malloc(sizeof(double) * num_points_per_node);
    double *local_max_euclidean_per_point = (double *)malloc(sizeof(double) * num_points_per_node);

    for(int i = 0; i < num_points_per_node; i++)
    {
        local_min_manhattan_per_point[i] = INT_MAX;
        local_max_manhattan_per_point[i] = INT_MIN;
        local_min_euclidean_per_point[i] = DBL_MAX;
        local_max_euclidean_per_point[i] = DBL_MIN;
    }

    // Calculate the distances between the points that each node already has
    for(int i = 0; i < num_points_per_node; i++)
    {
        local_min_manhattan = INT_MAX;
        local_max_manhattan = INT_MIN;
        local_min_euclidean = DBL_MAX;
        local_max_euclidean = DBL_MIN;

        for(int j = 0; j < num_points_per_node; j++)
        {
            if(i != j)
            {
                int manhattan_dist = abs2(x[i] - x[j]) + abs2(y[i] - y[j]) + abs2(z[i] - z[j]);
                double euclidean_dist = sqrt(pow(x[i] - x[j], 2) + pow(y[i] - y[j], 2) + pow(z[i] - z[j], 2));

                if(manhattan_dist < local_min_manhattan)
                    local_min_manhattan = manhattan_dist;

                if(manhattan_dist > local_max_manhattan)
                    local_max_manhattan = manhattan_dist;

                if(euclidean_dist < local_min_euclidean)
                    local_min_euclidean = euclidean_dist;

                if(euclidean_dist > local_max_euclidean)
                    local_max_euclidean = euclidean_dist;
            }
        }
        
        if(local_min_manhattan < local_min_manhattan_per_point[i])
            local_min_manhattan_per_point[i] = local_min_manhattan;
        if(local_max_manhattan > local_max_manhattan_per_point[i])
            local_max_manhattan_per_point[i] = local_max_manhattan;
        if(local_min_euclidean < local_min_euclidean_per_point[i])  
            local_min_euclidean_per_point[i] = local_min_euclidean;
        if(local_max_euclidean > local_max_euclidean_per_point[i])
            local_max_euclidean_per_point[i] = local_max_euclidean;
    }

    // Calculate the distances between its points and the points of the other nodes
    


    int global_min_manhattan = 0;
    int global_max_manhattan = 0;
    double global_min_euclidean = 0.0;
    double global_max_euclidean = 0.0;
    int total_sum_min_manhattan = 0;
    int total_sum_max_manhattan = 0;
    double total_sum_min_euclidean = 0.0;
    double total_sum_max_euclidean = 0.0;

    MPI_Reduce(&sum_min_manhattan, &total_sum_min_manhattan, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_max_manhattan, &total_sum_max_manhattan, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_min_euclidean, &total_sum_min_euclidean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_max_euclidean, &total_sum_max_euclidean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_min_manhattan, &global_min_manhattan, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_manhattan, &global_max_manhattan, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min_euclidean, &global_min_euclidean, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_euclidean, &global_max_euclidean, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Distância de Manhattan mínima: %d (soma min: %d) e máxima: %d (soma max: %d).\n", global_min_manhattan, total_sum_min_manhattan, global_max_manhattan, total_sum_max_manhattan);
        printf("Distância Euclidiana mínima: %.2lf (soma min: %.2lf) e máxima: %.2lf (soma max: %.2lf).\n", global_min_euclidean, total_sum_min_euclidean, global_max_euclidean, total_sum_max_euclidean);
    }

    // Free of all point vectors -> Esta parte da memória alocada torna-se agora, inútil
    free(x);
    x = NULL;
    free(y);
    y = NULL;
    free(z);
    z = NULL;

    free(points);
    points = NULL;

    // Finalize MPI
    fflush(0);
    if (MPI_Finalize() != MPI_SUCCESS)
        return 1;

    return 0;
}