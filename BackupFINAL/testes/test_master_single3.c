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

#define abs(x) ((x) < 0 ? -(x) : (x))

int main(int argc, char *argv[])
{
    // Read command line arguments

    int n = atoi(argv[1]);
    int seed = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    int num_points = n * n;

    printf("n: %d | seed: %d | num_threads: %d\n", n, seed, num_threads);

    srand(seed);

    // Global variables
    int min_manhattan_per_point, min_manhattan = INT_MAX, sum_min_manhattan = 0;
    int max_manhattan_per_point, max_manhattan = INT_MIN, sum_max_manhattan = 0;
    double min_euclidean_per_point, min_euclidean = DBL_MAX, sum_min_euclidean = 0.0;
    double max_euclidean_per_point, max_euclidean = DBL_MIN, sum_max_euclidean = 0.0;

    // Temporary vector for Min and Max of each point on each process
    int *min_manhattan_per_point_vector = (int *)malloc(sizeof(int) * (num_points / num_threads));
    int *max_manhattan_per_point_vector = (int *)malloc(sizeof(int) * (num_points / num_threads));
    double *min_euclidean_per_point_vector = (double *)malloc(sizeof(double) * (num_points / num_threads));
    double *max_euclidean_per_point_vector = (double *)malloc(sizeof(double) * (num_points / num_threads));

    for(int i = 0; i < (num_points / num_threads); i++)
    {
        min_manhattan_per_point_vector[i] = INT_MAX;
        max_manhattan_per_point_vector[i] = INT_MIN;
        min_euclidean_per_point_vector[i] = DBL_MAX;
        max_euclidean_per_point_vector[i] = DBL_MIN;
    }

    // Initialize MPI
    int num_nodes, rank, src, dest, tag = 0, ret;
    MPI_Status status;
    
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("Aviso: O ambiente MPI não suporta multithreading\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Each process will have vector of size %d\n", (num_points / num_nodes));

    // Node 0 generates x,y and z and distribute them equally to all nodes
    int *x = (int *)malloc(sizeof(int) * num_points / num_nodes);
    int *y = (int *)malloc(sizeof(int) * num_points / num_nodes);
    int *z = (int *)malloc(sizeof(int) * num_points / num_nodes);

    // Create aux vars and fill then with current values of x, y and z
    int *x_aux = (int *)malloc(sizeof(int) * num_points / num_nodes);
    int *y_aux = (int *)malloc(sizeof(int) * num_points / num_nodes);
    int *z_aux = (int *)malloc(sizeof(int) * num_points / num_nodes);

    // Rank 0 generate and distribute the numbers between each process
    if (rank == 0)
    {
        // Generate the X coordinates of the points and send them to the other nodes
        // First generate X for himself and then for the other nodes
        for (int i = 0; i < (num_points / num_nodes); i++)
        {
            x[i] = rand() % 100;
        }

        for (int i = 1; i < num_nodes; i++)
        {
            for (int j = 0; j < (num_points / num_nodes); j++)
            {
                x_aux[j] = rand() % 100;
            }
            MPI_Send(x_aux, (num_points / num_nodes), MPI_INT, i, tag, MPI_COMM_WORLD);
        }

        // Generate the Y coordinates of the points and send them to the other nodes
        // First generate Y for himself and then for the other nodes
        for (int i = 0; i < (num_points / num_nodes); i++)
        {
            y[i] = rand() % 100;
        }

        for (int i = 1; i < num_nodes; i++)
        {
            for (int j = 0; j < (num_points / num_nodes); j++)
            {
                y_aux[j] = rand() % 100;
            }
            MPI_Send(y_aux, (num_points / num_nodes), MPI_INT, i, tag, MPI_COMM_WORLD);
        }

        // Generate the Z coordinates of the points and send them to the other nodes
        // First generate Z for himself and then for the other nodes
        for (int i = 0; i < (num_points / num_nodes); i++)
        {
            z[i] = rand() % 100;
        }

        for (int i = 1; i < num_nodes; i++)
        {
            for (int j = 0; j < (num_points / num_nodes); j++)
            {
                z_aux[j] = rand() % 100;
            }
            MPI_Send(z_aux, (num_points / num_nodes), MPI_INT, i, tag, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(x, num_points / num_nodes, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        fflush(0);
        MPI_Recv(y, num_points / num_nodes, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        fflush(0);
        MPI_Recv(z, num_points / num_nodes, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        fflush(0);

        printf("\n\nRank %d received values:\n", rank);

        printf("\nVector X of rank %d:\n", rank);
        for (int i = 0; i < num_points / num_nodes; i++)
        {
            printf("%d[X%d] ", x[i], rank);
        }

        printf("\nVector Y of rank %d:\n", rank);
        for (int i = 0; i < num_points / num_nodes; i++)
        {
            printf("%d[Y%d] ", y[i], rank);
        }

        printf("\nVector Z of rank %d:\n", rank);
        for (int i = 0; i < num_points / num_nodes; i++)
        {
            printf("%d[Z%d] ", z[i], rank);
        }
    }

    // The first part, of sending all the needed information to the processes is finished
    printf("\n\n\n");

    /*for (int i = 0; i < num_points / num_nodes; i++)
    {
        x_aux[i] = x[i];
        y_aux[i] = y[i];
        z_aux[i] = z[i];
    }*/

    // One thread will be responsible for receiving messages and sending the current values of x, y and z to the node that requested it
    // The other threads will calculate the distances and update the global variables
    int i, j, k;

    #pragma omp parallel num_threads(num_threads) default(shared)
    {
        // 3 -> 2, 1, 0
        // 2 -> 1, 0
        // 1 -> 0
        // 0 -> nada
        #pragma omp for nowait
        for (int i = rank - 1; i >= 0; i--)
        {
            // Receive message from any node requesting mine x,y and z and send to it
            MPI_Recv(&src, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);

            printf("Rank %d received a request from rank %d\n", rank, src);

            MPI_Send(x, (num_points / num_nodes), MPI_INT, i, i + 1, MPI_COMM_WORLD);
            MPI_Send(y, (num_points / num_nodes), MPI_INT, i, i + 2, MPI_COMM_WORLD);
            MPI_Send(z, (num_points / num_nodes), MPI_INT, i, i + 3, MPI_COMM_WORLD);
        }

        #pragma omp for nowait
        for (int i = rank; i < num_nodes; i++)
        {
            if(i == rank)
            {
                for(int j = 0; j < (num_points / num_nodes); j++)
                {
                    #pragma omp simd reduction(min:min_manhattan) reduction(max:max_manhattan) reduction(min:min_euclidean) reduction(max:max_euclidean)
                    for(int k = j+1; k < (num_points / num_nodes); k++)
                    {
                        int manhattan = abs(x[j] - x[k]) + abs(y[j] - y[k]) + abs(z[j] - z[k]);
                        double euclidean = sqrt(pow(x[j] - x[k], 2) + pow(y[j] - y[k], 2) + pow(z[j] - z[k], 2));
                        printf("Manhattan: %d | Euclidean: %lf\n", manhattan, euclidean);

                        if(manhattan < min_manhattan)
                            min_manhattan = manhattan;
                        if(manhattan > max_manhattan)
                            max_manhattan = manhattan;
                        if(euclidean < min_euclidean)
                            min_euclidean = euclidean;
                        if(euclidean > max_euclidean)
                            max_euclidean = euclidean;
                    }
                    if(min_manhattan < min_manhattan_per_point_vector[j])
                        min_manhattan_per_point_vector[j] = min_manhattan;
                    if(max_manhattan > max_manhattan_per_point_vector[j])
                        max_manhattan_per_point_vector[j] = max_manhattan;
                    if(min_euclidean < min_euclidean_per_point_vector[j])
                        min_euclidean_per_point_vector[j] = min_euclidean;
                    if(max_euclidean > max_euclidean_per_point_vector[j])
                        max_euclidean_per_point_vector[j] = max_euclidean;
                }
            }
            else
            {
                printf("Rank %d will request data from rank %d\n", rank, i);
                MPI_Send(&rank, 1, MPI_INT, i, tag, MPI_COMM_WORLD);

                MPI_Recv(x_aux, (num_points / num_nodes), MPI_INT, i, rank + 1, MPI_COMM_WORLD, &status);
                printf("Rank %d recieved X from rank %d\n", rank, i);

                MPI_Recv(y_aux, (num_points / num_nodes), MPI_INT, i, rank + 2, MPI_COMM_WORLD, &status);
                printf("Rank %d recieved Y from rank %d\n", rank, i);

                MPI_Recv(z_aux, (num_points / num_nodes), MPI_INT, i, rank + 3, MPI_COMM_WORLD, &status);
                printf("Rank %d recieved Z from rank %d\n", rank, i);

                for(int j = 0; j < (num_points / num_nodes); j++)
                {
                    #pragma omp simd reduction(min:min_manhattan) reduction(max:max_manhattan) reduction(min:min_euclidean) reduction(max:max_euclidean)
                    for(int k = 0; k < (num_points / num_nodes); k++)
                    {
                        int manhattan = abs(x[j] - x_aux[k]) + abs(y[j] - y_aux[k]) + abs(z[j] - z_aux[k]);
                        double euclidean = sqrt(pow(x[j] - x_aux[k], 2) + pow(y[j] - y_aux[k], 2) + pow(z[j] - z_aux[k], 2));
                        printf("Manhattan: %d | Euclidean: %lf\n", manhattan, euclidean);

                        if(manhattan < min_manhattan)
                            min_manhattan = manhattan;
                        if(manhattan > max_manhattan)
                            max_manhattan = manhattan;
                        if(euclidean < min_euclidean)
                            min_euclidean = euclidean;
                        if(euclidean > max_euclidean)
                            max_euclidean = euclidean;
                    }
                    if(min_manhattan < min_manhattan_per_point_vector[j])
                        min_manhattan_per_point_vector[j] = min_manhattan;
                    if(max_manhattan > max_manhattan_per_point_vector[j])
                        max_manhattan_per_point_vector[j] = max_manhattan;
                    if(min_euclidean < min_euclidean_per_point_vector[j])
                        min_euclidean_per_point_vector[j] = min_euclidean;
                    if(max_euclidean > max_euclidean_per_point_vector[j])
                        max_euclidean_per_point_vector[j] = max_euclidean;
                }
            }
        }
    }

    for(int i = 0; i < (num_points / num_nodes); i++)
    {
        printf("Rank %d: min_manhattan_per_point[%d] = %d\n", rank, i, min_manhattan_per_point_vector[i]);
        printf("Rank %d: max_manhattan_per_point[%d] = %d\n", rank, i, max_manhattan_per_point_vector[i]);
        printf("Rank %d: min_euclidean_per_point[%d] = %lf\n", rank, i, min_euclidean_per_point_vector[i]);
        printf("Rank %d: max_euclidean_per_point[%d] = %lf\n", rank, i, max_euclidean_per_point_vector[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Gather dos mins, max, sumMin, sumMax

    printf("Distância de Manhattan mínima: %d (soma min: %d) e máxima: %d (soma max: %d).\n", min_manhattan, sum_min_manhattan, max_manhattan, sum_max_manhattan);
    printf("Distância Euclidiana mínima: %.2lf (soma min: %.2lf) e máxima: %.2lf (soma max: %.2lf).\n", min_euclidean, sum_min_euclidean, max_euclidean, sum_max_euclidean);

    // Free allocation
    free(x);
    x = NULL;
    free(y);
    y = NULL;
    free(z);
    z = NULL;
    free(x_aux);
    x_aux = NULL;
    free(y_aux);
    y_aux = NULL;
    free(z_aux);
    z_aux = NULL;

    free(min_manhattan_per_point_vector);
    min_manhattan_per_point_vector = NULL;
    free(max_manhattan_per_point_vector);
    max_manhattan_per_point_vector = NULL;
    free(min_euclidean_per_point_vector);
    min_euclidean_per_point_vector = NULL;
    free(max_euclidean_per_point_vector);
    max_euclidean_per_point_vector = NULL;

    // Finalize MPI
    fflush(0);
    if (MPI_Finalize() != MPI_SUCCESS)
        return 1;

    return 0;
}