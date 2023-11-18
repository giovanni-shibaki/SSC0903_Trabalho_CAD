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

int manhattan_distance(int x1, int y1, int z1, int x2, int y2, int z2) 
{
    return abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2);
}
double euclidean_distance(int x1, int y1, int z1, int x2, int y2, int z2) 
{
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
}

int main(int argc, char *argv[])
{
    // Read command line arguments

    int n = argv[1];
    int seed = argv[2];
    int num_threads = argv[3];
    int num_points = n*n;

    srand(seed);

    // Global variables

    int min_manhattan_per_point, min_manhattan = INT_MAX, sum_min_manhattan = 0;
    int max_manhattan_per_point, max_manhattan = 0, sum_max_manhattan = 0;
    double min_euclidean_per_point, min_euclidean = DBL_MAX, sum_min_euclidean = 0.0;
    double max_euclidean_per_point, max_euclidean = 0.0, sum_max_euclidean = 0.0;

    // Initialize MPI

    int num_nodes, rank, src, dest, tag = 0, ret;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Node 0 generates x,y and z and distribute them equally to all nodes
    int *x = (int *)malloc(sizeof(int) * num_points/num_nodes);
    int *y = (int *)malloc(sizeof(int) * num_points/num_nodes);
    int *z = (int *)malloc(sizeof(int) * num_points/num_nodes);

    if(rank == 0)
    {
        for(int i = 0; i < num_nodes; i++)
        {
            for (int j = 0; j < num_points/num_nodes; j++)
            {
                x[j] = rand() % 100;
                y[j] = rand() % 100;
                z[j] = rand() % 100;
            }

            if(i == num_nodes - 1)
                break;

            MPI_Send(x, num_points/num_nodes, MPI_INT, i, tag, MPI_COMM_WORLD);
            MPI_Send(y, num_points/num_nodes, MPI_INT, i, tag, MPI_COMM_WORLD);
            MPI_Send(z, num_points/num_nodes, MPI_INT, i, tag, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(x, num_points/num_nodes, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(y, num_points/num_nodes, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(z, num_points/num_nodes, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
    }

    // Create aux vars and fill then with current values of x, y and z

    int *x_aux = (int *)malloc(sizeof(int) * num_points/num_nodes);
    int *y_aux = (int *)malloc(sizeof(int) * num_points/num_nodes);
    int *z_aux = (int *)malloc(sizeof(int) * num_points/num_nodes);

    for(int i = 0; i < num_points/num_nodes; i++)
    {
        x_aux[i] = x[i];
        y_aux[i] = y[i];
        z_aux[i] = z[i];
    }

    // One thread will be responsible for receiving messages and sending the current values of x, y and z to the node that requested it
    // The other threads will calculate the distances and update the global variables

    #pragma omp parallel num_threads(4)   
    {
        #pragma omp master
        {
            while(true)
            {
                // Receive message from any node requesting mine x,y and z and send to it
                MPI_Recv(&src, 1, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);

                MPI_Send(x, num_points/num_nodes, MPI_INT, src, tag, MPI_COMM_WORLD);
                MPI_Send(y, num_points/num_nodes, MPI_INT, src, tag, MPI_COMM_WORLD);
                MPI_Send(z, num_points/num_nodes, MPI_INT, src, tag, MPI_COMM_WORLD);
            }
        }

        #pragma omp single
        {
            // For every point in mine x, y and z calculate the distances, fetching when needed and updating the global variables
            for(int p = 0; p < num_points/num_nodes; p++)
            {
                for(int i = rank; i < num_nodes; i++)
                {
                    if(i != rank)
                    {
                        MPI_Send(&rank, 1, MPI_INT, i, tag, MPI_COMM_WORLD);

                        MPI_Recv(x_aux, num_points/num_nodes, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
                        MPI_Recv(y_aux, num_points/num_nodes, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
                        MPI_Recv(z_aux, num_points/num_nodes, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
                    }

                    int start_j = i == rank ? p + 1 : 0;

                    for(int j = start_j; j < num_points/num_nodes; j++)
                    {
                        int manhattan_dist = manhattan_distance(x[p], y[p], z[p], x_aux[j], y_aux[j], z_aux[j]);
                        double euclidean_dist = euclidean_distance(x[p], y[p], z[p], x_aux[j], y_aux[j], z_aux[j]);

                        if(manhattan_dist < min_manhattan_per_point)
                            min_manhattan_per_point = manhattan_dist;

                        if(manhattan_dist > max_manhattan_per_point)
                            max_manhattan_per_point = manhattan_dist;

                        if(euclidean_dist < min_euclidean_per_point)
                            min_euclidean_per_point = euclidean_dist;

                        if(euclidean_dist > max_euclidean_per_point)
                            max_euclidean_per_point = euclidean_dist;
                    }

                    if(min_manhattan_per_point < min_manhattan)
                        min_manhattan = min_manhattan_per_point;

                    if(max_manhattan_per_point > max_manhattan)
                        max_manhattan = max_manhattan_per_point;

                    if(min_euclidean_per_point < min_euclidean)
                        min_euclidean = min_euclidean_per_point;

                    if(max_euclidean_per_point > max_euclidean)
                        max_euclidean = max_euclidean_per_point;

                    sum_min_manhattan += min_manhattan_per_point;
                    sum_max_manhattan += max_manhattan_per_point;
                    sum_min_euclidean += min_euclidean_per_point;
                    sum_max_euclidean += max_euclidean_per_point;
                }
            }
            
        }
    }

    // Send results to node 0
    if(rank != 0)
    {
        MPI_Send(&min_manhattan, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
        MPI_Send(&max_manhattan, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
        MPI_Send(&min_euclidean, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        MPI_Send(&max_euclidean, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        MPI_Send(&sum_min_manhattan, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
        MPI_Send(&sum_max_manhattan, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
        MPI_Send(&sum_min_euclidean, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        MPI_Send(&sum_max_euclidean, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    }
    else
    {
        int aux_min_manhattan, aux_sum_min_manhattan;
        int aux_max_manhattan, aux_sum_max_manhattan;
        double aux_min_euclidean_per_point, aux_min_euclidean, aux_sum_min_euclidean;
        double aux_max_euclidean_per_point, aux_max_euclidean, aux_sum_max_euclidean;

        for(int i = 1; i < num_nodes; i++)
        {
            MPI_Recv(aux_min_manhattan, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(aux_max_manhattan, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(aux_min_euclidean, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(aux_max_euclidean, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(aux_sum_min_manhattan, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(aux_sum_max_manhattan, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(aux_sum_min_euclidean, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(aux_sum_max_euclidean, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);

            if(aux_min_manhattan < min_manhattan)
                min_manhattan = aux_min_manhattan;

            if(aux_max_manhattan > max_manhattan)
                max_manhattan = aux_max_manhattan;

            if(aux_min_euclidean < min_euclidean)
                min_euclidean = aux_min_euclidean;

            if(aux_max_euclidean > max_euclidean)
                max_euclidean = aux_max_euclidean;

            sum_min_manhattan += aux_sum_min_manhattan;
            sum_max_manhattan += aux_sum_max_manhattan;
            sum_min_euclidean += aux_sum_min_euclidean;
            sum_max_euclidean += aux_sum_max_euclidean;
        }
    }

    // Print results

    if(rank == 0)
    {
        printf("Distância de Manhattan mínima: %d (soma min: %d) e máxima: %d (soma max: %d).\n", min_manhattan, sum_min_manhattan, max_manhattan, sum_max_manhattan);
        printf("Distância Euclidiana mínima: %.2lf (soma min: %.2lf) e máxima: %.2lf (soma max: %.2lf).\n", min_euclidean, sum_min_euclidean, max_euclidean, sum_max_euclidean);
    }

    // Free allocation

    free(x);
    free(y);
    free(z);
    free(x_aux);
    free(y_aux);
    free(z_aux);

    // Finalize MPI

    fflush(0);
    
    if(MPI_Finalize() != MPI_SUCCESS)
        return 1;

    return 0;
}