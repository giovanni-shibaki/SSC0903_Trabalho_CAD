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

int main(int argc, char *argv[])
{
    // Read command line arguments

    int n = atoi(argv[1]);
    int seed = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    int num_points = n*n;

    printf("n: %d | seed: %d | num_threads: %d\n", n, seed, num_threads);

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

    printf("Each process will have vector of size %d\n", (num_points/num_nodes));

    // Node 0 generates x,y and z and distribute them equally to all nodes
    int *x = (int *)malloc(sizeof(int) * num_points/num_nodes);
    int *y = (int *)malloc(sizeof(int) * num_points/num_nodes);
    int *z = (int *)malloc(sizeof(int) * num_points/num_nodes);

    // Rank 0 generate and distribute the numbers between each process
    if(rank == 0)
    {
        for(int i = 0; i < num_nodes; i++)
        {
            for (int j = 0; j < (num_points/num_nodes); j++)
            {
                x[j] = rand() % 100;
                y[j] = rand() % 100;
                z[j] = rand() % 100;
            }

            if(i == 0)
                continue;

            printf("Rank 0 is sending the data to rank %d\n", i);
            MPI_Send(x, (num_points/num_nodes), MPI_INT, i, tag, MPI_COMM_WORLD);
            MPI_Send(y, (num_points/num_nodes), MPI_INT, i, tag, MPI_COMM_WORLD);
            MPI_Send(z, (num_points/num_nodes), MPI_INT, i, tag, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(x, num_points/num_nodes, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(y, num_points/num_nodes, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(z, num_points/num_nodes, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

        printf("\n\nRank %d received values:\n", rank);

        printf("\nVector X of rank %d:\n", rank);
        for(int i = 0; i < num_points/num_nodes; i++)
        {
            printf("%d[X%d] ", x[i], rank);
        }

        printf("\nVector Y of rank %d:\n", rank);
        for(int i = 0; i < num_points/num_nodes; i++)
        {
            printf("%d[Y%d] ", y[i], rank);
        }

        printf("\nVector Z of rank %d:\n", rank);
        for(int i = 0; i < num_points/num_nodes; i++)
        {
            printf("%d[Z%d] ", z[i], rank);
        }
    }

	
    # The first part, of sending all the needed information to the processes is finished
    printf("\n\n\n\n\n\n");
	
    // Free allocation
    free(x);
    free(y);
    free(z);

    // Finalize MPI
    fflush(0);
    
    if(MPI_Finalize() != MPI_SUCCESS)
        return 1;

    return 0;
}