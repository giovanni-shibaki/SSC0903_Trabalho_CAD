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

    // The first part, of sending all the needed information to the processes is finished
    //printf("\n\n\n");

    /*for (int i = 0; i < num_points_per_node; i++)
    {
        x_aux[i] = x[i];
        y_aux[i] = y[i];
        z_aux[i] = z[i];
    }*/

    // One thread will be responsible for receiving messages and sending the current values of x, y and z to the node that requested it
    // The other threads will calculate the distances and update the global variables
    int i, j, k;

    double *manhattan_euclidean_min_max = (double *)malloc(sizeof(double) * (4 * num_points_per_node));

    int min_manhattan = INT_MAX;
    int max_manhattan = INT_MIN;
    double min_euclidean = DBL_MAX;
    double max_euclidean = DBL_MIN;

    // Make a broadcast of the values of x, y and z to all nodes
    int point[3] = {};
    int *points = (int *)malloc(sizeof(int) * 3 * num_points_per_node);

    #pragma omp parallel num_threads(num_threads) private(point)
    {
        #pragma omp for
        for(int i=0; i < num_points_per_node; i++)
        {
            points[3*i] = x[i];
            points[3*i+1] = y[i];
            points[3*i+2] = z[i];
        }
        
        #pragma omp for
        for(int i = rank + 1; i < num_nodes; i++)
        {
            MPI_Send(points, 3 * num_points_per_node, MPI_INT, i, rank, MPI_COMM_WORLD);
        }
    }

    double manhattan_euclidean_min_max_point[4] = {};

    #pragma omp parallel num_threads(num_threads) private(i, j, k, point, manhattan_euclidean_min_max_point, min_manhattan, max_manhattan, min_euclidean, max_euclidean) default(shared)
    {
        #pragma omp for reduction(min : local_min_manhattan, local_min_euclidean) reduction(max : local_max_manhattan, local_max_euclidean) \
        reduction(+:sum_min_manhattan, sum_max_manhattan, sum_min_euclidean, sum_max_euclidean)
        for(int i = 0; i < num_points_per_node; i++)
        {
            min_manhattan = INT_MAX;
            max_manhattan = INT_MIN;
            min_euclidean = DBL_MAX;
            max_euclidean = DBL_MIN;

            #pragma omp simd reduction(min : min_manhattan) reduction(max : max_manhattan) reduction(min : min_euclidean) reduction(max : max_euclidean)
            for(int j = i+1; j < num_points_per_node; j++)
            {
                int manhattan = abs(x[i] - x[j]) + abs(y[i] - y[j]) + abs(z[i] - z[j]);
                double euclidean = sqrt(pow(x[i] - x[j], 2) + pow(y[i] - y[j], 2) + pow(z[i] - z[j], 2));

                // Now, check if the results are lower than the minimum and higher than the maximum
                if(manhattan < min_manhattan)
                    min_manhattan = manhattan;
                if(manhattan > max_manhattan)
                    max_manhattan = manhattan;
                if(euclidean < min_euclidean)
                    min_euclidean = euclidean;
                if(euclidean > max_euclidean)
                    max_euclidean = euclidean;
            }

            //printf("Ponto (%d, %d, %d) - min_manhattan: %d, max_manhattan: %d, min_euclidean: %lf, max_euclidean: %lf\n\n", x[i], y[i], z[i], min_manhattan, max_manhattan, min_euclidean, max_euclidean);

            for(int k = rank + 1; k < num_nodes; k++)
            {
                double manhattan_euclidean_min_max_point[4] = {};
                MPI_Recv(&manhattan_euclidean_min_max_point, 4, MPI_DOUBLE, k, i+rank, MPI_COMM_WORLD, &status);

                //printf("%d -> %d - P %d, recebeu: (%lf, %lf, %lf, %lf)\n", k, rank, i, manhattan_euclidean_min_max_point[0], manhattan_euclidean_min_max_point[1], manhattan_euclidean_min_max_point[2], manhattan_euclidean_min_max_point[3]);

                if(manhattan_euclidean_min_max_point[0] < min_manhattan)
                    min_manhattan = manhattan_euclidean_min_max_point[0];
                if(manhattan_euclidean_min_max_point[1] > max_manhattan)
                    max_manhattan = manhattan_euclidean_min_max_point[1];
                if(manhattan_euclidean_min_max_point[2] < min_euclidean)
                    min_euclidean = manhattan_euclidean_min_max_point[2];
                if(manhattan_euclidean_min_max_point[3] > max_euclidean)
                    max_euclidean = manhattan_euclidean_min_max_point[3];
            }

            //printf("Ponto %d (%d, %d, %d) rank %d - min_manhattan: %d, max_manhattan: %d, min_euclidean: %lf, max_euclidean: %lf\n\n", i, x[i], y[i], z[i], rank, min_manhattan, max_manhattan, min_euclidean, max_euclidean);

            // Here, we have the minimum and maximum values of the distances of the current point i
            if(min_manhattan < local_min_manhattan)
                local_min_manhattan = min_manhattan;
            if(max_manhattan > local_max_manhattan)
                local_max_manhattan = max_manhattan;
            if(min_euclidean < local_min_euclidean) 
                local_min_euclidean = min_euclidean;
            if(max_euclidean > local_max_euclidean)
                local_max_euclidean = max_euclidean;

            if(min_manhattan != INT_MAX && max_manhattan != INT_MIN && min_euclidean != INT_MAX && max_euclidean != INT_MIN)
            {
                sum_min_manhattan += min_manhattan;
                sum_max_manhattan += max_manhattan;
                sum_min_euclidean += min_euclidean;
                sum_max_euclidean += max_euclidean;
            }
        }

        // Respond to the request of the other nodes
        #pragma omp single
        for (int i = rank - 1; i >= 0; i--)
        {
            MPI_Recv(points, 3 * num_points_per_node, MPI_INT, i, i, MPI_COMM_WORLD, &status);

            for(int j = 0; j < num_points_per_node; j++)
            {            
                min_manhattan = INT_MAX;
                max_manhattan = INT_MIN;
                min_euclidean = DBL_MAX;
                max_euclidean = DBL_MIN;

                // Calculate the distances and send them to the node that requested it
                #pragma omp simd reduction(min : min_manhattan) reduction(max : max_manhattan) reduction(min : min_euclidean) reduction(max : max_euclidean)
                for(int k = 0; k < num_points_per_node; k++)
                {
                    double manhattan = abs(points[3*j] - x[k]) + abs(points[3*j+1] - y[k]) + abs(points[3*j+2] - z[k]);
                    //printf("manhattan: %d\n", manhattan);
                    double euclidean = sqrt(pow(points[3*j] - x[k], 2) + pow(points[3*j+1] - y[k], 2) + pow(points[3*j+2] - z[k], 2));
                    //printf("euclidean: %lf\n", euclidean);

                    // Now, check if the results are lower than the minimum and higher than the maximum
                    if(manhattan < min_manhattan)
                        min_manhattan = manhattan;
                    if(manhattan > max_manhattan)
                        max_manhattan = manhattan;
                    if(euclidean < min_euclidean)
                        min_euclidean = euclidean;
                    if(euclidean > max_euclidean)
                        max_euclidean = euclidean;
                }

                // Send the results to the node that requested it with MPI_Send
                manhattan_euclidean_min_max_point[0] = min_manhattan;
                manhattan_euclidean_min_max_point[1] = max_manhattan;
                manhattan_euclidean_min_max_point[2] = min_euclidean;
                manhattan_euclidean_min_max_point[3] = max_euclidean;

                /*printf("manhattan_euclidean_min_max_point[0]: %lf\n", manhattan_euclidean_min_max_point[0]);
                printf("manhattan_euclidean_min_max_point[1]: %lf\n", manhattan_euclidean_min_max_point[1]);
                printf("manhattan_euclidean_min_max_point[2]: %lf\n", manhattan_euclidean_min_max_point[2]);
                printf("manhattan_euclidean_min_max_point[3]: %lf\n", manhattan_euclidean_min_max_point[3]);*/
                
                MPI_Send(&manhattan_euclidean_min_max_point, 4, MPI_DOUBLE, i, j+i, MPI_COMM_WORLD);
                //printf("Rank %d enviou a %d: %lf %lf %lf %lf referente ao ponto {%d}\n", rank, i, manhattan_euclidean_min_max_point[0], manhattan_euclidean_min_max_point[1], manhattan_euclidean_min_max_point[2], manhattan_euclidean_min_max_point[3], j);
            }
        }
    }

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