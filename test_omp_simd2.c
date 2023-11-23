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

    printf("n: %d | seed: %d | num_threads: %d\n", n, seed, num_threads);

    srand(seed);

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
    int global_min_manhattan = INT_MAX, sum_min_manhattan = 0;
    int global_max_manhattan = INT_MIN, sum_max_manhattan = 0;
    double global_min_euclidean = DBL_MAX, sum_min_euclidean = 0.0;
    double global_max_euclidean = DBL_MIN, sum_max_euclidean = 0.0;

    // Temporary vector for Min and Max of each point on each process
    int *min_manhattan_per_point_vector = (int *)malloc(sizeof(int) * num_points_per_node);
    int *max_manhattan_per_point_vector = (int *)malloc(sizeof(int) * num_points_per_node);
    double *min_euclidean_per_point_vector = (double *)malloc(sizeof(double) * num_points_per_node);
    double *max_euclidean_per_point_vector = (double *)malloc(sizeof(double) * num_points_per_node);

    for (int i = 0; i < num_points_per_node; i++)
    {
        min_manhattan_per_point_vector[i] = INT_MAX;
        max_manhattan_per_point_vector[i] = INT_MIN;
        min_euclidean_per_point_vector[i] = DBL_MAX;
        max_euclidean_per_point_vector[i] = DBL_MIN;
    }

    printf("Each process will have vector of size %d\n", (num_points_per_node));

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

    // The first part, of sending all the needed information to the processes is finished
    MPI_Barrier(MPI_COMM_WORLD);
    printf("\n\n\n");

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

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for
        for(int j = 0; j < num_points_per_node; j++)
        {
            manhattan_euclidean_min_max[4*j] = DBL_MAX;
            manhattan_euclidean_min_max[4*j+1] = DBL_MIN;
            manhattan_euclidean_min_max[4*j+2] = DBL_MAX;
            manhattan_euclidean_min_max[4*j+3] = DBL_MIN;
        }
    }


    double min_manhattan = DBL_MAX;
    double max_manhattan = DBL_MIN;
    double min_euclidean = DBL_MAX;
    double max_euclidean = DBL_MIN;

    for(int i = 0; i < num_points_per_node; i++)
    {
        min_manhattan = DBL_MAX;
        max_manhattan = DBL_MIN;
        min_euclidean = DBL_MAX;
        max_euclidean = DBL_MIN;

        //#pragma omp simd reduction(min : min_manhattan) reduction(max : max_manhattan) reduction(min : min_euclidean) reduction(max : max_euclidean)
        for(int j = i+1; j < num_points_per_node; j++)
        {
            double manhattan = abs(x[i] - x[j]) + abs(y[i] - y[j]) + abs(z[i] - z[j]);
            //printf("manhattan: %d\n", manhattan);
            double euclidean = sqrt(pow(x[i] - x[j], 2) + pow(y[i] - y[j], 2) + pow(z[i] - z[j], 2));
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

        if(min_manhattan < manhattan_euclidean_min_max[4*i])
            manhattan_euclidean_min_max[4*i] = min_manhattan;
        if(max_manhattan > manhattan_euclidean_min_max[4*i+1])
            manhattan_euclidean_min_max[4*i+1] = max_manhattan;
        if(min_euclidean < manhattan_euclidean_min_max[4*i+2])
            manhattan_euclidean_min_max[4*i+2] = min_euclidean;
        if(max_euclidean > manhattan_euclidean_min_max[4*i+3])
            manhattan_euclidean_min_max[4*i+3] = max_euclidean;
        
        
        //printf("manhattan_euclidean_min_max[%d]: %lf\n", 4*i, manhattan_euclidean_min_max[4*i]);
        //printf("manhattan_euclidean_min_max[%d]: %lf\n", 4*i+1, manhattan_euclidean_min_max[4*i+1]);
    }

    // Make a broadcast of the values of x, y and z to all nodes
    int point[3] = {};
    double manhattan_euclidean_min_max_point[4] = {};

    #pragma omp parallel num_threads(1) private(i, j, k, point, manhattan_euclidean_min_max_point) default(shared)
    {
        // Broadcast the values of x, y and z to all nodes
        for(int i = 0; i < num_points_per_node; i++)
        {
            point[0] = x[i];
            point[1] = y[i];
            point[2] = z[i];
            MPI_Bcast(&point, 3, MPI_INT, rank, MPI_COMM_WORLD);
        }

        #pragma omp for collapse(2)
        for(int j = rank + 1; j < num_nodes; j++)
        {
            for(int k = 0; k < num_points_per_node; k++)
            {
                // Recieve two arrays with the minimum and maximum values of the manhattan and euclidian distances
                double manhattan_euclidean_min_max_point[4] = {};
                MPI_Recv(&manhattan_euclidean_min_max_point, 4, MPI_DOUBLE, j, rank, MPI_COMM_WORLD, &status);
                //printf("Rank %d recebeu de %d: %lf %lf %lf %lf - k = %d\n", rank, j, manhattan_euclidean_min_max_point[0], manhattan_euclidean_min_max_point[1], manhattan_euclidean_min_max_point[2], manhattan_euclidean_min_max_point[3], k);

                //printf("Rank %d received from rank %d: %lf %lf %lf %lf\n", rank, j, manhattan_euclidean_min_max_point[0], manhattan_euclidean_min_max_point[1], manhattan_euclidean_min_max_point[2], manhattan_euclidean_min_max_point[3]);
                
                //manhattan_euclidean_min_max[counter++] = manhattan_euclidean_min_max_point[0];
                //manhattan_euclidean_min_max[counter++] = manhattan_euclidean_min_max_point[1];
                //manhattan_euclidean_min_max[counter++] = manhattan_euclidean_min_max_point[2];
                //manhattan_euclidean_min_max[counter++] = manhattan_euclidean_min_max_point[3];

                if(manhattan_euclidean_min_max_point[0] < manhattan_euclidean_min_max[4*k])
                    manhattan_euclidean_min_max[4*k] = manhattan_euclidean_min_max_point[0];
                if(manhattan_euclidean_min_max_point[1] > manhattan_euclidean_min_max[4*k+1])
                    manhattan_euclidean_min_max[4*k+1] = manhattan_euclidean_min_max_point[1];
                if(manhattan_euclidean_min_max_point[2] < manhattan_euclidean_min_max[4*k+2])
                    manhattan_euclidean_min_max[4*k+2] = manhattan_euclidean_min_max_point[2];
                if(manhattan_euclidean_min_max_point[3] > manhattan_euclidean_min_max[4*k+3])
                    manhattan_euclidean_min_max[4*k+3] = manhattan_euclidean_min_max_point[3];
            }
        }

        // Request distance calculations to all nodes
        #pragma omp for reduction(min : global_min_manhattan) reduction(max : global_max_manhattan) reduction(min : global_min_euclidean) reduction(max : global_max_euclidean) reduction(+:sum_min_manhattan) reduction(+:sum_max_manhattan) reduction(+:sum_min_euclidean) reduction(+:sum_max_euclidean)
        for(int i = 0; i < num_points_per_node; i++)
        {
            if(manhattan_euclidean_min_max[4*i] != DBL_MAX && manhattan_euclidean_min_max[4*i+1] != DBL_MAX && manhattan_euclidean_min_max[4*i+2] != DBL_MAX && manhattan_euclidean_min_max[4*i+3] != DBL_MAX)
            {
                sum_min_manhattan += manhattan_euclidean_min_max[4*i];
                sum_max_manhattan += manhattan_euclidean_min_max[4*i+1];
                sum_min_euclidean += manhattan_euclidean_min_max[4*i+2];
                sum_max_euclidean += manhattan_euclidean_min_max[4*i+3];
            }
            
            //printf("manhattan_euclidean_min_max[%d]: %lf\n", 4*i, manhattan_euclidean_min_max[4*i]);
            //printf("manhattan_euclidean_min_max[%d]: %lf\n", 4*i+1, manhattan_euclidean_min_max[4*i+1]);

            if(manhattan_euclidean_min_max[4*i] < global_min_manhattan)
                global_min_manhattan = manhattan_euclidean_min_max[4*i];
            if(manhattan_euclidean_min_max[4*i+1] > global_max_manhattan)
                global_max_manhattan = manhattan_euclidean_min_max[4*i+1];
            if(manhattan_euclidean_min_max[4*i+2] < global_min_euclidean)
                global_min_euclidean = manhattan_euclidean_min_max[4*i+2];
            if(manhattan_euclidean_min_max[4*i+3] > global_max_euclidean)
                global_max_euclidean = manhattan_euclidean_min_max[4*i+3];
        }

        // Respond to the request of the other nodes
        #pragma omp for
        for (int i = rank - 1; i >= 0; i--)
        {
            for(int j = 0; j < num_points_per_node; j++)
            {            
                // Recieve the broadcast message 
                MPI_Bcast(&point, 3, MPI_INT, i, MPI_COMM_WORLD);

                min_manhattan = DBL_MAX;
                max_manhattan = DBL_MIN;
                min_euclidean = DBL_MAX;
                max_euclidean = DBL_MIN;

                // Calculate the distances and send them to the node that requested it
                #pragma omp simd reduction(min : min_manhattan) reduction(max : max_manhattan) reduction(min : min_euclidean) reduction(max : max_euclidean)
                for(int k = 0; k < num_points_per_node; k++)
                {
                    double manhattan = abs(point[0] - x[k]) + abs(point[1] - y[k]) + abs(point[2] - z[k]);
                    //printf("manhattan: %d\n", manhattan);
                    double euclidean = sqrt(pow(point[0] - x[k], 2) + pow(point[1] - y[k], 2) + pow(point[2] - z[k], 2));
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
                
                MPI_Send(&manhattan_euclidean_min_max_point, 4, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
                //printf("Rank %d enviou a %d: %lf %lf %lf %lf referente ao ponto {%d, %d, %d}\n", rank, i, manhattan_euclidean_min_max_point[0], manhattan_euclidean_min_max_point[1], manhattan_euclidean_min_max_point[2], manhattan_euclidean_min_max_point[3], point[0], point[1], point[2]);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Print the global min and max euclidean and manhattan
    //printf("global_min_manhattan: %d\n", global_min_manhattan);
    //printf("global_max_manhattan: %d\n", global_max_manhattan);
    //printf("global_min_euclidean: %lf\n", global_min_euclidean);
    //printf("global_max_euclidean: %lf\n", global_max_euclidean);

    // Gather para o rank 0 dos minimos, máximos e soma de minimos e máximos encontrados por cada processo
    int *final_min_manhattan = (int *)malloc(sizeof(int) * num_nodes);
    int *final_max_manhattan = (int *)malloc(sizeof(int) * num_nodes);
    double *final_min_euclidean = (double *)malloc(sizeof(double) * num_nodes);
    double *final_max_euclidean = (double *)malloc(sizeof(double) * num_nodes);
    int *gather_sum_min_manhattan = (int *)malloc(sizeof(int) * num_nodes);
    int *gather_sum_max_manhattan = (int *)malloc(sizeof(int) * num_nodes);
    double *gather_sum_min_euclidean = (double *)malloc(sizeof(double) * num_nodes);
    double *gather_sum_max_euclidean = (double *)malloc(sizeof(double) * num_nodes);

    MPI_Gather(&global_min_manhattan, 1, MPI_INT, final_min_manhattan, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&global_max_manhattan, 1, MPI_INT, final_max_manhattan, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&global_min_euclidean, 1, MPI_DOUBLE, final_min_euclidean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&global_max_euclidean, 1, MPI_DOUBLE, final_max_euclidean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&sum_min_manhattan, 1, MPI_INT, gather_sum_min_manhattan, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&sum_max_manhattan, 1, MPI_INT, gather_sum_max_manhattan, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&sum_min_euclidean, 1, MPI_DOUBLE, gather_sum_min_euclidean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&sum_max_euclidean, 1, MPI_DOUBLE, gather_sum_max_euclidean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int global_min_manhattan = INT_MAX;
        int global_max_manhattan = INT_MIN;
        double global_min_euclidean = DBL_MAX;
        double global_max_euclidean = DBL_MIN;
        int total_sum_min_manhattan = 0;
        int total_sum_max_manhattan = 0;
        double total_sum_min_euclidean = 0.0;
        double total_sum_max_euclidean = 0.0;

        for (int i = 0; i < num_nodes; i++)
        {
            if (final_min_manhattan[i] < global_min_manhattan)
                global_min_manhattan = final_min_manhattan[i];
            if (final_max_manhattan[i] > global_max_manhattan)
                global_max_manhattan = final_max_manhattan[i];
            if (final_min_euclidean[i] < global_min_euclidean)
                global_min_euclidean = final_min_euclidean[i];
            if (final_max_euclidean[i] > global_max_euclidean)
                global_max_euclidean = final_max_euclidean[i];
            total_sum_min_manhattan += gather_sum_min_manhattan[i];
            total_sum_max_manhattan += gather_sum_max_manhattan[i];
            total_sum_min_euclidean += gather_sum_min_euclidean[i];
            total_sum_max_euclidean += gather_sum_max_euclidean[i];
        }

        printf("Distância de Manhattan mínima: %d (soma min: %d) e máxima: %d (soma max: %d).\n", global_min_manhattan, total_sum_min_manhattan, global_max_manhattan, total_sum_max_manhattan);
        printf("Distância Euclidiana mínima: %.2lf (soma min: %.2lf) e máxima: %.2lf (soma max: %.2lf).\n", global_min_euclidean, total_sum_min_euclidean, global_max_euclidean, total_sum_max_euclidean);
    }


/*#pragma omp for nowait
        for (int i = rank - 1; i >= 0; i--)
        {
            // Receive message from any node requesting mine x,y and z and send to it
            MPI_Recv(&src, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);

            printf("Rank %d received a request from rank %d\n", rank, src);

            MPI_Send(x, (num_points_per_node), MPI_INT, i, i + 1, MPI_COMM_WORLD);
            MPI_Send(y, (num_points_per_node), MPI_INT, i, i + 2, MPI_COMM_WORLD);
            MPI_Send(z, (num_points_per_node), MPI_INT, i, i + 3, MPI_COMM_WORLD);
        }

#pragma omp for nowait
        for (int i = rank; i < num_nodes; i++)
        {
            if (i == rank)
            {
                for (int j = 0; j < (num_points_per_node); j++)
                {
                    min_manhattan = INT_MAX;
                    max_manhattan = INT_MIN;
                    min_euclidean = DBL_MAX;
                    max_euclidean = DBL_MIN;

#pragma omp simd reduction(min : min_manhattan) reduction(max : max_manhattan) reduction(min : min_euclidean) reduction(max : max_euclidean)
                    for (int k = 0; k < (num_points_per_node); k++)
                    {
                        if (j == k)
                            continue;

                        int manhattan = abs(x[j] - x[k]) + abs(y[j] - y[k]) + abs(z[j] - z[k]);
                        double euclidean = sqrt(pow(x[j] - x[k], 2) + pow(y[j] - y[k], 2) + pow(z[j] - z[k], 2));
                        // printf("Manhattan: %d | Euclidean: %lf\n", manhattan, euclidean);

                        if (manhattan < min_manhattan)
                            min_manhattan = manhattan;
                        if (manhattan > max_manhattan)
                            max_manhattan = manhattan;
                        if (euclidean < min_euclidean)
                            min_euclidean = euclidean;
                        if (euclidean > max_euclidean)
                            max_euclidean = euclidean;
                    }
                
                    if (min_manhattan < min_manhattan_per_point_vector[j])
                        min_manhattan_per_point_vector[j] = min_manhattan;
                    if (max_manhattan > max_manhattan_per_point_vector[j])
                        max_manhattan_per_point_vector[j] = max_manhattan;
                    if (min_euclidean < min_euclidean_per_point_vector[j])
                        min_euclidean_per_point_vector[j] = min_euclidean;
                    if (max_euclidean > max_euclidean_per_point_vector[j])
                        max_euclidean_per_point_vector[j] = max_euclidean;
                }
            }
            else
            {
                printf("Rank %d will request data from rank %d\n", rank, i);
                MPI_Send(&rank, 1, MPI_INT, i, tag, MPI_COMM_WORLD);

                MPI_Recv(x_aux, (num_points_per_node), MPI_INT, i, rank + 1, MPI_COMM_WORLD, &status);
                printf("Rank %d recieved X from rank %d\n", rank, i);

                MPI_Recv(y_aux, (num_points_per_node), MPI_INT, i, rank + 2, MPI_COMM_WORLD, &status);
                printf("Rank %d recieved Y from rank %d\n", rank, i);

                MPI_Recv(z_aux, (num_points_per_node), MPI_INT, i, rank + 3, MPI_COMM_WORLD, &status);
                printf("Rank %d recieved Z from rank %d\n", rank, i);

                for (int j = 0; j < (num_points_per_node); j++)
                {
                    min_manhattan = INT_MAX;
                    max_manhattan = INT_MIN;
                    min_euclidean = DBL_MAX;
                    max_euclidean = DBL_MIN;

                    #pragma omp simd reduction(min : min_manhattan) reduction(max : max_manhattan) reduction(min : min_euclidean) reduction(max : max_euclidean)
                    for (int k = 0; k < (num_points_per_node); k++)
                    {
                        int manhattan = abs(x[j] - x_aux[k]) + abs(y[j] - y_aux[k]) + abs(z[j] - z_aux[k]);
                        double euclidean = sqrt(pow(x[j] - x_aux[k], 2) + pow(y[j] - y_aux[k], 2) + pow(z[j] - z_aux[k], 2));
                        printf("Manhattan: %d | Euclidean: %lf\n", manhattan, euclidean);

                        if (manhattan < min_manhattan)
                            min_manhattan = manhattan;
                        if (manhattan > max_manhattan)
                            max_manhattan = manhattan;
                        if (euclidean < min_euclidean)
                            min_euclidean = euclidean;
                        if (euclidean > max_euclidean)
                            max_euclidean = euclidean;
                    }

                    if (min_manhattan < min_manhattan_per_point_vector[j])
                        min_manhattan_per_point_vector[j] = min_manhattan;
                    if (max_manhattan > max_manhattan_per_point_vector[j])
                        max_manhattan_per_point_vector[j] = max_manhattan;
                    if (min_euclidean < min_euclidean_per_point_vector[j])
                        min_euclidean_per_point_vector[j] = min_euclidean;
                    if (max_euclidean > max_euclidean_per_point_vector[j])
                        max_euclidean_per_point_vector[j] = max_euclidean;
                }
            }
        }
    }*/

    // Get the sum of the mins and maxs of each point
    /*
    for (int i = 0; i < (num_points_per_node); i++)
    {
        sum_min_manhattan += min_manhattan_per_point_vector[i];
        sum_max_manhattan += max_manhattan_per_point_vector[i];
        sum_min_euclidean += min_euclidean_per_point_vector[i];
        sum_max_euclidean += max_euclidean_per_point_vector[i];

        if (min_manhattan_per_point_vector[i] < min_manhattan)
            min_manhattan = min_manhattan_per_point_vector[i];
        if (max_manhattan_per_point_vector[i] > max_manhattan)
            max_manhattan = max_manhattan_per_point_vector[i];
        if (min_euclidean_per_point_vector[i] < min_euclidean)
            min_euclidean = min_euclidean_per_point_vector[i];
        if (max_euclidean_per_point_vector[i] > max_euclidean)
            max_euclidean = max_euclidean_per_point_vector[i];
    }
    printf("in_euclidean_per_point_vector: ");
    for (int i = 0; i < num_nodes; i++)
    {
        printf("%.2lf ", min_euclidean_per_point_vector[i]);
    }
    printf("\n");
    */

    // Free of all point vectors -> Esta parte da memória alocada torna-se agora, inútil
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

    /*free(min_manhattan_per_point_vector);
    min_manhattan_per_point_vector = NULL;
    free(max_manhattan_per_point_vector);
    max_manhattan_per_point_vector = NULL;
    free(min_euclidean_per_point_vector);
    min_euclidean_per_point_vector = NULL;
    free(max_euclidean_per_point_vector);
    max_euclidean_per_point_vector = NULL;

    MPI_Barrier(MPI_COMM_WORLD);

    // Gather para o rank 0 dos minimos, máximos e soma de minimos e máximos encontrados por cada processo
    int *gather_min_manhattan_per_point_vector = (int *)malloc(sizeof(int) * num_nodes);
    int *gather_max_manhattan_per_point_vector = (int *)malloc(sizeof(int) * num_nodes);
    double *gather_min_euclidean_per_point_vector = (double *)malloc(sizeof(double) * num_nodes);
    double *gather_max_euclidean_per_point_vector = (double *)malloc(sizeof(double) * num_nodes);

    int *gather_sum_min_manhattan_per_point_vector = (int *)malloc(sizeof(int) * num_nodes);
    int *gather_sum_max_manhattan_per_point_vector = (int *)malloc(sizeof(int) * num_nodes);
    double *gather_sum_min_euclidean_per_point_vector = (double *)malloc(sizeof(double) * num_nodes);
    double *gather_sum_max_euclidean_per_point_vector = (double *)malloc(sizeof(double) * num_nodes);

    // Gather dos mins, max, sumMin, sumMax
    MPI_Gather(&min_manhattan, 1, MPI_INT, gather_min_manhattan_per_point_vector, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&max_manhattan, 1, MPI_INT, gather_max_manhattan_per_point_vector, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&min_euclidean, 1, MPI_DOUBLE, gather_min_euclidean_per_point_vector, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&max_euclidean, 1, MPI_DOUBLE, gather_max_euclidean_per_point_vector, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&sum_min_manhattan, 1, MPI_INT, gather_sum_min_manhattan_per_point_vector, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&sum_max_manhattan, 1, MPI_INT, gather_sum_max_manhattan_per_point_vector, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&sum_min_euclidean, 1, MPI_DOUBLE, gather_sum_min_euclidean_per_point_vector, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&sum_max_euclidean, 1, MPI_DOUBLE, gather_sum_max_euclidean_per_point_vector, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("gather_min_manhattan_per_point_vector: ");
        for (int i = 0; i < num_nodes; i++)
        {
            printf("%d ", gather_min_manhattan_per_point_vector[i]);
        }
        printf("\n");

        printf("\ngather_max_manhattan_per_point_vector: ");
        for (int i = 0; i < num_nodes; i++)
        {
            printf("%d ", gather_max_manhattan_per_point_vector[i]);
        }
        printf("\n");

        printf("\ngather_min_euclidean_per_point_vector: ");
        for (int i = 0; i < num_nodes; i++)
        {
            printf("%.2lf ", gather_min_euclidean_per_point_vector[i]);
        }
        printf("\n");

        printf("\ngather_max_euclidean_per_point_vector: ");
        for (int i = 0; i < num_nodes; i++)
        {
            printf("%.2lf ", gather_max_euclidean_per_point_vector[i]);
        }
        printf("\n");

        printf("\ngather_sum_min_manhattan_per_point_vector: ");
        for (int i = 0; i < num_nodes; i++)
        {
            printf("%d ", gather_sum_min_manhattan_per_point_vector[i]);
        }
        printf("\n");

        printf("\ngather_sum_max_manhattan_per_point_vector: ");
        for (int i = 0; i < num_nodes; i++)
        {
            printf("%d ", gather_sum_max_manhattan_per_point_vector[i]);
        }
        printf("\n");

        printf("\ngather_sum_min_euclidean_per_point_vector: ");
        for (int i = 0; i < num_nodes; i++)
        {
            printf("%.2lf ", gather_sum_min_euclidean_per_point_vector[i]);
        }     
        printf("\n");

        int global_min_manhattan = INT_MAX;
        int global_max_manhattan = INT_MIN;
        double global_min_euclidean = DBL_MAX;
        double global_max_euclidean = DBL_MIN;
        int total_sum_min_manhattan = 0;
        int total_sum_max_manhattan = 0;
        double total_sum_min_euclidean = 0.0;
        double total_sum_max_euclidean = 0.0;

        for (int i = 0; i < num_nodes; i++)
        {
            if (gather_min_manhattan_per_point_vector[i] < global_min_manhattan)
                global_min_manhattan = gather_min_manhattan_per_point_vector[i];
            if (gather_max_manhattan_per_point_vector[i] > global_max_manhattan)
                global_max_manhattan = gather_max_manhattan_per_point_vector[i];
            if (gather_min_euclidean_per_point_vector[i] < global_min_euclidean)
                global_min_euclidean = gather_min_euclidean_per_point_vector[i];
            if (gather_max_euclidean_per_point_vector[i] > global_max_euclidean)
                global_max_euclidean = gather_max_euclidean_per_point_vector[i];
            total_sum_min_manhattan += gather_sum_min_manhattan_per_point_vector[i];
            total_sum_max_manhattan += gather_sum_max_manhattan_per_point_vector[i];
            total_sum_min_euclidean += gather_sum_min_euclidean_per_point_vector[i];
            total_sum_max_euclidean += gather_sum_max_euclidean_per_point_vector[i];
        }

        printf("Distância de Manhattan mínima: %d (soma min: %d) e máxima: %d (soma max: %d).\n", global_min_manhattan, total_sum_min_manhattan, global_max_manhattan, total_sum_max_manhattan);
        printf("Distância Euclidiana mínima: %.2lf (soma min: %.2lf) e máxima: %.2lf (soma max: %.2lf).\n", global_min_euclidean, total_sum_min_euclidean, global_max_euclidean, total_sum_max_euclidean);
    }

    free(gather_min_manhattan_per_point_vector);
    gather_min_manhattan_per_point_vector = NULL;
    free(gather_max_manhattan_per_point_vector);
    gather_max_manhattan_per_point_vector = NULL;
    free(gather_min_euclidean_per_point_vector);
    gather_min_euclidean_per_point_vector = NULL;
    free(gather_max_euclidean_per_point_vector);
    gather_max_euclidean_per_point_vector = NULL;
    free(gather_sum_min_manhattan_per_point_vector);
    gather_sum_min_manhattan_per_point_vector = NULL;
    free(gather_sum_max_manhattan_per_point_vector);
    gather_sum_max_manhattan_per_point_vector = NULL;
    free(gather_sum_min_euclidean_per_point_vector);
    gather_sum_min_euclidean_per_point_vector = NULL;
    free(gather_sum_max_euclidean_per_point_vector);
    gather_sum_max_euclidean_per_point_vector = NULL;*/

    // Finalize MPI
    fflush(0);
    if (MPI_Finalize() != MPI_SUCCESS)
        return 1;

    return 0;
}