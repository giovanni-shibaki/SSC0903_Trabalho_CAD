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
 * > make all_seq -> Compila código sequencial dist-seq.c
 * > make all_par -> Compila código paralelo entrega.c
 * > make run_seq N=100 SEED=1 -> Roda código sequencial com N = 100 e SEED = 1
 * > make run_par NP=10 N=100 SEED=1 NTHREADS=4 -> Roda código paralelo com -np = 10, N = 100, SEED = 1 e NTHREADS = 4
 * > make clean -> Limpa os executáveis
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

int get_num_points_per_node(int rank, int num_points, int num_nodes, int remainder)
{
    int num_points_per_node = num_points / num_nodes;

    // If the division between the number of points and the number of nodes is not exact,
    // the remainder will be added to the first nodes, so they will have more points to calculate
    if(rank < remainder)
        num_points_per_node++;
    
    return num_points_per_node;
}

void generate_points_for_one_dimension(int *p_aux, int num_points, int num_nodes, int remainder)
{
    int index = 0;
    for(int i = 0; i < num_points; i++)
    {
        p_aux[index] = rand() % 100;

        int rank_given_node = i % num_nodes;
        index += get_num_points_per_node(rank_given_node, num_points, num_nodes, remainder);

        if(index >= num_points)
        {
            index++;
            index = index % num_points;
        }
    }
}

void distribute_points_for_one_dimension(int *p_aux, int *p, int num_points, int num_nodes, int remainder)
{
    int index = 0;
    for(int n = 0; n < num_nodes; n++)
    {
        int num_points_given_node = get_num_points_per_node(n, num_points, num_nodes, remainder);

        if(n == 0)
        {
            for(int i = 0; i < num_points_given_node; i++)
            {
                p[i] = p_aux[i];
            }
        }
        else
        {
            MPI_Send(&p_aux[index], num_points_given_node, MPI_INT, n, 0, MPI_COMM_WORLD);
        }

        index += num_points_given_node;
    }
}

int main(int argc, char *argv[])
{
    // Read command line arguments
    int n = atoi(argv[1]);
    int seed = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    int num_points = n * n;

    // Set the seed to generate numbers
    srand(seed);

    // Initialize MPI and get info about the nodes
    int num_nodes, rank, tag = 0, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        printf("Aviso: O ambiente MPI não suporta multithreading\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int remainder = num_points % num_nodes;
    int num_points_my_node = get_num_points_per_node(rank, num_points, num_nodes, remainder);

    // Global variables -> will be used on the reduce process
    int local_min_manhattan = INT_MAX, sum_min_manhattan = 0;
    int local_max_manhattan = INT_MIN, sum_max_manhattan = 0;
    double local_min_euclidean = DBL_MAX, sum_min_euclidean = 0.0;
    double local_max_euclidean = DBL_MIN, sum_max_euclidean = 0.0;

    // Node 0 generates x,y and z and distribute alternately to the other nodes
    int *p_aux =  (int *)malloc(sizeof(int) * num_points);
    int *x = (int *)malloc(sizeof(int) * num_points_my_node);
    int *y = (int *)malloc(sizeof(int) * num_points_my_node);
    int *z = (int *)malloc(sizeof(int) * num_points_my_node);

    // Rank 0 generate and distribute the numbers between each process
    if (rank == 0)
    {
        generate_points_for_one_dimension(p_aux, num_points, num_nodes, remainder);
        distribute_points_for_one_dimension(p_aux, x, num_points, num_nodes, remainder);

        generate_points_for_one_dimension(p_aux, num_points, num_nodes, remainder);
        distribute_points_for_one_dimension(p_aux, y, num_points, num_nodes, remainder);

        generate_points_for_one_dimension(p_aux, num_points, num_nodes, remainder);
        distribute_points_for_one_dimension(p_aux, z, num_points, num_nodes, remainder);
    }
    else
    {
        MPI_Recv(x, num_points_my_node, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        fflush(0);

        MPI_Recv(y, num_points_my_node, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        fflush(0);
        
        MPI_Recv(z, num_points_my_node, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        fflush(0);
    }

    int min_manhattan = INT_MAX;
    int max_manhattan = INT_MIN;
    double min_euclidean = DBL_MAX;
    double max_euclidean = DBL_MIN;

    // Send the points to the other nodes for them to calculate the distances
    int point[3] = {};

    #pragma omp parallel num_threads(num_threads) private(point)
    {
        #pragma omp for
        for(int i=0; i < num_points_my_node; i++)
        {
            point[0] = x[i];
            point[1] = y[i];
            point[2] = z[i];

            for(int j = 0; j < num_nodes; j++)
            {
                if(j == rank)
                    continue;

                MPI_Send(point, 3, MPI_INT, j, i, MPI_COMM_WORLD);
            }
        }
    }

    // This array will be used to return the results of the distances to the node that requested it
    double manhattan_euclidean_min_max_point[4] = {};

    #pragma omp parallel num_threads(num_threads) private(point, manhattan_euclidean_min_max_point, min_manhattan, max_manhattan, min_euclidean, max_euclidean) default(shared)
    {
        // Respond to the request of the other nodes
        #pragma omp for
        for (int i = 0; i < num_nodes; i++)
        {
            if(i == rank)
                continue;

            // j -> index of the point of the sender
            // k -> index of the point of the receiver

            // Calculate the number of points that the node will receive from the node with rank i
            int num_points_sender = num_points_my_node;
            if(i < remainder && rank >= remainder)
                num_points_sender++;
            else if(i >= remainder && rank < remainder)
                num_points_sender--;

            for(int j = 0; j < num_points_sender; j++)
            {           
                MPI_Recv(point, 3, MPI_INT, i, j, MPI_COMM_WORLD, &status);

                min_manhattan = INT_MAX;
                max_manhattan = INT_MIN;
                min_euclidean = DBL_MAX;
                max_euclidean = DBL_MIN;

                int start_k = (rank < i) ? j+1 : j;

                // Calculate the distances and send them to the node that requested it
                #pragma omp simd reduction(min : min_manhattan) reduction(max : max_manhattan) reduction(min : min_euclidean) reduction(max : max_euclidean)
                for(int k = start_k; k < num_points_my_node; k++)
                {
                    double manhattan = abs(point[0] - x[k]) + abs(point[1] - y[k]) + abs(point[2] - z[k]);
                    double euclidean = sqrt(pow(point[0] - x[k], 2) + pow(point[1] - y[k], 2) + pow(point[2] - z[k], 2));

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

                // Store the results in the array
                manhattan_euclidean_min_max_point[0] = min_manhattan;
                manhattan_euclidean_min_max_point[1] = max_manhattan;
                manhattan_euclidean_min_max_point[2] = min_euclidean;
                manhattan_euclidean_min_max_point[3] = max_euclidean;
                
                // Send the results to the node that requested it with MPI_Send
                MPI_Send(&manhattan_euclidean_min_max_point, 4, MPI_DOUBLE, i, j+i, MPI_COMM_WORLD);
            }
        }

        // Calculate the distances and update the global variables. Later, a reduce will be done to get the final result in the node 0
        #pragma omp for reduction(min : local_min_manhattan, local_min_euclidean) reduction(max : local_max_manhattan, local_max_euclidean) \
        reduction(+:sum_min_manhattan, sum_max_manhattan, sum_min_euclidean, sum_max_euclidean)
        for(int i = 0; i < num_points_my_node; i++)
        {
            min_manhattan = INT_MAX;
            max_manhattan = INT_MIN;
            min_euclidean = DBL_MAX;
            max_euclidean = DBL_MIN;

            // Calculate the distance to the points that the node already has
            #pragma omp simd reduction(min : min_manhattan) reduction(max : max_manhattan) reduction(min : min_euclidean) reduction(max : max_euclidean)
            for(int j = i+1; j < num_points_my_node; j++)
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

            // Get the results of the distance calculation from the other nodes
            for(int k = 0; k < num_nodes; k++)
            {
                if(k == rank)
                    continue;

                double manhattan_euclidean_min_max_point[4] = {};
                MPI_Recv(&manhattan_euclidean_min_max_point, 4, MPI_DOUBLE, k, i+rank, MPI_COMM_WORLD, &status);

                if(manhattan_euclidean_min_max_point[0] < min_manhattan)
                    min_manhattan = manhattan_euclidean_min_max_point[0];
                if(manhattan_euclidean_min_max_point[1] > max_manhattan)
                    max_manhattan = manhattan_euclidean_min_max_point[1];
                if(manhattan_euclidean_min_max_point[2] < min_euclidean)
                    min_euclidean = manhattan_euclidean_min_max_point[2];
                if(manhattan_euclidean_min_max_point[3] > max_euclidean)
                    max_euclidean = manhattan_euclidean_min_max_point[3];
            }

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
    }

    // These variables will be used for the reduce process
    int global_min_manhattan = 0;
    int global_max_manhattan = 0;
    double global_min_euclidean = 0.0;
    double global_max_euclidean = 0.0;
    int total_sum_min_manhattan = 0;
    int total_sum_max_manhattan = 0;
    double total_sum_min_euclidean = 0.0;
    double total_sum_max_euclidean = 0.0;

    // Making the reduction to the sum of the minimum and maximum distances
    MPI_Reduce(&sum_min_manhattan, &total_sum_min_manhattan, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_max_manhattan, &total_sum_max_manhattan, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_min_euclidean, &total_sum_min_euclidean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_max_euclidean, &total_sum_max_euclidean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Making the reduction to the global min and max values of the distances
    MPI_Reduce(&local_min_manhattan, &global_min_manhattan, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_manhattan, &global_max_manhattan, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min_euclidean, &global_min_euclidean, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_euclidean, &global_max_euclidean, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Rank 0 will display the results
    if (rank == 0)
    {
        printf("Distância de Manhattan mínima: %d (soma min: %d) e máxima: %d (soma max: %d).\n", global_min_manhattan, total_sum_min_manhattan, global_max_manhattan, total_sum_max_manhattan);
        printf("Distância Euclidiana mínima: %.2lf (soma min: %.2lf) e máxima: %.2lf (soma max: %.2lf).\n", global_min_euclidean, total_sum_min_euclidean, global_max_euclidean, total_sum_max_euclidean);
    }

    // Free of all point vectors
    free(p_aux);
    p_aux = NULL;
    free(x);
    x = NULL;
    free(y);
    y = NULL;
    free(z);
    z = NULL;

    // Finalize MPI
    fflush(0);
    if (MPI_Finalize() != MPI_SUCCESS)
        return 1;

    return 0;
}