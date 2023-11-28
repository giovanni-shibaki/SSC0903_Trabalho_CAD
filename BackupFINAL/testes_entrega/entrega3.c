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

int get_num_points_per_node(int RANK, int NUM_POINTS, int NUM_NODES, int REMAINDER)
{
    int NUM_POINTS_PER_NODE = NUM_POINTS / NUM_NODES;

    // If the division between the number of points and the number of nodes is not exact,
    // the REMAINDER will be added to the first nodes, so they will have more points to calculate
    if(RANK < REMAINDER)
        NUM_POINTS_PER_NODE++;
    
    return NUM_POINTS_PER_NODE;
}

void generate_points_for_one_dimension(int *p_aux, int NUM_POINTS, int NUM_NODES, int REMAINDER)
{
    int index = 0;
    for(int i = 0; i < NUM_POINTS; i++)
    {
        p_aux[index] = rand() % 100;

        int rank_given_node = i % NUM_NODES;
        index += get_num_points_per_node(rank_given_node, NUM_POINTS, NUM_NODES, REMAINDER);

        if(index >= NUM_POINTS)
        {
            index++;
            index = index % NUM_POINTS;
        }
    }
}

void distribute_points_for_one_dimension(int *p_aux, int *p, int NUM_POINTS, int NUM_NODES, int REMAINDER)
{
    int index = 0;
    for(int n = 0; n < NUM_NODES; n++)
    {
        int num_points_given_node = get_num_points_per_node(n, NUM_POINTS, NUM_NODES, REMAINDER);

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
    int NUM_POINTS = n * n;

    // Set the seed to generate numbers
    srand(seed);

    // Set openMP number of threads
    omp_set_num_threads(num_threads);

    // Initialize MPI and get info about the nodes
    int NUM_NODES, RANK, tag = 0, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        printf("Aviso: O ambiente MPI não suporta multithreading\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_NODES);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);

    const int NUM_POINTS_PER_NODE = NUM_POINTS / NUM_NODES;
    const int REMAINDER = NUM_POINTS % NUM_NODES;
    const int NUM_POINTS_MY_NODE = get_num_points_per_node(RANK, NUM_POINTS, NUM_NODES, REMAINDER);

    // Global variables -> will be used on the reduce process
    int local_min_manhattan = INT_MAX, min_manhattan = INT_MAX, sum_min_manhattan = 0;
    int local_max_manhattan = INT_MIN, max_manhattan = INT_MIN, sum_max_manhattan = 0;
    double local_min_euclidean = DBL_MAX, min_euclidean = DBL_MAX, sum_min_euclidean = 0.0;
    double local_max_euclidean = DBL_MIN, max_euclidean = DBL_MIN, sum_max_euclidean = 0.0;

    // Temporary vector for Min and Max of each point on each process
    int *min_manhattan_per_point_vector = (int *)malloc(sizeof(int) * NUM_POINTS_MY_NODE);
    int *max_manhattan_per_point_vector = (int *)malloc(sizeof(int) * NUM_POINTS_MY_NODE);
    double *min_euclidean_per_point_vector = (double *)malloc(sizeof(double) * NUM_POINTS_MY_NODE);
    double *max_euclidean_per_point_vector = (double *)malloc(sizeof(double) * NUM_POINTS_MY_NODE);

    for(int i = 0; i < NUM_POINTS_MY_NODE; i++)
    {
        min_manhattan_per_point_vector[i] = INT_MAX;
        max_manhattan_per_point_vector[i] = INT_MIN;
        min_euclidean_per_point_vector[i] = DBL_MAX;
        max_euclidean_per_point_vector[i] = DBL_MIN;
    }

    // Node 0 generates x,y and z and distribute alternately to the other nodes
    int *x = (int *)malloc(sizeof(int) * NUM_POINTS_MY_NODE);
    int *y = (int *)malloc(sizeof(int) * NUM_POINTS_MY_NODE);
    int *z = (int *)malloc(sizeof(int) * NUM_POINTS_MY_NODE);
    int *p_aux =  (int *)malloc(sizeof(int) * NUM_POINTS);

    // Create aux vars and fill then with current values of x, y and z
    int *x_aux = (int *)malloc(sizeof(int) * (NUM_POINTS_PER_NODE+1));
    int *y_aux = (int *)malloc(sizeof(int) * (NUM_POINTS_PER_NODE+1));
    int *z_aux = (int *)malloc(sizeof(int) * (NUM_POINTS_PER_NODE+1));

    // Rank 0 generate and distribute the numbers between each process
    if (RANK == 0)
    {
        generate_points_for_one_dimension(p_aux, NUM_POINTS, NUM_NODES, REMAINDER);
        distribute_points_for_one_dimension(p_aux, x, NUM_POINTS, NUM_NODES, REMAINDER);

        generate_points_for_one_dimension(p_aux, NUM_POINTS, NUM_NODES, REMAINDER);
        distribute_points_for_one_dimension(p_aux, y, NUM_POINTS, NUM_NODES, REMAINDER);

        generate_points_for_one_dimension(p_aux, NUM_POINTS, NUM_NODES, REMAINDER);
        distribute_points_for_one_dimension(p_aux, z, NUM_POINTS, NUM_NODES, REMAINDER);
    }
    else
    {
        MPI_Recv(x, NUM_POINTS_MY_NODE, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        fflush(0);

        MPI_Recv(y, NUM_POINTS_MY_NODE, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        fflush(0);
        
        MPI_Recv(z, NUM_POINTS_MY_NODE, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        fflush(0);
    }

    // for(int i = 0; i < NUM_POINTS_MY_NODE; i++)
    // {
    //     printf("Rank %d: x[%d] = %d\n", RANK, i, x[i]);
    //     printf("Rank %d: y[%d] = %d\n", RANK, i, y[i]);
    //     printf("Rank %d: z[%d] = %d\n", RANK, i, z[i]);
    // }

    // One thread will be responsible for receiving messages and sending the current values of x, y and z to the node that requested it
    // The other threads will calculate the distances and update the global variables

    


    for (int i = 0; i < NUM_NODES; i++)
    {
        if(i == RANK)
        {
            MPI_Bcast(x, NUM_POINTS_MY_NODE, MPI_INT, RANK, MPI_COMM_WORLD);
            MPI_Bcast(y, NUM_POINTS_MY_NODE, MPI_INT, RANK, MPI_COMM_WORLD);
            MPI_Bcast(z, NUM_POINTS_MY_NODE, MPI_INT, RANK, MPI_COMM_WORLD);

            for(int j = 0; j < NUM_POINTS_MY_NODE; j++)
            {

                min_manhattan = INT_MAX;
                max_manhattan = INT_MIN;
                min_euclidean = DBL_MAX;
                max_euclidean = DBL_MIN;

                #pragma omp simd reduction(min:min_manhattan) reduction(max:max_manhattan) reduction(min:min_euclidean) reduction(max:max_euclidean)
                for(int k = j+1; k < NUM_POINTS_MY_NODE; k++)
                {
                    // Only works for n/p
                    int index_j = j*NUM_POINTS_PER_NODE + RANK;
                    int index_k = k*NUM_POINTS_PER_NODE + RANK;

                    //printf("I [%d] (%d, %d)\n", RANK, index_j, index_k);

                    int manhattan = abs(x[j] - x[k]) + abs(y[j] - y[k]) + abs(z[j] - z[k]);
                    double euclidean = sqrt(pow(x[j] - x[k], 2) + pow(y[j] - y[k], 2) + pow(z[j] - z[k], 2));
                    //printf("Manhattan: %d | Euclidean: %lf\n", manhattan, euclidean);

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

                // printf("Rank %d: min_manhattan_per_point[%d] = %d\n", RANK, j, min_manhattan_per_point_vector[j]);
                // printf("Rank %d: max_manhattan_per_point[%d] = %d\n", RANK, j, max_manhattan_per_point_vector[j]);
                // printf("Rank %d: min_euclidean_per_point[%d] = %lf\n", RANK, j, min_euclidean_per_point_vector[j]);
                // printf("Rank %d: max_euclidean_per_point[%d] = %lf\n", RANK, j, max_euclidean_per_point_vector[j]);
            }
        }
        else
        {
            // printf("Rank %d will request data from RANK %d\n", RANK, i);

            // Calculate the number of points that the node will receive from the node with RANK i
            int num_points_sender = NUM_POINTS_MY_NODE;
            if(i < REMAINDER && RANK >= REMAINDER)
                num_points_sender++;
            else if(i >= REMAINDER && RANK < REMAINDER)
                num_points_sender--;

            MPI_Bcast(x_aux, num_points_sender, MPI_INT, i, MPI_COMM_WORLD);
            MPI_Bcast(y_aux, num_points_sender, MPI_INT, i, MPI_COMM_WORLD);
            MPI_Bcast(z_aux, num_points_sender, MPI_INT, i, MPI_COMM_WORLD);

            // MPI_Recv(x_aux, num_points_sender, MPI_INT, i, RANK + 1, MPI_COMM_WORLD, &status);
            // // printf("Rank %d recieved X from RANK %d\n", RANK, i);

            // MPI_Recv(y_aux, num_points_sender, MPI_INT, i, RANK + 2, MPI_COMM_WORLD, &status);
            // // printf("Rank %d recieved Y from RANK %d\n", RANK, i);

            // MPI_Recv(z_aux, num_points_sender, MPI_INT, i, RANK + 3, MPI_COMM_WORLD, &status);
            // printf("Rank %d recieved Z from RANK %d\n", RANK, i);

            for(int j = 0; j < NUM_POINTS_MY_NODE; j++)
            {
                min_manhattan = INT_MAX;
                max_manhattan = INT_MIN;
                min_euclidean = DBL_MAX;
                max_euclidean = DBL_MIN;

                int start_k = (RANK < i) ? j : j+1;

                #pragma omp simd reduction(min:min_manhattan) reduction(max:max_manhattan) reduction(min:min_euclidean) reduction(max:max_euclidean)
                for(int k = start_k; k < num_points_sender; k++)
                {
                    // Only works for n/p
                    int index_j = j*NUM_POINTS_PER_NODE + RANK;
                    int index_k = k*NUM_POINTS_PER_NODE + i;

                    //printf("[%d] (%d, %d)\n", RANK, index_j, index_k);


                    int manhattan = abs(x[j] - x_aux[k]) + abs(y[j] - y_aux[k]) + abs(z[j] - z_aux[k]);
                    double euclidean = sqrt(pow(x[j] - x_aux[k], 2) + pow(y[j] - y_aux[k], 2) + pow(z[j] - z_aux[k], 2));
                    //printf("Manhattan: %d | Euclidean: %lf\n", manhattan, euclidean);

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

    #pragma omp parallel for \
    reduction(+:sum_min_manhattan, sum_max_manhattan, sum_min_euclidean, sum_max_euclidean) \
    reduction(min:local_min_manhattan, local_min_euclidean) \
    reduction(max:local_max_manhattan, local_max_euclidean)
    for(int i=0; i < NUM_POINTS_MY_NODE; i++)
    {
        // printf("(%d) %d\n", RANK, i);
        if(!(min_manhattan_per_point_vector[i]  != INT_MAX && max_manhattan_per_point_vector[i] != INT_MIN && min_euclidean_per_point_vector[i] != DBL_MAX && max_euclidean_per_point_vector[i]  != DBL_MIN))
            continue;

        sum_min_manhattan += min_manhattan_per_point_vector[i];
        sum_max_manhattan += max_manhattan_per_point_vector[i];
        sum_min_euclidean += min_euclidean_per_point_vector[i];
        sum_max_euclidean += max_euclidean_per_point_vector[i];

        // printf("(%d) %d\n", RANK, i);

        if(min_manhattan_per_point_vector[i] < local_min_manhattan)
            local_min_manhattan = min_manhattan_per_point_vector[i];
        if(max_manhattan_per_point_vector[i] > local_max_manhattan)
            local_max_manhattan = max_manhattan_per_point_vector[i];
        if(min_euclidean_per_point_vector[i] < local_min_euclidean)
            local_min_euclidean = min_euclidean_per_point_vector[i];
        if(max_euclidean_per_point_vector[i] > local_max_euclidean)
            local_max_euclidean = max_euclidean_per_point_vector[i];
    }

    for(int i = 0; i < (NUM_POINTS_MY_NODE); i++)
    {
        // printf("Rank %d: Distância de Manhattan mínima: %d e máxima: %d.\n", RANK, min_manhattan, max_manhattan);
        //     printf("Rank %d: Distância Euclidiana mínima: %.2lf e máxima: %.2lf.\n", RANK, min_euclidean, max_euclidean);

        // Write prints in the above format
        // printf("Rank %d: [M] (%d) | (%d)\n", RANK, min_manhattan_per_point_vector[i], max_manhattan_per_point_vector[i]);
        // printf("Rank %d: [E]: (%.2lf) | (%.2lf)\n", RANK, min_euclidean_per_point_vector[i], max_euclidean_per_point_vector[i]);
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
    if (RANK == 0)
    {
        printf("Distância de Manhattan mínima: %d (soma min: %d) e máxima: %d (soma max: %d).\n", global_min_manhattan, total_sum_min_manhattan, global_max_manhattan, total_sum_max_manhattan);
        printf("Distância Euclidiana mínima: %.2lf (soma min: %.2lf) e máxima: %.2lf (soma max: %.2lf).\n", global_min_euclidean, total_sum_min_euclidean, global_max_euclidean, total_sum_max_euclidean);
    }
    
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