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
 * > make run_seq N=1000000 SEED=1 -> Roda código sequencial com N = 1000000 e SEED = 1
 * > make run_par N=1000000 SEED=1 NTHREADS=4 -> Roda código paralelo com N = 1000000, SEED = 1 e NTHREADS = 4 (NP = 4)
 * > make clean -> Limpa os executáveis
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"

#define MASTER_RANK 0

#define abs2(x) ((x) < 0 ? -(x) : (x))
#define pow2(x) ((x) * (x))

int get_num_points_per_node(int RANK, int NUM_POINTS, int NUM_NODES, int REMAINDER)
{
    int num_points_per_node = NUM_POINTS / NUM_NODES;

    // If the division between the number of points and the number of nodes is not exact,
    // the remaining points will be allocated to the first nodes, so they will have more points to calculate
    if(RANK < REMAINDER)
        num_points_per_node++;
    
    return num_points_per_node;
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
    /* --- I. Initialize OMP, MPI and variables --- */

    // Read command line arguments and set params
    const int n = atoi(argv[1]);
    const int seed = atoi(argv[2]);
    const int num_threads = atoi(argv[3]);

    srand(seed);
    omp_set_num_threads(num_threads);

    // Initialize MPI 
    int tag = 0, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        printf("Aviso: O ambiente MPI não suporta multithreading\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Status status;

    // Get info about the nodes
    int NUM_NODES, RANK;
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_NODES);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    
    const int NUM_POINTS = n * n;
    const int REMAINDER = NUM_POINTS % NUM_NODES;
    const int NUM_POINTS_PER_NODE = NUM_POINTS / NUM_NODES;
    const int NUM_POINTS_MY_NODE = get_num_points_per_node(RANK, NUM_POINTS, NUM_NODES, REMAINDER);

    // Global variables -> will be used on the reduce process
    int local_min_manhattan = INT_MAX, sum_min_manhattan = 0;
    int local_max_manhattan = INT_MIN, sum_max_manhattan = 0;
    double local_min_euclidean = DBL_MAX, sum_min_euclidean = 0.0;
    double local_max_euclidean = DBL_MIN, sum_max_euclidean = 0.0;

    /* --- II. Node 0 generates x,y and z and distribute alternately to the other nodes --- */

    int *p_aux =  (int *)malloc(sizeof(int) * NUM_POINTS);
    int *x = (int *)malloc(sizeof(int) * NUM_POINTS_MY_NODE);
    int *y = (int *)malloc(sizeof(int) * NUM_POINTS_MY_NODE);
    int *z = (int *)malloc(sizeof(int) * NUM_POINTS_MY_NODE);

    if (RANK == MASTER_RANK)
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
        MPI_Recv(x, NUM_POINTS_MY_NODE, MPI_INT, 0, tag, MPI_COMM_WORLD, &status); fflush(0); 
        MPI_Recv(y, NUM_POINTS_MY_NODE, MPI_INT, 0, tag, MPI_COMM_WORLD, &status); fflush(0);
        MPI_Recv(z, NUM_POINTS_MY_NODE, MPI_INT, 0, tag, MPI_COMM_WORLD, &status); fflush(0);
    }

    
    /* --- III. Send local points to the other nodes for them to calculate the distances --- */

    int point[3] = {};

    #pragma omp parallel private(point)
    {
        #pragma omp for
        for(int i=0; i < NUM_POINTS_MY_NODE; i++)
        {
            point[0] = x[i];
            point[1] = y[i];
            point[2] = z[i];

            for(int j = 0; j < NUM_NODES; j++)
            {
                if(j == RANK)
                    continue;

                MPI_Send(point, 3, MPI_INT, j, i, MPI_COMM_WORLD);
            }
        }
    }

    /* --- IV. Calculate distances and reduce locally --- */

    int min_manhattan = INT_MAX;
    int max_manhattan = INT_MIN;
    double min_euclidean = DBL_MAX;
    double max_euclidean = DBL_MIN;

    // This array will be used to return the results of the distances to the node that requested it
    double manhattan_euclidean_min_max_point[4] = {};

    #pragma omp parallel \
        private(point, manhattan_euclidean_min_max_point, min_manhattan, max_manhattan, min_euclidean, max_euclidean) default(shared)
    {
        /* IV.a For every node, receive its points and calculate the distances */
        #pragma omp for
        for (int sender_node = 0; sender_node < NUM_NODES; sender_node++)
        {
            if(sender_node == RANK)
                continue;

            int num_points_sender = NUM_POINTS_PER_NODE + (sender_node < REMAINDER ? 1 : 0);

            // For every point of the sender, calculate the distances to the points of the current node
            for(int sender_index = 0; sender_index < num_points_sender; sender_index++)
            {           
                MPI_Recv(point, 3, MPI_INT, sender_node, sender_index, MPI_COMM_WORLD, &status);

                min_manhattan = INT_MAX; max_manhattan = INT_MIN;
                min_euclidean = DBL_MAX; max_euclidean = DBL_MIN;

                int starting_local_index = (RANK < sender_node) ? sender_index+1 : sender_index;

                #pragma omp simd reduction(min : min_manhattan, min_euclidean) reduction(max : max_manhattan, max_euclidean)
                for(int k = starting_local_index; k < NUM_POINTS_MY_NODE; k++)
                {
                    double manhattan = abs2(point[0] - x[k]) + abs2(point[1] - y[k]) + abs2(point[2] - z[k]);
                    double euclidean = sqrt(pow2(point[0] - x[k]) + pow2(point[1] - y[k]) + pow2(point[2] - z[k]));

                    if(manhattan < min_manhattan) min_manhattan = manhattan;
                    if(manhattan > max_manhattan) max_manhattan = manhattan;
                    if(euclidean < min_euclidean) min_euclidean = euclidean;
                    if(euclidean > max_euclidean) max_euclidean = euclidean;
                }

                // Send results to sending node
                manhattan_euclidean_min_max_point[0] = min_manhattan;
                manhattan_euclidean_min_max_point[1] = max_manhattan;
                manhattan_euclidean_min_max_point[2] = min_euclidean;
                manhattan_euclidean_min_max_point[3] = max_euclidean;
                
                MPI_Send(&manhattan_euclidean_min_max_point, 4, MPI_DOUBLE, sender_node, sender_index+sender_node, MPI_COMM_WORLD);
            }
        }

        /* IV.b Calculate the local distances and get results from other nodes to update vars */
        #pragma omp for reduction(min : local_min_manhattan, local_min_euclidean) \
            reduction(max : local_max_manhattan, local_max_euclidean) reduction(+:sum_min_manhattan, sum_max_manhattan, sum_min_euclidean, sum_max_euclidean)
        for(int i = 0; i < NUM_POINTS_MY_NODE; i++)
        {
            min_manhattan = INT_MAX; max_manhattan = INT_MIN;
            min_euclidean = DBL_MAX; max_euclidean = DBL_MIN;

            // Calculate the distance to the points that the node already has
            #pragma omp simd reduction(min : min_manhattan, min_euclidean) reduction(max : max_manhattan, max_euclidean)
            for(int j = i+1; j < NUM_POINTS_MY_NODE; j++)
            {
                int manhattan = abs2(x[i] - x[j]) + abs2(y[i] - y[j]) + abs2(z[i] - z[j]);
                double euclidean = sqrt(pow2(x[i] - x[j]) + pow2(y[i] - y[j]) + pow2(z[i] - z[j]));

                if(manhattan < min_manhattan) min_manhattan = manhattan;
                if(manhattan > max_manhattan) max_manhattan = manhattan;
                if(euclidean < min_euclidean) min_euclidean = euclidean;
                if(euclidean > max_euclidean) max_euclidean = euclidean;
            }

            // Get the results of the distance calculation from the other nodes
            for(int k = 0; k < NUM_NODES; k++)
            {
                if(k == RANK)
                    continue;

                double manhattan_euclidean_min_max_point[4] = {};
                MPI_Recv(&manhattan_euclidean_min_max_point, 4, MPI_DOUBLE, k, i+RANK, MPI_COMM_WORLD, &status);

                if(manhattan_euclidean_min_max_point[0] < min_manhattan) min_manhattan = manhattan_euclidean_min_max_point[0];
                if(manhattan_euclidean_min_max_point[1] > max_manhattan) max_manhattan = manhattan_euclidean_min_max_point[1];
                if(manhattan_euclidean_min_max_point[2] < min_euclidean) min_euclidean = manhattan_euclidean_min_max_point[2];
                if(manhattan_euclidean_min_max_point[3] > max_euclidean) max_euclidean = manhattan_euclidean_min_max_point[3];
            }

            // Here, we have the minimum and maximum values of the distances of the current point i
            if(min_manhattan < local_min_manhattan) local_min_manhattan = min_manhattan;
            if(max_manhattan > local_max_manhattan) local_max_manhattan = max_manhattan;
            if(min_euclidean < local_min_euclidean) local_min_euclidean = min_euclidean;
            if(max_euclidean > local_max_euclidean) local_max_euclidean = max_euclidean;

            if(min_manhattan != INT_MAX && max_manhattan != INT_MIN && min_euclidean != INT_MAX && max_euclidean != INT_MIN)
            {
                sum_min_manhattan += min_manhattan;
                sum_max_manhattan += max_manhattan;
                sum_min_euclidean += min_euclidean;
                sum_max_euclidean += max_euclidean;
            }
        }
    }

    /* --- V. Reduce globally and print results --- */

    int global_min_manhattan = 0;
    int global_max_manhattan = 0;
    double global_min_euclidean = 0.0;
    double global_max_euclidean = 0.0;
    int total_sum_min_manhattan = 0;
    int total_sum_max_manhattan = 0;
    double total_sum_min_euclidean = 0.0;
    double total_sum_max_euclidean = 0.0;

    // Reduction of global sum's
    MPI_Reduce(&sum_min_manhattan, &total_sum_min_manhattan, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_max_manhattan, &total_sum_max_manhattan, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_min_euclidean, &total_sum_min_euclidean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_max_euclidean, &total_sum_max_euclidean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Reduction of global min's and max's
    MPI_Reduce(&local_min_manhattan, &global_min_manhattan, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_manhattan, &global_max_manhattan, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min_euclidean, &global_min_euclidean, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_euclidean, &global_max_euclidean, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (RANK == MASTER_RANK)
    {
        printf("Distância de Manhattan mínima: %d (soma min: %d) e máxima: %d (soma max: %d).\n", global_min_manhattan, total_sum_min_manhattan, global_max_manhattan, total_sum_max_manhattan);
        printf("Distância Euclidiana mínima: %.2lf (soma min: %.2lf) e máxima: %.2lf (soma max: %.2lf).\n", global_min_euclidean, total_sum_min_euclidean, global_max_euclidean, total_sum_max_euclidean);
    }

    // Free's and finalize MPI
    free(p_aux); p_aux = NULL;
    free(x); x = NULL;
    free(y); y = NULL;
    free(z); z = NULL;

    fflush(0);
    if (MPI_Finalize() != MPI_SUCCESS)
        return 1;

    return 0;
}
