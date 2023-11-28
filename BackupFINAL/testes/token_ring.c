/*

Faça um programa concorrente em C/MPI que implemente um Token Ring de
processos, estes dispostos logicamente na forma de uma fila circular. O primeiro
processo (de rank zero), deverá gerar um token = 0 (zero, um valor inteiro), passar
este para o próximo processo (rank 1) e aguardar o recebimento do token do último
processo (rank n-1). O processo zero, ao receber de volta o token, vindo deste último
processo, imprime na tela o valor recebido e finaliza. Todos os processos devem
receber um valor de outro processo, incrementá-lo e então repassar esse novo token
ao próximo processo na fila circular. Após uma volta do token pelos processos, a
aplicação finaliza.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    int npes, myrank, src, dest, msgtag, ret;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    msgtag = 0;
    int token = 0;

    printf("Number of processes: %d  ", npes);
    printf("Running rank %d\n", myrank);

    if (myrank == 0)
    {
        // Send token to rank 1
        src = npes - 1;
        dest = myrank + 1;

        MPI_Send(&token, 1, MPI_INT, dest, msgtag, MPI_COMM_WORLD);

        msgtag = 0;

        MPI_Recv(&token, 1, MPI_INT, src, msgtag, MPI_COMM_WORLD, &status);
        printf("Node with rank %d recieved token with value: %d\n", myrank, token);
    }
    else
    {
        // Recieve token, check if the value is equal to the numer of nodes
        // If true, send the token to the first node
        // If false, send the token to the next node (myrank + 1)

        src = myrank - 1;

        MPI_Recv(&token, 1, MPI_INT, src, msgtag, MPI_COMM_WORLD, &status);
        msgtag = 0;

	printf("Rank %d received token: %d\n", myrank, token);
        if (token == npes-2)
        {
            // Send token back to the one with rank 0
            dest = 0;
	    token += 1;
	    printf("Rank %d will send token %d to rank %d\n", myrank, token, dest);
            MPI_Send(&token, 1, MPI_INT, dest, msgtag, MPI_COMM_WORLD);
        }
        else
        {
            // Send token to the next node
            dest = myrank + 1;
	    token += 1;
	    printf("Rank %d will send token %d to rank %d\n", myrank, token, dest);
            MPI_Send(&token, 1, MPI_INT, dest, msgtag, MPI_COMM_WORLD);
        }
    }

    fflush(0);

    ret = MPI_Finalize();
    if (ret == MPI_SUCCESS)
    {
        printf("MPI_Finalize success!\n");
    }
    return 0;
}