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
#include <math.h>
#include <limits.h>
#include <float.h>
#include <omp.h>

int main()
{
    #pragma omp parallel num_threads(4)
    {
        #pragma omp task
        {
            #pragma omp single
            {
                printf("Entrei\n");

                while(1)
                {
                    continue;
                    printf("Sla\n");
                }

                printf("Sai\n");
            }
        }

        printf("Continuando..");
    }

}