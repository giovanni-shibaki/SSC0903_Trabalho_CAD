#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    int i;

    // Execute o programa compilado com diferentes parâmetros
    for (i = 1; i <= 1; i++) {
        char command[100];
        char buffer[128];
        char realTime[20];  // Variável para armazenar o tempo real

        int np = 32;
        int N = 50;
        int seed = 1;
        int Nthreads = 12;

        // Substitua "seu_programa" pelo nome do seu programa compilado
        // e adicione os parâmetros desejados
        sprintf(command, "time mpirun -np %d --hostfile host.txt ./entrega %d %d %d",
                np, N, seed, Nthreads);

        // Abre o processo para leitura da saída do comando
        FILE *fp = popen(command, "r");
        if (fp == NULL) {
            perror("Erro ao executar o comando");
            exit(EXIT_FAILURE);
        }

        // Lê a saída do comando
        while (fgets(buffer, sizeof(buffer), fp) != NULL) {
            // Verifica se a linha contém "real" (tempo real)
            if (strstr(buffer, "real") != NULL) {
                // Extrai o valor do tempo real
                sscanf(buffer, "real\t%[^\n]", realTime);
                printf("\n\nIteração %d: Tempo Real: %s\n", i, realTime);
                break; // Encerra a leitura após encontrar o tempo real
            }
        }

        // Fecha o processo
        pclose(fp);
    }

    return 0;
}
