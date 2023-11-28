MPICC = mpicc
CC = gcc
CFLAGS = -lm -Wall -Wextra -fopenmp
MPIFLAGS = -np 4 --use-hwthread-cpus --oversubscribe --hostfile host.txt
CFILE_SEQ = dist-seq.c
CFILE_PAR = entrega.c
SEQ_TARGET = dist_seq
PAR_TARGET = entrega

all_seq: $(OFILES)
	@$(CC) $(CFILE_SEQ) -o $(SEQ_TARGET)  $(CFLAGS) -g

all_par: $(OFILES)
	@$(MPICC) $(CFILE_PAR) -o $(PAR_TARGET)  $(CFLAGS) -g

run_seq:
	@./$(SEQ_TARGET) ${N} ${SEED}

run_par:
	@mpirun $(MPIFLAGS) $(PAR_TARGET) ${N} ${SEED} ${NTHREADS}

clean:
	@rm $(SEQ_TARGET) $(PAR_TARGET)