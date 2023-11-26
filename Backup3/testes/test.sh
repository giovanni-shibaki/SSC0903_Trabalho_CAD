start = $(date +%s)
mpirun -np 4 test_omp_simd2 4 12 4
end = $(date +%s)

echo "Elapsed Time: $(($end-$start)) seconds"
