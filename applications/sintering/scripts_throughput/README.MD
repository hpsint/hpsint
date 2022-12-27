```
likwid-mpirun -np 40 -f -g CACHES   -m -O ./applications/sintering/sintering-throughput | tee throughput_caches.out
likwid-mpirun -np 40 -f -g FLOPS_DP -m -O ./applications/sintering/sintering-throughput | tee throughput_flops.out
```