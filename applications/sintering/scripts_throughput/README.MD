```
likwid-mpirun -np 40 -f -g CACHES   -m -O ./applications/sintering/sintering-throughput-scalar | tee throughput_caches.out
likwid-mpirun -np 40 -f -g FLOPS_DP -m -O ./applications/sintering/sintering-throughput-scalar | tee throughput_flops.out
```

```
likwid-mpirun -np 40 -f -g CACHES   -m -O ./applications/sintering/sintering-throughput-tensorial | tee throughput_caches_tensorial.out
likwid-mpirun -np 40 -f -g FLOPS_DP -m -O ./applications/sintering/sintering-throughput-tensorial | tee throughput_flops_tensorial.out
```
