```bash
mpirun -np 40 ./benchmarks/layer_wise ../benchmarks/layer_wise_00.json | tee layer_wise_00.output
mpirun -np 40 ./benchmarks/layer_wise ../benchmarks/layer_wise_01.json | tee layer_wise_01.output
mpirun -np 40 ./benchmarks/layer_wise ../benchmarks/layer_wise_02.json | tee layer_wise_02.output
mpirun -np 40 ./benchmarks/layer_wise ../benchmarks/layer_wise_03.json | tee layer_wise_03.output

```