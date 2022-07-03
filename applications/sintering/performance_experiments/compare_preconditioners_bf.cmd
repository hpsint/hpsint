#!/bin/sh -f
#SBATCH --exclude=node01,node02,node03,node04,node05,node06,node07,node08,node09,node10,node11,node12,node13,node14,node15,node16,node17,node18,node19,node20,node21,node22,node23,node24,node25,node26
#SBATCH --nodes=$1
#SBATCH --ntasks-per-node=24
#SBATCH -o 49particles.o
#SBATCH -e 49particles.e
#SBATCH --mail-user=munch@lnm.mw.tum.de 
#SBATCH --mail-type=BEGIN,END,FAIL
#Job name:
#SBATCH -J test
# Walltime:
#SBATCH --time=24:00:00

#module purge
#module load intel-studio-2016 mpi/openmpi/intel/1.10.1 comp/gcc/9.1.0

pwd

source /etc/profile.d/modules.sh

python ../../applications/sintering/performance_experiments/compare_preconditioners.py $2 $3 $4

mpiexec -np $SLURM_NPROCS ../applications/sintering/sintering-3D --cloud ../applications/sintering/sintering_cloud_examples/49particles.cloud reference.json | tee reference.output

for file in input_*.json
do
  mpiexec -np $SLURM_NPROCS ../applications/sintering/sintering-3D --restart restart_0 $file | tee ${file::-5}.output
done