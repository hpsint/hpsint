import json
import os

JOB="""#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J LIKWID
#Output and error (also --output, --error):
#SBATCH -o node-{1}.out
#SBATCH -e node-{1}.e
#Initial working directory (also --chdir):
#SBATCH -D ./
#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=peter.muench@uni-a.de
# Wall clock limit:
#SBATCH --time=0:30:00
#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --account=pn36li
#
## #SBATCH --switches=4@24:00:00
#SBATCH --partition={3}
#Number of nodes and MPI tasks per node:
#SBATCH --nodes={0}
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1

#module list

source ~/.bashrc

pwd

for i in 51 102 212 316 603 1037 3076 6140 10245
do
    mpirun -np {2} ../applications/sintering/sintering-3D-generic-scalar --cloud ../../applications/sintering/sintering_cloud_examples/packings_10k/${{i}}particles.cloud ./input.json || exit 1
done

"""

def configure_input(n_time_end):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/../analysis_examples/49particles_delta_min_40.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["TimeIntegration"]["MaxNTimeSteps"] = n_time_end


    # write data to output file
    with open("./input.json", 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():
    # settings
    n_time_end = 10
    max_nodes = 1000000

    configure_input(n_time_end)

    # create job files
    for n in [ a for a in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072] if a <= max_nodes]:
        label = ""
        if n <= 16:
            label = "test"
        elif n <= 768:
            label = "general"
        elif n <= 3072:
            label = "large"

        with open("node-%s.cmd" % (str(n).zfill(4)), 'w') as f:
            f.write(JOB.format(str(n), str(n).zfill(4), 48*n, label))

if __name__== "__main__":
  main()

