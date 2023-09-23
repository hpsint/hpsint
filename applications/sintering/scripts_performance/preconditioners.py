import json
import os
import sys

JOB="""#!/bin/bash
#SBATCH --nodes={1}
#SBATCH --ntasks-per-node=40
#SBATCH --time=16:00:00
#SBATCH -J job-precon-{0}
#SBATCH --partition gold

mkdir -p /scratch/munch
cd /scratch/munch


mpirun -np {2} {3}/../applications/sintering/sintering-3D-generic-scalar \\
    --cloud {3}/../../applications/sintering/sintering_cloud_examples/packings_10k/51particles.cloud \\
    {3}/job_{0}.json | tee {3}/job_{0}.out
"""

JOB_SKX="""#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J LIKWID
#Output and error (also --output, --error):
#SBATCH -o job_{0}_%j.out
#SBATCH -e job_{0}_%j.e
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
#SBATCH --nodes={1}
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1

#module list

source ~/.bashrc

pwd

mpirun -np {2} ../applications/sintering/sintering-3D-generic-scalar --cloud ../../applications/sintering/sintering_cloud_examples/packings_10k/51particles.cloud ./job_{0}.json | tee ./job_{0}.out

"""

counter = 0

def run_instance(time_end, n_nodes, use_gold, config):
    global counter 

    with open(os.path.dirname(os.path.abspath(__file__)) + "/../analysis_examples/49particles_delta_min_40.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["TimeIntegration"]["TimeEnd"] = time_end

    config = config.split("|")

    if len(config) == 1:
        if config[0] == "ILU":
            datastore["Preconditioners"]["OuterPreconditioner"] = "ILU"
        else:
            print("Not implemented!")
    else:
        datastore["Preconditioners"]["OuterPreconditioner"] = "BlockPreconditioner2"

        if config[0] == "ILU":
            datastore["Preconditioners"]["BlockPreconditioner2"]["Block0Preconditioner"] = "ILU"
        else:
            print("Not implemented!")


        if config[1] == "ILU":
            datastore["Preconditioners"]["BlockPreconditioner2"]["Block1Preconditioner"] = "ILU"
        elif config[1] == "BlockILU":
            datastore["Preconditioners"]["BlockPreconditioner2"]["Block1Preconditioner"] = "BlockILU"
            datastore["Preconditioners"]["BlockPreconditioner2"]["Block1Approximation"]  = config[2]

    # write data to output file
    with open("./job_%d.json" % (counter), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

    with open("./job_%d.cmd" % (counter), 'w') as f:
        if use_gold:
            f.write(JOB.format(counter, n_nodes, n_nodes * 40, os.getcwd()))
        else:
            label = ""
            if n_nodes <= 16:
                label = "test"
            elif n_nodes <= 768:
                label = "general"
            elif n_nodes <= 3072:
                label = "large"

            f.write(JOB_SKX.format(counter, n_nodes, n_nodes * 48, label))


    counter = counter + 1

def main():
    time_end = 500
    n_nodes  = 4
    use_gold = True

    short_simulation = True

    if (len(sys.argv) > 1) and (sys.argv[1] == "1"):
        short_simulation = False
        time_end = 15000

    if short_simulation:
        run_instance(time_end, n_nodes + 1, use_gold, "ILU")

    if short_simulation:
        run_instance(time_end, n_nodes, use_gold, "ILU|ILU")
    run_instance(time_end, n_nodes, use_gold, "ILU|BlockILU|all")

    if short_simulation:
        run_instance(time_end, n_nodes, use_gold, "ILU|BlockILU|const")

    run_instance(time_end, n_nodes, use_gold, "ILU|BlockILU|avg")
    run_instance(time_end, n_nodes, use_gold, "ILU|BlockILU|max")


if __name__== "__main__":
  main()
