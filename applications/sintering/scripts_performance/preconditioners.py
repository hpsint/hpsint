import json
import os

JOB="""#PBS -l nodes={1}:ppn=40
#PBS -l walltime=16:00:00
#PBS -N job-precon-{0}
#PBS -q gold
#PBS -j oe

cd ~/sw-sintering/pf-applications/build/preconditioners

mpirun -np {2} ../applications/sintering/sintering-3D-generic-scalar --cloud ../../applications/sintering/sintering_cloud_examples/49particles.cloud ./job_{0}.json | tee ./job_{0}.out
"""

def run_instance(counter, time_end, n_nodes, config):
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
        f.write(JOB.format(counter, n_nodes, n_nodes * 40))

def main():
    time_end = 500
    n_nodes  = 5

    run_instance(0, time_end, n_nodes, "ILU")
    run_instance(1, time_end, n_nodes, "ILU|ILU")
    run_instance(2, time_end, n_nodes, "ILU|BlockILU|all")
    run_instance(3, time_end, n_nodes, "ILU|BlockILU|const")
    run_instance(4, time_end, n_nodes, "ILU|BlockILU|avg")
    run_instance(5, time_end, n_nodes, "ILU|BlockILU|max")


if __name__== "__main__":
  main()
