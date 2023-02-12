import json
import os

JOB="""#PBS -l nodes={1}:ppn=40
#PBS -l walltime=16:00:00
#PBS -N job-precon-{0}
#PBS -q gold
#PBS -j oe

cd ~/sw-sintering/pf-applications/build/configurations

mpirun -np {2} ../applications/sintering/sintering-3D-generic-{3} --cloud ../../applications/sintering/sintering_cloud_examples/49particles.cloud ./job_{0}.json | tee ./job_{0}.out
"""

def run_instance(counter, time_end, n_nodes, jacobian_free = False, cut_off = False, tenosrial = False, advection = False):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/../analysis_examples/49particles_delta_min_40.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["TimeIntegration"]["TimeEnd"] = time_end

    if jacobian_free:
        datastore["NonLinearData"]["JacobiFree"] = True

    if cut_off:
        datastore["GrainCutOffTolerance"] = 0.00001

    if advection:
        datastore["Advection"]["Enable"] = True

    type = "scalar"

    if tenosrial:
        type = "tensorial"


    # write data to output file
    with open("./job_%d.json" % (counter), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

    with open("./job_%d.cmd" % (counter), 'w') as f:
        f.write(JOB.format(counter, n_nodes, n_nodes * 40, type))

def main():
    time_end = 500
    n_nodes  = 5

    run_instance(0, time_end, n_nodes)
    run_instance(1, time_end, n_nodes, jacobian_free=True)
    run_instance(2, time_end, n_nodes, cut_off=True)
    run_instance(3, time_end, n_nodes, jacobian_free=True, cut_off=True)

    run_instance(4, time_end, n_nodes, tenosrial=True, jacobian_free=True)
    run_instance(5, time_end, n_nodes, tenosrial=True, jacobian_free=True, cut_off=True)

    run_instance(6, time_end, n_nodes, advection=True, jacobian_free=True)
    run_instance(7, time_end, n_nodes, advection=True, jacobian_free=True, cut_off=True)


if __name__== "__main__":
  main()