import collections
import json
import os
import sys 

counter = 0

def run_instance(reference_file, output_file, sim_time, options):
    with open(reference_file, 'r') as f:
       datastore = json.load(f, object_pairs_hook=collections.OrderedDict)

    # make modifications
    datastore["TimeIntegration"]["TimeEnd"] = sim_time

    for option in options:
        path = option[0].split(":")

        if len(path)==1:
            datastore[path[0]] = option[1]
        elif len(path)==2:
            datastore[path[0]][path[1]] = option[1]
        elif len(path)==3:
            datastore[path[0]][path[1]][path[2]] = option[1]
        elif len(path)==4:
            datastore[path[0]][path[1]][path[2]][path[3]] = option[1]
        else:
            raise Exception('Not implemented!')

        datastore["Restart"]["Type"] = "never"

    # write data to output file
    with open(output_file, 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():    
    
    reference_file = sys.argv[1]
    sim_time_0     = sys.argv[2]
    sim_time_1     = sim_time_0 + sys.argv[3]

    run_instance(reference_file, "reference.json", sim_time_0, [])

    def run_parameter(options): 
        global counter
        run_instance(reference_file, "./input_%s.json" % (str(counter).zfill(4)), sim_time_1, options)
        counter = counter + 1;

    # ILU
    run_parameter([["Preconditioners:OuterPreconditioner", "ILU"]])

    # ILU + ILU/AMG/Diagonal
    for precon1 in ["ILU", "AMG", "InverseDiagonalMatrix"]:
        run_parameter([
            ["Preconditioners:OuterPreconditioner", "BlockPreconditioner2"],
            ["Preconditioners:BlockPreconditioner2:Block0Preconditioner", "ILU"],
            ["Preconditioners:BlockPreconditioner2:Block1Preconditioner", precon1]
            ])

    # ILU + BlockILU-X
    for precon1 in ["all", "const", "max", "avg"]:
        run_parameter([
            ["Preconditioners:OuterPreconditioner", "BlockPreconditioner2"],
            ["Preconditioners:BlockPreconditioner2:Block0Preconditioner", "ILU"],
            ["Preconditioners:BlockPreconditioner2:Block1Preconditioner", "BlockILU"],
            ["Preconditioners:BlockPreconditioner2:Block1Approximation", precon1]
            ])


if __name__== "__main__":
  main()
