from vtk import vtkXMLUnstructuredGridReader
from functools import reduce
import glob
import re
import numpy as np
import argparse

# Script arguments
parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-p", "--plot", help="Plot results", action="store_true", default=True)
parser.add_argument("-m", "--mask", type=str, help="File mask", required=True)
parser.add_argument("-o", "--output", type=str, help="Output csv file", required=False, default="")

args = parser.parse_args()

# Get vtk files according to the mask and sort it by number
vtk_files_list = glob.glob(args.mask)
vtk_files_list.sort(key=lambda f: int(re.sub('\D', '', f)))

# Build reader
reader = vtkXMLUnstructuredGridReader()

# CSV data
csv_header = ["dim_x", "dim_y", "dim_z", "volume", "shrinkage_x", "shrinkage_y", "shrinkage_z", "densification"]
n_rows = len(vtk_files_list)
n_cols = len(csv_header)
csv_data = np.zeros(shape=(n_rows, n_cols))

# Values of quantities of interest at t=0
vals0 = [0, 0, 0, 0]
n_qtys = len(vals0)

for idx, vtk_file in enumerate(vtk_files_list):

    print("Parsing file {} ({}/{})".format(vtk_file, idx + 1, n_rows))

    # Read next file
    reader.SetFileName(vtk_file)
    reader.Update()
    vtk_mesh = reader.GetOutput()

    # Get bounds of the domain and compute its size
    bounds = reader.GetOutput().GetBounds()
    dim = []
    for i in range(0, 6, 2):
        dim.append(bounds[i+1] - bounds[i])

    # Store sizes over Cartesian axes
    for i in range(3):
        csv_data[idx, i] = dim[i]

    # Store volume
    csv_data[idx, 3] = reduce(lambda x, y: x*(y if y > 0 else 1.), dim)
    
    # Compute shrinkage and densification
    if idx > 0:
        for i in range(n_qtys):
            csv_data[idx, i+n_qtys] = (vals0[i] - csv_data[idx, i]) / vals0[i] if vals0[i] > 0 else 0.

    else:
        for i in range(n_qtys):
            vals0[i] = csv_data[idx, i]

# Save to csv
if len(args.output) > 0:
    np.savetxt(args.output, csv_data, header=','.join(csv_header), comments='', delimiter=',')

# Plot graphs
if args.plot:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Shrinkage and densification')

    for i in range(n_qtys):
        axes[0].plot(csv_data[:,i], label=csv_header[i])
        axes[1].plot(csv_data[:,i+n_qtys], label=csv_header[i+n_qtys])

    for i in range(2):
        axes[i].grid(True)
        axes[i].legend()

    plt.show()