import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import re
import os
import math
import pathlib
import library
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from sklearn.neighbors import BallTree

parser = argparse.ArgumentParser(description='Compute densification using the packing convex hull from the grain tracker data')
parser.add_argument("-m", "--mask", type=str, help="File mask", required=True)
parser.add_argument("-f", "--file", type=str, help="Solution file", required=False, default="solution.log")
parser.add_argument("-p", "--path", type=str, help="Common path", required=False, default=None)
parser.add_argument("-o", "--output", type=str, required=False, help="Destination csv file", default=None)
parser.add_argument("-c", "--collapse", dest='collapse', required=False, help="Shorten labels", action="store_true", default=False)
parser.add_argument("-e", "--extend-to", dest='extend_to', required=False, help="Extend labels when shortening to", type=str, default=None)
parser.add_argument("-u", "--suffix", dest='suffix', required=False, help="Suffix to append to the save file", type=str, default="_chull_density")
parser.add_argument("-s", "--save", dest='save', required=False, help="Save shrinkage data", action="store_true", default=False)
parser.add_argument("-k", "--skip-plot", action='store_true', help="Skip plots", required=False, default=False)
parser.add_argument("-g", "--grid", dest='grid', required=False, help="Number of mesh points along the smallest dimension", type=int, default=50)

# Parse arguments
args = parser.parse_args()

# Deal with path names
if args.path is not None:
    list_solution_files = library.get_solutions([os.path.join(args.path, args.file)])
    list_distribution_folders = [os.path.dirname(s) for s in list_solution_files]
    print("")

    if not list_solution_files:
        raise Exception("No files detected that would fit the provided masks")

else:
    list_solution_files = [args.file]
    list_distribution_folders = [os.path.dirname(args.file)]

    if not os.path.isfile(args.file):
        raise Exception("The provided solution file does not exist")

# Read distribution data
list_distributions = []
for f in list_distribution_folders:
    files_list = glob.glob(os.path.join(f, args.mask))
    files_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    list_distributions.append(files_list)

# Save csv names
if args.save:
    if args.output is not None:
        file_names = library.generate_short_labels(list_solution_files, args.extend_to)
        csv_names = [os.path.join(args.output, n.replace(os.sep, "_") + args.suffix + ".csv")  for n in file_names]
    else:
        csv_names = [os.path.splitext(f)[0] + args.suffix + ".csv" for f in list_solution_files]

# Generate labels
if args.collapse:
    labels = library.generate_short_labels(list_solution_files, args.extend_to)
else:
    labels = list_solution_files.copy()

# Axes for plottign results
fig, axes = plt.subplots(nrows=1, ncols=3)
fig.suptitle('Shrinkage and densification')

f_counter = 0
n_folders = len(list_solution_files)

colors = library.get_hex_colors(n_folders)

for file_solution, files_list, lbl, clr in zip(list_solution_files, list_distributions, labels, colors):

    print("Parsing folder  {} ({}/{})".format(os.path.dirname(file_solution), f_counter + 1, n_folders))

    # Read solution file
    fdata = np.genfromtxt(file_solution, dtype=None, names=True)

    # Total number of stats files
    n_rows = len(files_list)

    if n_rows == 0:
        continue

    # Build a CSV header
    csv_header = ["time", "dt", "n_grains", "solid_vol", "hull_vol", "rel_dens_hull", "rel_dens_mc"]
    n_qtys = len(csv_header)
    csv_header = " ".join(csv_header)

    # Build CSV format
    if args.save:
        csv_format = ["%g"] * n_qtys
        csv_format = " ".join(csv_format)

    # Init CSV data
    csv_data = np.empty((n_rows, n_qtys), float)

    for idx, log_file in enumerate(files_list):

        if idx >= fdata.shape[0]:
            prefix = "├" if idx + 1 < n_rows else "└"
            print("{}─ Skipping file {} ({}/{}) due to data inconsistency".format(prefix, log_file, idx + 1, n_rows))
            continue

        is3D = False
        qdata = np.genfromtxt(log_file, dtype=None, names=True)
        if 'x' not in qdata.dtype.names or 'y' not in qdata.dtype.names:
            raise Exception("There is no either 'x' or 'y' coordinate specified in the provided output file")
        
        if 'z' in qdata.dtype.names:
            is3D = True

        if np.isnan(float(qdata['volume'][0])):
            prefix = "├" if idx + 1 < n_rows else "└"
            print("{}─ Skipping file {} ({}/{}) due to nan values present".format(prefix, log_file, idx + 1, n_rows))
            continue
        
        dim = 3 if is3D else 2

        n_grains = int(fdata["n_grains"][idx])
        grain_coordinates = np.zeros((n_grains, dim))
        grain_radii = np.zeros((n_grains))
        
        for gid, row in enumerate(qdata):
            measure = float(row['volume'])

            if measure > 0:
                radius = math.pow(3./4. * measure/np.pi, 1./3.) if is3D else math.sqrt(measure/np.pi)
            else:
                radius = 0

            grain_radii[gid] = radius

            grain_coordinates[gid, 0] = float(row['x'])
            grain_coordinates[gid, 1] = float(row['y'])
            if is3D:
                grain_coordinates[gid, 2] = float(row['z'])

        # Create MC points mesh
        bottom_left = np.min(grain_coordinates, 0)
        top_right = np.max(grain_coordinates, 0)
        domain_size = top_right - bottom_left

        ref_size = np.min(domain_size)
        ref_dim = np.where(domain_size == ref_size)[0][0]
        ref_grid_size = ref_size / args.grid

        grid_splits = [0]*dim
        grid_size = [0]*dim
        grid_ranges = [None]*dim

        grid_splits[ref_dim] = args.grid
        grid_size[ref_dim] = ref_size / args.grid
        for i in range(dim):
            if i != ref_dim:
                grid_splits[i] = round(domain_size[i] / ref_grid_size)
                grid_size[i] = domain_size[i] / grid_splits[i]
            
            grid_ranges[i] = np.arange(bottom_left[i], 1.0001*top_right[i], grid_size[i])

        if dim == 2:
            X, Y = np.meshgrid(grid_ranges[0], grid_ranges[1])
            mc_points = np.vstack(list(zip(X.ravel(), Y.ravel())))
        elif dim == 3:
            X, Y, Z = np.meshgrid(grid_ranges[0], grid_ranges[1], grid_ranges[2])
            mc_points = np.vstack(list(zip(X.ravel(), Y.ravel(), Z.ravel())))
        else:
            raise Exception("Invalid dimensionality")
        
        # Eliminate the points outside of the convex hull
        chull = ConvexHull(grain_coordinates)
        triangulation = Delaunay(grain_coordinates)

        n_mc_points = len(mc_points)
        filter = [False]*n_mc_points
        for i in range(mc_points.shape[0]):
            p = mc_points[i,...]
            filter[i] = triangulation.find_simplex(p) >= 0

        mc_points = mc_points[filter]
        n_mc_points = len(mc_points)

        # Build ball tree
        ball_tree = BallTree(mc_points, leaf_size=16, metric='euclidean')
        coverage = ball_tree.query_radius(grain_coordinates, r=grain_radii)

        # Mark sample points which are covered by at least one grain
        mc_covered = [0]*n_mc_points
        for cvr in coverage:
            for p in cvr:
                mc_covered[p] = 1

        ratio_covered = sum(mc_covered) / n_mc_points

        csv_data[idx, 0] = fdata["time"][idx]
        csv_data[idx, 1] = fdata["dt"][idx]
        csv_data[idx, 2] = fdata["n_grains"][idx]
        csv_data[idx, 3] = fdata["solid_vol"][idx]
        csv_data[idx, 4] = chull.volume
        csv_data[idx, 5] = csv_data[idx, 3] / csv_data[idx, 4]
        csv_data[idx, 6] = ratio_covered

        prefix = "├" if idx + 1 < n_rows else "└"
        print("{}─ Parsing file {} ({}/{})".format(prefix, log_file, idx + 1, n_rows))

    if not args.skip_plot:
        axes[0].plot(csv_data[:,0], csv_data[:,4], label=lbl, color=clr)
        axes[1].plot(csv_data[:,0], csv_data[:,5], label=lbl, color=clr)
        axes[2].plot(csv_data[:,0], csv_data[:,6], label=lbl, color=clr)

    if args.save:
        file_path = csv_names.pop(0)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        np.savetxt(file_path, csv_data, delimiter=' ', header=csv_header, fmt=csv_format, comments='')
        print("   Saving result to {}".format(file_path))
        print("")

    f_counter += 1

if not args.skip_plot:

    titles = ['convex hull volume', 'density from hull', 'density from MC']
    labels = ['volume', 'rel_density', 'rel_density']

    for i in range(3):
        axes[i].grid(True)
        axes[i].legend()
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("time")
        axes[i].set_ylabel(labels[i])

    plt.show()
