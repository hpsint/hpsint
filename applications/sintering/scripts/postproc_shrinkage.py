import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import pathlib
import collections.abc
import library
import numpy.lib.recfunctions as rfn

parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-f", "--files", dest="files", nargs='+', required=True, help="Source filenames, can be defined as masks")
parser.add_argument("-d", "--directions", dest="directions", nargs='+',
    required=False, default="all", help="Directions", choices=['x', 'y', 'z', 'vol', 'all'])
parser.add_argument("-l", "--limits", dest='limits', nargs=2, required=False, help="Limits for x-axis", type=float, default=[-sys.float_info.max, sys.float_info.max])
parser.add_argument("-m", "--markers", dest='markers', required=False, help="Number of markers", type=int, default=30)
parser.add_argument("-c", "--collapse", dest='collapse', required=False, help="Shorten labels", action="store_true", default=False)
parser.add_argument("-e", "--extend-to", dest='extend_to', required=False, help="Extend labels when shortening to", type=str, default=None)
parser.add_argument("-s", "--save", dest='save', required=False, help="Save shrinkage data", action="store_true", default=False)
parser.add_argument("-o", "--output", dest='output', required=False, help="Destination path to output csv files", type=str, default=None)
parser.add_argument("-b", "--labels", dest='labels', required=False, nargs='+', help="Customized labels", default=None)
parser.add_argument("-r", "--delimiter", dest='delimiter', required=False, help="Input file delimiter", default=None)

args = parser.parse_args()

if not isinstance(args.directions, collections.abc.Sequence):
    args.directions = [args.directions]

# Get all files to process
files_list = library.get_solutions(args.files)

if not files_list:
    raise Exception("The files list is empty, nothing to postprocess")

colors = library.get_hex_colors(len(files_list))

header = ["dim_x", "dim_y", "dim_z", "volume", "shrinkage_x", "shrinkage_y", "shrinkage_z", "densification"]
n_qtys = 4
markers = ["s", "D", "o", "x"]

if 'all' in args.directions:
    active = [True]*n_qtys
else:
    active = [(item in args.directions) for item in ['x', 'y', 'z', 'vol']]

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.suptitle('Shrinkage and densification')

# Generate labels
if args.collapse:
    labels = library.generate_short_labels(files_list, args.extend_to)
else:
    labels = files_list.copy()

# If we have custom labels
if args.labels:
    for i in range(min(len(labels), len(args.labels))):
        labels[i] = args.labels[i]

# CSV names
if args.save:
    if args.output is not None:
        file_names = library.generate_short_labels(files_list, args.extend_to) if not args.collapse else labels
        csv_names = [os.path.join(args.output, n.replace(os.sep, "_") + "_shrinkage.csv")  for n in file_names]
    else:
        csv_names = [os.path.splitext(f)[0] + "_shrinkage.csv" for f in files_list]

# Pairs for getting graint tracker ouput results
dirs_from_velocities = {"x": "vx", "y": "vy", "z": "vz"}

for f, lbl, clr in zip(files_list, labels, colors):

    fdata = np.genfromtxt(f, dtype=None, names=True, delimiter=args.delimiter)

    alpha = 1

    # Try get info from particles - if we estimate shrinkage from the GT data
    particles_list = {"x": [], "y": [], "z": []}
    for key, value in dirs_from_velocities.items():
        has_key = [i for i in fdata.dtype.names if i.startswith(key)]
        has_value = [i for i in fdata.dtype.names if i.startswith(value)]

        for p in has_key:
            ids = p.split("_")
            q = "v" + p
            if q in has_value:
                particles_list[ids[0]].append((p, q))

    if particles_list:
        indices = {"x": 0, "y": 1, "z": 2}

        for direction, items in particles_list.items():
            i = indices[direction]

            if active[i] and items:
                qty_name = header[i]

                # Find the reference particles
                ref_particles = {"min": None, "max": None}

                pos_min0 = np.inf
                pos_max0 = -np.inf
                for p in items:
                    if fdata[p[0]][0] < pos_min0:
                        pos_min0 = fdata[p[0]][0]
                        ref_particles["min"] = p
                    if fdata[p[0]][0] > pos_max0:
                        pos_max0 = fdata[p[0]][0]
                        ref_particles["max"] = p

                ref0 = pos_max0 - pos_min0

                displ_min = np.nan_to_num(fdata["dt"]) * np.nan_to_num(fdata[ref_particles["min"][1]])
                displ_max = np.nan_to_num(fdata["dt"]) * np.nan_to_num(fdata[ref_particles["max"][1]])

                pos_min = np.full_like(displ_min, 0)
                pos_min[0] = pos_min0

                pos_max = np.full_like(displ_max, 0)
                pos_max[0] = pos_max0

                for idx, val in np.ndenumerate(pos_min):
                    i = idx[0]
                    if i > 0:
                        pos_min[i] = pos_min[i-1] + displ_min[i]
                        pos_max[i] = pos_max[i-1] + displ_max[i]

                length = pos_max - pos_min

                fdata = rfn.append_fields(fdata, "dim_" + direction, length)

                if direction == 'x':
                    fdata = rfn.append_fields(fdata, "volume", length)
                else:
                    fdata["volume"] *= length

    mask = (args.limits[0] <= fdata["time"]) & (fdata["time"] <= args.limits[1])

    if args.save:

        # Determine the number of active fields that exist in the file
        n_active = 0
        for i in range(n_qtys):
            if header[i] in fdata.dtype.names and active[i]:
                n_active += 1

        csv_data = np.zeros((len(fdata["time"][mask]), 2 * n_active + 1))
        csv_data[:,0] = fdata["time"][mask]

        csv_format = ["%g"] * (2*n_active + 1)
        csv_format = " ".join(csv_format)

        csv_header = [""] * (2*n_active + 1)
        csv_header[0] = "time"

        i_active = 0
        for i in range(n_qtys):
            if header[i] in fdata.dtype.names and active[i]:
                csv_header[i_active+1] = header[i]
                csv_header[i_active+n_active+1] = header[i+n_qtys]
                i_active += 1

        csv_header = " ".join(csv_header)

    i_active = 0
    for i in range(n_qtys):
        if header[i] in fdata.dtype.names and active[i]:
            qty_name = header[i]

            # In case the first entry is zero, then pick the next non-zero
            i_ref = 0
            ref0 = fdata[qty_name][i_ref]
            while ref0 == 0 and i_ref < len(fdata[qty_name]) - 1:
                i_ref += 1
                ref0 = fdata[qty_name][i_ref]

            # Skip field if nothing to plot
            if ref0 == 0:
                continue

            ref_qty = (ref0 - fdata[qty_name]) / ref0

            mask_plt = mask[:]
            for i in range(i_ref):
                mask_plt[i] = False

            m_type, n_every = library.get_markers(i, len(fdata["time"][mask_plt]), args.markers, markers)

            axes[0].plot(fdata["time"][mask_plt], fdata[qty_name][mask_plt], label=" ".join([lbl, header[i]]), 
                marker=m_type, color=clr, alpha=alpha, markevery=n_every)
            axes[1].plot(fdata["time"][mask_plt], ref_qty[mask_plt], label=" ".join([lbl, header[i+n_qtys]]),
                marker=m_type, color=clr, alpha=alpha, markevery=n_every)

            alpha -= 0.2

            if args.save:
                csv_data[:,i_active+1] = fdata[qty_name][mask]
                csv_data[:,i_active+n_active+1] = ref_qty[mask]
                i_active += 1

    if args.save:
        file_path = csv_names.pop(0)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        np.savetxt(file_path, csv_data, delimiter=' ', header=csv_header, fmt=csv_format, comments='')

for i in range(2):
    axes[i].grid(True)
    axes[i].legend()

plt.show()
