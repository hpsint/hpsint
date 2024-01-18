import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import pathlib
import collections.abc
import library

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

for f, lbl, clr in zip(files_list, labels, colors):

    fdata = np.genfromtxt(f, dtype=None, names=True, delimiter=args.delimiter)

    alpha = 1

    mask = (args.limits[0] <= fdata["time"]) & (fdata["time"] <= args.limits[1])

    if args.save:

        # Determine the number of active fields that exist in the file
        n_active = 0
        for i in range(n_qtys):
            if header[i] in fdata.dtype.names and active[i]:
                n_active += 1

        csv_data = np.empty((len(fdata["time"][mask]), 2 * n_active + 1), float)
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
            while ref0 == 0:
                i_ref += 1
                ref0 = fdata[qty_name][i_ref]

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
