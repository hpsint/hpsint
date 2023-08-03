import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import collections.abc
import library

parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-f", "--files", dest="files", nargs='+', required=True, help="Source filenames, can be defined as masks")
parser.add_argument("-d", "--directions", dest="directions", nargs='+',
    required=False, default="all", help="Directions", choices=['x', 'y', 'z', 'vol', 'all'])
parser.add_argument("-l", "--limits", dest='limits', nargs=2, required=False, help="Limits for x-axis", type=float, default=[-sys.float_info.max, sys.float_info.max])
parser.add_argument("-m", "--markers", dest='markers', required=False, help="Number of markers", type=int, default=30)
parser.add_argument("-c", "--collapse", dest='collapse', required=False, help="Shorten labels", action="store_true", default=False)

args = parser.parse_args()

if not isinstance(args.directions, collections.abc.Sequence):
    args.directions = [args.directions]

# Get all files to process
files_list = library.get_solutions(args.files)

colors = library.get_hex_colors(len(files_list))

csv_header = ["dim_x", "dim_y", "dim_z", "volume", "shrinkage_x", "shrinkage_y", "shrinkage_z", "densification"]
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
    labels = library.generate_short_labels(files_list)
else:
    labels = files_list

for f, lbl, clr in zip(files_list, labels, colors):

    fdata = np.genfromtxt(f, dtype=None, names=True)

    alpha = 1

    for i in range(n_qtys):
        if csv_header[i] in fdata.dtype.names and active[i]:
            qty_name = csv_header[i]

            ref0 = fdata[qty_name][0]
            ref_qty = (ref0 - fdata[qty_name]) / ref0

            mask = (args.limits[0] <= fdata["time"]) & (fdata["time"] <= args.limits[1])

            m_type, n_every = library.get_markers(i, len(fdata["time"][mask]), args.markers, markers)

            axes[0].plot(fdata["time"][mask], fdata[qty_name][mask], label=" ".join([lbl, csv_header[i]]), 
                marker=m_type, color=clr, alpha=alpha, markevery=n_every)
            axes[1].plot(fdata["time"][mask], ref_qty[mask], label=" ".join([lbl, csv_header[i+n_qtys]]),
                marker=m_type, color=clr, alpha=alpha, markevery=n_every)

            alpha -= 0.2

for i in range(2):
    axes[i].grid(True)
    axes[i].legend()

plt.show()
