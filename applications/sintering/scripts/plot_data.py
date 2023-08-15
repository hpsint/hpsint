import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import library

def create_axes(n):
    if n==1:
        fig, ax = plt.subplots(1, 1)
        return fig, [ax]
    elif n==2:
        return plt.subplots(1, 2)
    elif n==3:
        return plt.subplots(2, 2)
    elif n==4:
        return plt.subplots(2, 2)
    elif n==5:
        return plt.subplots(2, 3)
    elif n==6:
        return plt.subplots(2, 3)
    elif n==7:
        return plt.subplots(2, 4)
    elif n==8:
        return plt.subplots(2, 4)
    elif n==9:
        return plt.subplots(3, 3)
    else:
        exit("Maximum 9 plots can be created")

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--files", dest="files", nargs='+', required=True, help="Source filenames, can be defined as masks")
parser.add_argument("-x", "--xaxis", dest="xaxis", required=False, help="x-axis variable", default="time")
parser.add_argument("-y", "--yaxes", dest="yaxes", nargs='+', required=False, help="y-axis variables")
parser.add_argument("-l", "--limits", dest='limits', required=False, nargs=2, help="Limits for x-axis", type=float, default=None)
parser.add_argument("-m", "--markers", dest='markers', required=False, help="Number of markers", type=int, default=30)
parser.add_argument("-c", "--collapse", dest='collapse', required=False, help="Shorten labels", action="store_true", default=False)
parser.add_argument("-s", "--skip-first", dest='skip_first', required=False, help="Skip first entry", action="store_true", default=True)
parser.add_argument("-g", "--single-legend", dest='single_legend', required=False, help="Use single legend", action="store_true", default=False)

args = parser.parse_args()

# Get all files to process
files_list = library.get_solutions(args.files)

colors = library.get_hex_colors(len(files_list))

if not args.yaxes:
    print("You did not specify any y-axes field to plot. These are the fields shared between all the provided files:")

    available_fields = None
    for f in files_list:
        cdata = np.genfromtxt(f, dtype=None, names=True)
        if available_fields is None:
            available_fields = set(cdata.dtype.names)
        else:
            available_fields = available_fields.intersection(set(cdata.dtype.names))

    sorted_available_fields = sorted(list(available_fields))
        
    for name in list(sorted_available_fields):
        print(" -- {}".format(name))
    
    print("Rerun the script specifying at least one of them")
    exit()

print("")
print("         x-axis: {}".format(args.xaxis))
print(" fields to plot: " + ", ".join(args.yaxes))
print("number of files: {}".format(len(files_list)))

n_fields = len(args.yaxes)
fig, ax = create_axes(n_fields)

# Generate labels
if args.collapse:
    labels = library.generate_short_labels(files_list)
else:
    labels = files_list

for i in range(n_fields):
    field = args.yaxes[i]
    a = ax[i]

    x_lims = [sys.float_info.max, -sys.float_info.max]

    for f, lbl, clr in zip(files_list, labels, colors):
        cdata = np.genfromtxt(f, dtype=None, names=True)
        
        mask = [True] * len(cdata[args.xaxis])
        if args.limits:
            mask = (args.limits[0] <= cdata[args.xaxis]) & (cdata[args.xaxis] <= args.limits[1])    
        elif args.skip_first:
            mask[0] = False
        
        x = cdata[args.xaxis][mask]
        y = cdata[field][mask]

        x_lims[0] = min(x_lims[0], x[0])
        x_lims[1] = max(x_lims[1], x[-1])

        a.plot(x, y, color=clr, linestyle='-', linewidth=2, label=lbl)

    a.grid(True)
    a.set_xlabel(args.xaxis)
    a.set_ylabel(field)
    a.set_title(field)
    a.set_xlim(x_lims)

    if not args.single_legend:
        a.legend()
    elif i == n_fields - 1:
        a_handles, a_labels = a.get_legend_handles_labels()
        fig.legend(a_handles, a_labels, loc='upper center')

plt.show()
