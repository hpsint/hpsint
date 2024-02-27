import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import library
from collections.abc import Iterable
from itertools import cycle, islice

def create_axes(n):
    ax = None
    if n==1:
        fig, ax = plt.subplots(1, 1)
        ax = [ax]
    elif n==2:
        fig, axr = plt.subplots(1, 2)
    elif n==3:
        fig, axr = plt.subplots(2, 2)
    elif n==4:
        fig, axr = plt.subplots(2, 2)
    elif n==5:
        fig, axr = plt.subplots(2, 3)
    elif n==6:
        fig, axr = plt.subplots(2, 3)
    elif n==7:
        fig, axr = plt.subplots(2, 4)
    elif n==8:
        fig, axr = plt.subplots(2, 4)
    elif n==9:
        fig, axr = plt.subplots(3, 3)
    else:
        exit("Maximum 9 plots can be created")

    if ax is None:
        ax = []
        for x in axr:
            if isinstance(x, Iterable):
                for y in x:
                    ax.append(y)
            else:
                ax.append(x)

    return fig, ax

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--files", dest="files", nargs='+', required=True, help="Source filenames, can be defined as masks")
parser.add_argument("-x", "--xaxis", dest="xaxis", required=False, help="x-axis variable", default="time")
parser.add_argument("-y", "--yaxes", dest="yaxes", nargs='+', required=False, help="y-axis variables")
parser.add_argument("-l", "--limits", dest='limits', required=False, nargs=2, help="Limits for x-axis", type=float, default=None)
parser.add_argument("-m", "--markers", dest='markers', required=False, help="Number of markers", type=int, default=30)
parser.add_argument("-c", "--collapse", dest='collapse', required=False, help="Shorten labels", action="store_true", default=False)
parser.add_argument("-e", "--extend-to", dest='extend_to', required=False, help="Extend labels when shortening to", type=str, default=None)
parser.add_argument("-s", "--skip-first", dest='skip_first', required=False, help="Skip first entry", action="store_true", default=True)
parser.add_argument("-g", "--single-legend", dest='single_legend', required=False, help="Use single legend", action="store_true", default=False)
parser.add_argument("-b", "--labels", dest='labels', required=False, nargs='+', help="Customized labels", default=None)
parser.add_argument("-r", "--delimiter", dest='delimiter', required=False, help="Input file delimiter", default=None)
parser.add_argument("--normalize-xaxis", dest='normalize_xaxis', required=False, help="Normalize x-axis", action="store_true", default=False)
parser.add_argument("--normalize-yaxis", dest='normalize_yaxis', required=False, help="Normalize y-axis", action="store_true", default=False)

args = parser.parse_args()

# Get all files to process
files_list = library.get_solutions(args.files)

if not files_list:
    raise Exception("The files list is empty, nothing to plot")

n_files = len(files_list)

colors = library.get_hex_colors(n_files)

# Always get available fields since we can use formulas
available_fields = None
for f in files_list:
    cdata = np.genfromtxt(f, dtype=None, names=True, delimiter=args.delimiter)
    if available_fields is None:
        available_fields = set(cdata.dtype.names)
    else:
        available_fields = available_fields.intersection(set(cdata.dtype.names))

available_fields = list(sorted(list(available_fields)))

if not args.yaxes:
    print("You did not specify any y-axes field to plot. These are the fields shared between all the provided files:")
        
    for name in list(available_fields):
        print(" -- {}".format(name))
    
    print("Rerun the script specifying at least one of them")
    exit()

print("")
print("         x-axis: {}".format(args.xaxis))
print(" fields to plot: " + ", ".join(args.yaxes))
print("number of files: {}".format(n_files))

n_fields = len(args.yaxes)
fig, ax = create_axes(n_fields)

# Generate labels
if args.collapse:
    labels = library.generate_short_labels(files_list, args.extend_to)
else:
    labels = files_list.copy()

# If we have custom labels
if args.labels:
    for i in range(min(len(labels), len(args.labels))):
        labels[i] = args.labels[i]

# Markers
markers = ["s", "D", "o", "x", "P", "*", "v"]
markers = list(islice(cycle(markers), n_files))

# We will start from the longest field
available_fields.sort(key=len, reverse=True)

for i in range(n_fields):
    field = args.yaxes[i]
    a = ax[i]

    x_lims = [sys.float_info.max, -sys.float_info.max]

    token_prefix = '%token_{}'

    for f, lbl, clr, mrk in zip(files_list, labels, colors, markers):
        cdata = np.genfromtxt(f, dtype=None, names=True, delimiter=args.delimiter)

        tokens = []
        token_numberer = 0
        
        mask = [True] * len(cdata[args.xaxis])
        if args.limits:
            mask = (args.limits[0] <= cdata[args.xaxis]) & (cdata[args.xaxis] <= args.limits[1])    
        elif args.skip_first:
            mask[0] = False
        
        x = cdata[args.xaxis][mask]
        if args.normalize_xaxis and max(x):
            x /= max(x)

        # Maybe it is a formula - we will try to interpret it, its safety is implied
        y_label = field
        if not field in available_fields:
            formula = field
            for possible_field in available_fields:
                if possible_field in formula: 
                    token = token_prefix.format(token_numberer)
                    formula = formula.replace(possible_field, token)
                    tokens.append((token, "cdata['{}'][mask]".format(possible_field)))
                    token_numberer += 1

            for token_data in reversed(tokens):
                formula = formula.replace(token_data[0], token_data[1])

            # Try to split expression for custom y label
            expressions = formula.split('=')
            if len(expressions) > 2:
                raise Exception("Invalid expression provided, contains too many '=' signs")
            elif len(expressions) == 2:
                y_label = expressions[0].strip()
                formula = expressions[1].strip()

            y = eval(formula)
        else:
            y = cdata[field][mask]

        x_lims[0] = min(x_lims[0], x[0])
        x_lims[1] = max(x_lims[1], x[-1])

        if args.normalize_yaxis and max(y):
            y /= max(y)

        a.plot(x, y, color=clr, linestyle='-', linewidth=2, label=lbl, marker=mrk, markevery=n_files)

    a.grid(True)
    a.set_xlabel(args.xaxis)
    a.set_ylabel(y_label)
    a.set_title(field)
    a.set_xlim(x_lims)

    if not args.single_legend:
        a.legend()
    elif i == n_fields - 1:
        a_handles, a_labels = a.get_legend_handles_labels()
        fig.legend(a_handles, a_labels, loc='upper center')

plt.show()
