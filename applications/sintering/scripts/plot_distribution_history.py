import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import library
from pathlib import Path

parser = argparse.ArgumentParser(description='Plot evolution of the average particle or pore size')
parser.add_argument("-p", "--path", type=str, help="Common path", required=False, default=None)
parser.add_argument("-q", "--quantity", type=str, help="Quantity name", required=False, default=None)
parser.add_argument("-t", "--start", type=str, help="File start", required=False, default=None)
parser.add_argument("-e", "--end", type=str, help="File end", required=False, default=None)
parser.add_argument("-s", "--history", type=str, help="File history", required=True)
parser.add_argument("-g", "--common-range", action='store_true', help="Use common range for bins", required=False, default=False)

args = parser.parse_args()

ax_init = plt.subplot(221)
ax_final = plt.subplot(223)
ax_mu_std = plt.subplot(222)
ax_total = plt.subplot(224)

path_history = os.path.join(args.path, args.history) if args.path else args.history

if not args.start or not args.end:
    base_name = os.path.splitext(args.history)[0]
    name_parts = base_name.split("_")
    name_parts[-1] = "histogram"
    mask = "_".join(name_parts)
    mask += "*"

    if args.path:
        mask = os.path.join(args.path, mask)

    files = glob.glob(mask, recursive=False)

    if not args.start and len(files) > 0:
        path_init = files[0]
        files.pop(0)
    else:
        raise Exception('No histogram for the initial configuration was detected, provide it or check your data folder')

    if not args.end and len(files) > 0:
        path_final = files[-1]
    else:
        raise Exception('No histogram for the final configuration was detected, provide it or check your data folder')

if args.start:
    path_init = os.path.join(args.path, args.start) if args.path else args.start

if args.end:
    path_final = os.path.join(args.path, args.end) if args.path else args.end

data_init = np.genfromtxt(path_init, dtype=None)
data_final = np.genfromtxt(path_final, dtype=None)
fdata = np.genfromtxt(path_history, dtype=None, names=True)

def plot_histogram(ax, bins, counts, time, qty_name):
    ax.hist(bins[:-1], bins, weights=counts[:-1], density=True, alpha=0.6, color='g', edgecolor='black', linewidth=1.2)

    ax.set_title("Time t = {}".format(time))
    ax.set_xlabel(qty_name)
    ax.set_ylabel("content ratio")
    ax.grid(True)

meta_init = Path(path_init).stem.split("_")
meta_final = Path(path_final).stem.split("_")
meta_history = Path(path_history).stem.split("_")

plot_histogram(ax_init, data_init[:,0], data_init[:,1], meta_init[-1], args.quantity if args.quantity else meta_init[0])
plot_histogram(ax_final, data_final[:,0], data_final[:,1], meta_final[-1], args.quantity if args.quantity else meta_final[0])

if args.common_range:
    x_min = min(data_init[:,0].min(), data_final[:,0].min())
    x_max = max(data_init[:,0].max(), data_final[:,0].max())

    ax_init.set_xlim((x_min, x_max))
    ax_final.set_xlim((x_min, x_max))

library.plot_distribution_history(ax_mu_std, ax_total, fdata["time"], fdata["average"], fdata["stds"], fdata["total"])

library.format_distribution_plot(ax_mu_std, ax_total, args.quantity if args.quantity else meta_history[0])

plt.tight_layout()

plt.show()