import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import library
from pathlib import Path

parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-p", "--path", type=str, help="Common path", required=False, default=None)
parser.add_argument("-q", "--quantity", type=str, help="Quantity name", required=False, default=None)
parser.add_argument("-t", "--start", type=str, help="File start", required=True)
parser.add_argument("-e", "--end", type=str, help="File end", required=True)
parser.add_argument("-s", "--history", type=str, help="File history", required=True)

args = parser.parse_args()

ax_init = plt.subplot(221)
ax_final = plt.subplot(223)
ax_mu_std = plt.subplot(222)
ax_total = plt.subplot(224)

path_init = os.path.join(args.path, args.start) if args.path else args.start
path_final = os.path.join(args.path, args.end) if args.path else args.end
path_history = os.path.join(args.path, args.history) if args.path else args.history

data_init = np.genfromtxt(path_init, dtype=None)
data_final = np.genfromtxt(path_final, dtype=None)
fdata = np.genfromtxt(path_history, dtype=None, names=True)

def plot_histogram(ax, bins, counts, time, qty_name):
    ax.hist(bins[:-1], bins, weights=counts[:-1], density=True, alpha=0.6, color='g', edgecolor='black', linewidth=1.2)

    ax.set_title("Time t = {}".format(time))
    ax.set_xlabel(qty_name)
    ax.grid(True)

meta_init = Path(path_init).stem.split("_")
meta_final = Path(path_final).stem.split("_")
meta_history = Path(path_history).stem.split("_")

plot_histogram(ax_init, data_init[:,0], data_init[:,1], meta_init[-1], args.quantity if args.quantity else meta_init[0])
plot_histogram(ax_final, data_final[:,0], data_final[:,1], meta_final[-1], args.quantity if args.quantity else meta_final[0])

library.plot_distribution_history(ax_mu_std, ax_total, fdata["time"], fdata["average"], fdata["stds"], fdata["total"])

library.format_distribution_plot(ax_mu_std, ax_total, args.quantity if args.quantity else meta_history[0])

plt.tight_layout()

plt.show()