import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import re
import os
import numpy.ma as ma
from scipy.stats import norm
from matplotlib.ticker import PercentFormatter
import library

parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-m", "--mask", type=str, help="File mask", required=True)
parser.add_argument("-f", "--file", type=str, help="Solution file", required=False, default="solution.log")
parser.add_argument("-p", "--path", type=str, help="Common path", required=False, default=None)
parser.add_argument("-b", "--bins", type=int, help="Number of bins", default=10)
parser.add_argument("-t", "--start", type=int, help="Step to start with", default=0)
parser.add_argument("-e", "--end", type=int, help="Step to end with", default=0)
parser.add_argument("-d", "--delete", action='store_true', help="Delete the largest entity", required=False, default=False)
parser.add_argument("-o", "--output", type=str, help="Destination folder to save data", default=None, required=False)
parser.add_argument("-y", "--density", action='store_true', help="Plot probability density", required=False, default=False)
parser.add_argument("-g", "--common-range", action='store_true', help="Use common range for bins", required=False, default=False)
parser.add_argument("-n", "--common-bins", action='store_true', help="Use common bins", required=False, default=False)
parser.add_argument("-c", "--decimals", dest="decimals", type=int, help="Number of decimals in percents", default=None)

group = parser.add_mutually_exclusive_group()
group.add_argument("-r", "--radius", action='store_true', default=True)
group.add_argument("-u", "--measure", action='store_true')
group.add_argument("-s", "--save", action='store_true')

args = parser.parse_args()

qty_name = "measure" if args.measure else "radius"

def plot_histogram(quantity, ax, time, n_bins = 10, val_range = None, save_file = None):
    # Fit a normal distribution to the data:
    mu, std = norm.fit(quantity)

    q_min = min(quantity)
    q_max = max(quantity)

    counts, bins = np.histogram(quantity, bins=n_bins, range=val_range)
    counts_plt = counts/sum(counts) if not args.density else counts
    ax.hist(bins[:-1], bins, weights=counts_plt, density=args.density, alpha=0.6, color='g', edgecolor='black', linewidth=1.2)

    if args.density:
        q_arr = np.linspace(q_min, q_max, 100)
        p_arr = norm.pdf(q_arr, mu, std)
        ax.plot(q_arr, p_arr, 'k', linewidth=2)

    ax.set_title("Time t = {}".format(time))
    ax.set_xlabel(qty_name)
    ax.set_ylabel("fraction")
    ax.grid(True)

    if not args.density:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=args.decimals))

    if save_file:
        n_particles = len(quantity)
        counts_save = np.append(counts, 0)
        counts_save = counts_save/n_particles*100
        np.savetxt(save_file, np.column_stack((bins, counts_save)))

# Deal with path names
file_solution = args.file
file_distribution = args.mask
if args.path is not None:
    file_solution = os.path.join(args.path, file_solution)
    file_distribution = os.path.join(args.path, file_distribution)

# Get files according to the mask and sort them by number
files_list = glob.glob(file_distribution)
files_list.sort(key=lambda f: int(re.sub('\D', '', f)))

n_rows = len(files_list)

if not n_rows:
    raise Exception("The data files set for analysis is empty, check your path and mask")

ax_init = plt.subplot(221)
ax_final = plt.subplot(223)
ax_mu_std = plt.subplot(222)
ax_total = plt.subplot(224)


fdata = np.genfromtxt(file_solution, dtype=None, names=True)

curves = []

curves.append({
    'means': [],
    'stds': [],
    'total': [],
    'time': [],
    'data': []
})

if args.save:
    if args.output:
        pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
        output_folder = args.output
    else:
        output_folder = os.path.dirname(file_solution)
else:
    output_folder = ""

# Bins ranges
bins_range = None
if args.common_bins or args.common_range:
    bins_range = (1e16, -1e16)

# Initial and final histograms
hist_init = None
hist_final = None

for idx, log_file in enumerate(files_list):

    t = fdata["time"][idx]

    if t < args.start:
        continue

    print("Parsing file {} ({}/{})".format(log_file, idx + 1, n_rows))

    qdata = np.genfromtxt(log_file, dtype=None, names=True)

    data_to_plot = qdata[qty_name]
    if args.delete:
        data_to_plot = np.delete(data_to_plot, data_to_plot.argmax())

    if data_to_plot.size == 0:
        if len(curves[-1]['means']) > 0:
            curves[-1]['means'].append(curves[-1]['means'][-1])
            curves[-1]['stds'].append(curves[-1]['stds'][-1])
            curves[-1]['total'].append(curves[-1]['total'][-1])
            curves[-1]['time'].append(curves[-1]['time'][-1])

            curves.append({
                'means': [],
                'stds': [],
                'total': [],
                'time': [],
                'data': []
            })

        continue

    if args.common_bins or args.common_range:
        bins_range = (min(bins_range[0], data_to_plot.min()), max(bins_range[1], data_to_plot.max()))

    if t >= args.start and not hist_init:
        hist_init = {'data': data_to_plot, 'time': fdata["time"][idx]}
    elif ((args.end != 0 and t >= args.end) or idx == n_rows - 1) and not hist_final:
        hist_final = {'data': data_to_plot, 'time': fdata["time"][idx]}

    mu, std = norm.fit(data_to_plot)

    curves[-1]['means'].append(mu)
    curves[-1]['stds'].append(std)
    curves[-1]['total'].append(qdata.size)
    curves[-1]['time'].append(fdata["time"][idx])

    if args.save:
        curves[-1]['data'].append(data_to_plot)

    if hist_final:
        break

# Plot histograms if we have the data for them
bins_effective = bins_range if args.common_bins else None
if hist_init:
    save_file = os.path.join(output_folder, "{}_distribution_histogram_t_{}.csv".format(qty_name, hist_init['time'])) if args.save else None
    plot_histogram(hist_init['data'], ax_init, hist_init['time'], args.bins, bins_effective, save_file)
if hist_final:
    save_file = os.path.join(output_folder, "{}_distribution_histogram_t_{}.csv".format(qty_name, hist_final['time'])) if args.save else None
    plot_histogram(hist_final['data'], ax_final, hist_final['time'], args.bins, bins_effective, save_file)

if args.common_range:
    ax_init.set_xlim(bins_range)
    ax_final.set_xlim(bins_range)

# Headers for CSV to save if needed
csv_header = ["time", "average", "stds", "total"]
csv_format = ["%g"] * (len(csv_header))
csv_format = " ".join(csv_format)
csv_header = " ".join(csv_header)

data_csv_bins = []

for idx, curves_set in enumerate(curves):
    means = np.array(curves_set['means'])
    stds = np.array(curves_set['stds'])
    time = curves_set['time']
    total = curves_set['total']

    library.plot_distribution_history(ax_mu_std, ax_total, time, means, stds, total)

    if idx < len(curves) - 1:
        ax_mu_std.axvspan(time[-1], curves[idx + 1]['time'][0], color='blue', alpha=0.1)
        ax_total.axvspan(time[-1], curves[idx + 1]['time'][0], color='red', alpha=0.1)

    if args.save:
        csv_data = np.column_stack((time, means, stds, total))

        csv_suffix = "" if len(curves) == 1 else "_{}".format(idx)
        file_path = os.path.join(output_folder, "{}_distribution_history{}.csv".format(qty_name, csv_suffix))
        np.savetxt(file_path, csv_data, delimiter=' ', header=csv_header, fmt=csv_format, comments='')

        data_set = curves_set['data']
        for t, data in zip(time, data_set):
            counts, bins = np.histogram(data, args.bins, range=bins_effective)
            counts = counts/sum(counts)

            data_csv_bins.append({
                't': t,
                'bins': bins,
                'counts': counts,
            })

if args.save:
    file_path = os.path.join(output_folder, "{}_distribution_data.dat".format(qty_name))

    def format(value):
        return "%.5f" % value

    with open(file_path, 'w') as f:
        for entry in data_csv_bins:
            f.write(f"{entry['t']}\n")
            [f.write(f"{format(val)} ") for val in entry['bins']]
            f.write(f"\n")
            [f.write(f"{format(val)} ") for val in entry['counts']]
            f.write(f"\n")

library.format_distribution_plot(ax_mu_std, ax_total, qty_name)

plt.tight_layout()

plt.suptitle("Solution: {}  |  data: {}".format(file_solution, file_distribution))

plt.show()