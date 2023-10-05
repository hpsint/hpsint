import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import re
import os
import numpy.ma as ma
from scipy.stats import norm

parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-m", "--mask", type=str, help="File mask", required=True)
parser.add_argument("-f", "--file", type=str, help="Solution file", required=False, default="solution.log")
parser.add_argument("-p", "--path", type=str, help="Common path", required=False, default=None)
parser.add_argument("-b", "--bins", type=int, help="Number of bins", default=10)
parser.add_argument("-t", "--start", type=int, help="Step to start with", default=0)
parser.add_argument("-e", "--end", type=int, help="Step to end with", default=0)
parser.add_argument("-d", "--delete", action='store_true', help="Delete the largest entity", required=False, default=False)
parser.add_argument("-o", "--output", type=str, help="Destination folder to save data", default=None, required=False)

group = parser.add_mutually_exclusive_group()
group.add_argument("-r", "--radius", action='store_true', default=True)
group.add_argument("-u", "--measure", action='store_true')
group.add_argument("-s", "--save", action='store_true')

args = parser.parse_args()

qty_name = "measure" if args.measure else "radius"

def plot_histogram(quantity, ax, time, n_bins = 10, save_file = None):
    # Fit a normal distribution to the data:
    mu, std = norm.fit(quantity)

    q_min = min(quantity)
    q_max = max(quantity)

    counts, bins = np.histogram(quantity, bins=n_bins)
    ax.hist(bins[:-1], bins, weights=counts, density=True, alpha=0.6, color='g', edgecolor='black', linewidth=1.2)

    q_arr = np.linspace(q_min, q_max, 100)
    p_arr = norm.pdf(q_arr, mu, std)
    ax.plot(q_arr, p_arr, 'k', linewidth=2)
    ax.set_title("Time t = {}".format(time))
    ax.set_xlabel(qty_name)
    ax.grid(True)

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
    'time': []
})

# Flags if we saved the first and last distributions
done_start = False
done_end = False

if args.save:
    if args.output:
        pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
        output_folder = args.output
    else:
        output_folder = os.path.dirname(file_solution)
else:
    output_folder = ""

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
                'time': []
            })

        continue

    save_file = os.path.join(output_folder, "{}_distribution_hist_t_{}.csv".format(qty_name, t)) if args.save else None

    if t >= args.start and not done_start:
        plot_histogram(data_to_plot, ax_init, fdata["time"][idx], args.bins, save_file)
        done_start = True
    elif ((args.end != 0 and t >= args.end) or idx == n_rows - 1) and not done_end:
        plot_histogram(data_to_plot, ax_final, fdata["time"][idx], args.bins, save_file)
        done_end = True

    mu, std = norm.fit(data_to_plot)

    curves[-1]['means'].append(mu)
    curves[-1]['stds'].append(std)
    curves[-1]['total'].append(qdata.size)
    curves[-1]['time'].append(fdata["time"][idx])

    if done_end:
        break

# Headers for CSV to save if needed
csv_header = ["time", "average", "stds", "total"]
csv_format = ["%g"] * (len(csv_header))
csv_format = " ".join(csv_format)
csv_header = " ".join(csv_header)

for idx, curves_set in enumerate(curves):
    means = np.array(curves_set['means'])
    stds = np.array(curves_set['stds'])
    time = curves_set['time']
    total = curves_set['total']

    ax_mu_std.plot(time, means, linewidth=2, color='blue')
    ax_mu_std.fill_between(time, means-stds, means+stds, alpha=0.4, color='#888888')
    ax_mu_std.fill_between(time, means-2*stds, means+2*stds, alpha=0.4, color='#cccccc')

    ax_total.plot(time, total, linewidth=2, color='red')

    if idx < len(curves) - 1:
        ax_mu_std.axvspan(time[-1], curves[idx + 1]['time'][0], color='blue', alpha=0.1)
        ax_total.axvspan(time[-1], curves[idx + 1]['time'][0], color='red', alpha=0.1)

    if args.save:
        csv_data = np.column_stack((time, means, stds, total))

        csv_suffix = "" if len(curves) == 1 else "_{}".format(idx)
        file_path = os.path.join(output_folder, "{}_distribution_history{}.csv".format(qty_name, csv_suffix))
        np.savetxt(file_path, csv_data, delimiter=' ', header=csv_header, fmt=csv_format, comments='')

ax_mu_std.grid(True)
ax_mu_std.set_title("Average value")
ax_mu_std.set_xlabel("time")
ax_mu_std.set_ylabel(qty_name)
#ax_mu_std.set_ylim([0.9*np.min(means-3*stds), 1.1*np.max(means+3*stds)])

ax_total.grid(True)
ax_total.set_title("Number of entities")
ax_total.set_xlabel("time")
ax_total.set_ylabel("#")

plt.tight_layout()

plt.suptitle("Solution: {}  |  data: {}".format(file_solution, file_distribution))

plt.show()