import numpy as np
import matplotlib.pyplot as plt
import argparse
import colorsys
import glob
import re
import sys
import collections.abc

def get_hex_colors(N):
    hsv_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in hsv_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-f", "--files", dest="files", nargs='+', required=True, help="Source filenames, can be defined as masks")
parser.add_argument("-d", "--directions", dest="directions", nargs='+',
    required=False, default="all", help="Directions", choices=['x', 'y', 'z', 'vol', 'all'])
parser.add_argument("-t", "--limit_t", dest='limit_t', required=False, help="Max time limit", type=float, default=sys.float_info.max)
parser.add_argument("-m", "--markers", dest='markers', required=False, help="Number of markers", type=int, default=30)
parser.add_argument("-c", "--collapse", dest='collapse', required=False, help="Number of markers", action="store_true", default=False)

args = parser.parse_args()

if not isinstance(args.directions, collections.abc.Sequence):
    args.directions = [args.directions]

# Get all files to process
files_list = []
for fm in args.files:
    current_files_list = glob.glob(fm, recursive=True)
    current_files_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    files_list += current_files_list

files_list = sorted(list(set(files_list)))

print("The complete list of files to process:")
for f in files_list:
    print(f)

n_files = len(files_list)
colors = get_hex_colors(n_files)

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
    min_len = min([len(f) for f in files_list])

    s_start = 0
    s_end = 0
    
    for i in range(min_len):
        string_equal = True
        for f in files_list:
            if not(f[i] == files_list[0][i]):
                string_equal = False
                break

        if string_equal:
            s_start += 1
        else:
            break

    for i in range(min_len):
        string_equal = True
        for f in files_list:
            if not(f[-1-i] == files_list[0][-1-i]):
                string_equal = False
                break

        if string_equal:
            s_end += 1
        else:
            break

    labels = [f[s_start:-s_end] for f in files_list]

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

            mask = fdata["time"] < args.limit_t

            if args.markers > 0:
                n_every = round(len(fdata["time"][mask]) / args.markers)
                n_every = max(1, n_every)

                m_type = markers[i % len(markers)]
            else:
                n_every = 1
                m_type = None

            axes[0].plot(fdata["time"][mask], fdata[qty_name][mask], label=" ".join([lbl, csv_header[i]]), 
                marker=m_type, color=clr, alpha=alpha, markevery=n_every)
            axes[1].plot(fdata["time"][mask], ref_qty[mask], label=" ".join([lbl, csv_header[i+n_qtys]]),
                marker=m_type, color=clr, alpha=alpha, markevery=n_every)

            alpha -= 0.2

for i in range(2):
    axes[i].grid(True)
    axes[i].legend()

plt.show()
