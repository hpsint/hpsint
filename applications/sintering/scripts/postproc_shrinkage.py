import numpy as np
import matplotlib.pyplot as plt
import argparse
import colorsys

def get_hex_colors(N):
    hsv_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in hsv_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-f", "--files", dest="files", nargs='+', required=True, help="Source filename")

args = parser.parse_args()

n_files = len(args.files)
colors = get_hex_colors(n_files)

csv_header = ["dim_x", "dim_y", "dim_z", "volume", "shrinkage_x", "shrinkage_y", "shrinkage_z", "densification"]
n_qtys = 4
markers = ["s", "D", "o", "x"]

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.suptitle('Shrinkage and densification')

for f, clr in zip(args.files, colors):

    fdata = np.genfromtxt(f, dtype=None, names=True)

    alpha = 1

    for i in range(n_qtys):
        if csv_header[i] in fdata.dtype.names:
            qty_name = csv_header[i]

            ref0 = fdata[qty_name][0]
            ref_qty = (ref0 - fdata[qty_name]) / ref0

            m_type = markers[i % len(markers)]
            axes[0].plot(fdata["time"], fdata[qty_name], label=" ".join([f, csv_header[i]]), marker=m_type, color=clr, alpha=alpha)
            axes[1].plot(fdata["time"], ref_qty, label=" ".join([f, csv_header[i+n_qtys]]), marker=m_type, color=clr, alpha=alpha)

            alpha -= 0.2

for i in range(2):
    axes[i].grid(True)
    axes[i].legend()

plt.show()
