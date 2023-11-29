import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import glob
import re
import os
import numpy.ma as ma
import pathlib
import library
from scipy.stats import norm
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Polygon

def plot_step(bins, counts, y_max, t = 0, step = None, show = False):

    fig, ax = library.animation_init_plot(args.format_color_background, args.xsize, args.ysize)

    if step is not(None):
        pngName = "step_{:04d}.png".format(step)
    else:
        pngName = "step_none.png"

    # Plot
    ax.hist(bins[:-1], bins, weights=counts, alpha=0.6, color=args.format_color_fill, edgecolor=args.format_color_line, linewidth=args.format_line_width)
    ax.set_ylim(0, y_max)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(args.label)
    ax.set_ylabel("percentage of particles")

    library.animation_format_plot(ax, args.format_color_axes, args.format_color_background, args.format_label_size, args.format_font_size)

    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=args.decimals))

    fig.tight_layout(pad=1.5)

    plt.savefig(os.path.join(output_folder, pngName))
    
    if show:
        plt.show()
        exit()

    plt.clf()
    plt.close()

parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-f", "--file", type=str, help="Solution file", required=True)
parser.add_argument("-l", "--label", type=str, required=False, help="Quantity name", default="radius")
parser.add_argument("--xsize", dest="xsize", required=False, help="x size of the figure", default=19.2, type=float)
parser.add_argument("--ysize", dest="ysize", required=False, help="y size of the figure", default=10.8, type=float)
parser.add_argument("-s", "--span", dest="span", required=False, help="y-axis span", default=0.05)
parser.add_argument("-o", "--output", type=str, help="Destination folder to save data", default=None, required=False)
parser.add_argument("-t", "--start", dest="start", type=int, help="Step to start with", default=None)
parser.add_argument("-e", "--end", dest="end", type=int, help="Step to end with", default=None)
parser.add_argument("-c", "--decimals", dest="decimals", type=int, help="Number of decimals in percents", default=None)

# Plot settings
parser.add_argument("--format-color-axes", dest="format_color_axes", required=False, help="Format color axes", default='#bbbbbb', type=str)
parser.add_argument("--format-color-background", dest="format_color_background", required=False, help="Format color background", default='#52576e', type=str)
parser.add_argument("--format-color-line", dest="format_color_line", required=False, help="Format color line", default='#bbbbbb', type=str)
parser.add_argument("--format-color-fill", dest="format_color_fill", required=False, help="Format color fill", default='#1f77b4', type=str)
parser.add_argument("--format-font-size", dest="format_font_size", required=False, help="Format font size", default=20, type=int)
parser.add_argument("--format-label-size", dest="format_label_size", required=False, help="Format label size", default=14, type=int)
parser.add_argument("--format-line-width", dest="format_line_width", required=False, help="Format line width", default=1.5, type=float)

args = parser.parse_args()

# Save location
if args.output:
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    output_folder = args.output
else:
    output_folder = os.path.dirname(args.file)

with open(args.file) as file:
    lines = [line.rstrip() for line in file]

n_lines = len(lines)
if n_lines % 3 != 0:
    raise Exception('The input file contains inconsistent data')

y_max = None
step_counter = 0
n_steps = int(n_lines / 3)

for i in range(n_steps):
    t = float(lines[3*i])

    if (not args.start or t >= args.start) and (not args.end or t <= args.end):
        print("Rendering step {}/{}".format(i+1, n_steps))
        step_counter += 1
    else:
        print(" Skipping step {}/{}".format(i+1, n_steps))
        continue

    bins = [float(val) for val in lines[3*i+1].split()]
    counts = [float(val) for val in lines[3*i+2].split()]

    if i == 0:
        y_max = max(counts) * (1. + args.span)

    plot_step(bins, counts, y_max, t, step_counter)
