import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import glob
import re
import os
import numpy.ma as ma
import pathlib
from scipy.stats import norm
from matplotlib.patches import Polygon

def format_plot(ax):
    ax.spines['bottom'].set_color(color_axes)
    ax.spines['top'].set_color(color_axes) 
    ax.spines['right'].set_color(color_axes)
    ax.spines['left'].set_color(color_axes)
    ax.xaxis.label.set_color(color_axes)
    ax.yaxis.label.set_color(color_axes)
    #ax.yaxis.set_label_coords(-0.08,0.5)
    #ax.xaxis.set_label_coords(0.5,-0.1)
    ax.tick_params(axis='x', colors=color_axes, labelsize=label_size)
    ax.tick_params(axis='y', colors=color_axes, labelsize=label_size)
    ax.set_facecolor(color_background)

def plot_step(cdata, x_min, x_max, y_min, y_max, x_size, y_size, step = None, show = True):

    #fig = plt.figure(figsize=(10, 11.25), dpi=96, facecolor=color_background)
    #ax = plt.axes()
    fig, ax = plt.subplots(1, 1, dpi=96, facecolor=color_background)
    fig.set_figheight(y_size)
    fig.set_figwidth(x_size)
    #fig.patch.set_facecolor(color_background)

    if step is not(None):
        local_limit = step + 1
        pngName = "step_{:04d}.png".format(step)
    else:
        local_limit = data_limit
        pngName = "step_none.png"

    # data
    x = cdata[args.xaxis][0:local_limit]
    y = cdata[args.yaxis][0:local_limit]

    # Plot
    gradient_fill(x, y, fill_color=color_fill, ax=ax, ymax_custom=y_max, color=color_line)
    format_plot(ax)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    xlabel = args.xaxis if not args.xlabel else args.xlabel
    ylabel = args.yaxis if not args.ylabel else args.ylabel

    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)

    fig.tight_layout(pad=1.5)
    #plt.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.9)
    #box = ax.get_position()
    #box.y0 = box.y0 + 0.03
    #box.y1 = box.y1 + 0.03
    #ax.set_position(box)

    plt.savefig(os.path.join(output_folder, pngName))
    
    if show:
        plt.show()
        exit()

    plt.clf()
    plt.close()
    

def gradient_fill(x, y, fill_color=None, ax=None, ymax_custom=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    if ymax_custom is not(None):
        ymax = ymax_custom
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    #ax.autoscale(True)
    return line, im

parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-f", "--file", type=str, help="Solution file", required=True)
parser.add_argument("-x", "--xaxis", dest="xaxis", required=False, help="x-axis variable", default="time")
parser.add_argument("-y", "--yaxis", dest="yaxis", required=False, help="y-axis variable")
parser.add_argument("--xlabel", dest="xlabel", required=False, help="x-axis label")
parser.add_argument("--ylabel", dest="ylabel", required=False, help="y-axis label")
parser.add_argument("--xsize", dest="xsize", required=False, help="x size of the figure", default=5., type=float)
parser.add_argument("--ysize", dest="ysize", required=False, help="y size of the figure", default=5., type=float)
parser.add_argument("-s", "--span", dest="span", required=False, help="y-axis span", default=0.05)
parser.add_argument("-o", "--output", type=str, help="Destination folder to save data", default=None, required=False)
parser.add_argument("-r", "--delimiter", dest='delimiter', required=False, help="Input file delimiter", default=None)

# Plot settings
data_limit = 9999
color_axes = '#bbbbbb'
color_background = '#52576e'
color_line = 'white'
color_fill = '#1f77b4'
font_size = 16
label_size = 14

args = parser.parse_args()

if not args.yaxis:
    print("You did not specify any y-axes field to plot. These are the fields you can choose from:")

    cdata = np.genfromtxt(args.file, dtype=None, names=True, delimiter=args.delimiter)
    available_fields = set(cdata.dtype.names)
    sorted_available_fields = sorted(list(available_fields))
        
    for name in list(sorted_available_fields):
        print(" -- {}".format(name))
    
    print("Rerun the script specifying at least one of them")
    exit()

# Save location
if args.output:
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    output_folder = args.output
else:
    output_folder = os.path.dirname(file_solution)


cdata = np.genfromtxt(args.file, delimiter=args.delimiter, dtype=None, names=True)

x_min = np.min(cdata[args.xaxis])
x_max = np.max(cdata[args.xaxis])
y_min = np.min(cdata[args.yaxis])
y_max = np.max(cdata[args.yaxis])

if args.span:
    y_range = y_max - y_min
    y_max += args.span * y_range
    y_min -= args.span * y_range

step_start = 0

for istep in range(step_start, cdata.size):
    print("Plotting step {}".format(istep))
    plot_step(cdata, x_min, x_max, y_min, y_max, args.xsize, args.ysize, istep, False)
