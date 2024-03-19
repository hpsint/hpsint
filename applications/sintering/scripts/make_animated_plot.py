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
from matplotlib.patches import Polygon

def plot_step(xdata, ydata, x_min, x_max, y_min, y_max, step = None, show = True):

    fig, ax = library.animation_init_plot(args.format_color_background, args.xsize, args.ysize)

    if step is not(None):
        local_limit = step + 1
        png_name = "step_{:04d}.png".format(step)
    else:
        local_limit = 999999
        png_name = "step_none.png"

    # data
    x = xdata[0:local_limit]
    y = ydata[0:local_limit]

    # Plot
    gradient_fill(x, y, fill_color=args.format_color_fill, ax=ax, ymin_custom=y_min, ymax_custom=y_max, color=args.format_color_line, linewidth=args.format_line_width)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    xlabel = args.xaxis if not args.xlabel else args.xlabel
    ylabel = args.yaxis if not args.ylabel else args.ylabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    library.animation_format_plot(ax, args.format_color_axes, args.format_color_background, args.format_label_size, args.format_font_size)

    fig.tight_layout(pad=1.5)

    plt.savefig(os.path.join(output_folder, png_name))
    
    if show:
        plt.show()
        exit()

    plt.clf()
    plt.close()
    

def gradient_fill(x, y, fill_color=None, ax=None, ymin_custom=None, ymax_custom=None, **kwargs):
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
    if ymin_custom is not(None):
        ymin = ymin_custom
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
parser.add_argument("--xsize", dest="xsize", required=False, help="x size of the figure", default=19.2, type=float)
parser.add_argument("--ysize", dest="ysize", required=False, help="y size of the figure", default=10.8, type=float)
parser.add_argument("-s", "--span", dest="span", required=False, help="y-axis span", default=0.05)
parser.add_argument("-o", "--output", type=str, help="Destination folder to save data", default=None, required=False)
parser.add_argument("-r", "--delimiter", dest='delimiter', required=False, help="Input file delimiter", default=None)
parser.add_argument("-t", "--start", dest="start", type=int, help="Step to start with", default=None)
parser.add_argument("-e", "--end", dest="end", type=int, help="Step to end with", default=None)

# Plot settings
parser.add_argument("--format-color-axes", dest="format_color_axes", required=False, help="Format color axes", default='#bbbbbb', type=str)
parser.add_argument("--format-color-background", dest="format_color_background", required=False, help="Format color background", default='#52576e', type=str)
parser.add_argument("--format-color-line", dest="format_color_line", required=False, help="Format color line", default='white', type=str)
parser.add_argument("--format-color-fill", dest="format_color_fill", required=False, help="Format color fill", default='#1f77b4', type=str)
parser.add_argument("--format-font-size", dest="format_font_size", required=False, help="Format font size", default=20, type=int)
parser.add_argument("--format-label-size", dest="format_label_size", required=False, help="Format label size", default=14, type=int)
parser.add_argument("--format-line-width", dest="format_line_width", required=False, help="Format line width", default=2.0, type=float)

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
    output_folder = os.path.dirname(args.file)


cdata = np.genfromtxt(args.file, delimiter=args.delimiter, dtype=None, names=True)

xdata = cdata[args.xaxis]
ydata = cdata[args.yaxis]

xfilter = np.arange(len(xdata))
if args.start is not None:
    xfilter = xfilter[np.in1d(xfilter, np.where((xdata >= args.start)))]
if args.end is not None:
    xfilter = xfilter[np.in1d(xfilter, np.where((xdata <= args.end)))]

xdata = xdata[xfilter]
ydata = ydata[xfilter]

x_min = np.min(xdata)
x_max = np.max(xdata)
y_min = np.min(ydata)
y_max = np.max(ydata)

if args.span:
    y_range = y_max - y_min
    y_max += args.span * y_range
    y_min -= args.span * y_range

n = len(xdata)

if n <= 1:
    raise Exception("There is too few data (len(xdata) = {}) to plot - nothing to animate".format(n))

for istep in range(n):
    print("Rendering step {}/{}".format(istep+1, n))
    plot_step(xdata, ydata, x_min, x_max, y_min, y_max, istep, False)
