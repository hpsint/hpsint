import os
import numpy as np
import argparse
import library
from scipy.spatial import ConvexHull

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--files", dest="files", required=True, nargs='+', help="Files to process, can be mask")
parser.add_argument("-e", "--headers", dest='headers', required=False, nargs='+', help="Headers")
parser.add_argument("-o", "--output", dest='output', required=False, help="Destination path to output tex file", type=str, default=None)
parser.add_argument("-x", "--x-shift", dest='xshift', required=False, help="x-axis shift", type=float, default=0.0)
parser.add_argument("-y", "--y-shift", dest='yshift', required=False, help="y-axis shift", type=float, default=0.0)
parser.add_argument("-u", "--x-interval", dest='xinterval', required=False, help="x-axis interval", type=float, default=0.05)
parser.add_argument("-v", "--y-interval", dest='yinterval', required=False, help="y-axis interval", type=float, default=0.05)
parser.add_argument("-w", "--row-limit", dest='row_limit', required=False, help="Maximum number of pictures per row", type=int, default=3)
parser.add_argument("-s", "--scale", dest='scale', required=False, help="Scale", type=float, default=3.0)
parser.add_argument("-a", "--coarsening", dest='coarsening', required=False, help="Coarsening - picks every n-th point", type=int, default=1)
parser.add_argument("-p", "--order-params", dest='order_params', required=False, nargs='+', help="Pick only selected order parameters", type=int, default=None)

# Styling settings
parser.add_argument("-c", "--center-tick-ratio", dest='center_tick_ratio', required=False, help="Center tick ratio", type=float, default=0.2)
parser.add_argument("-i", "--circle-opacity", dest='circle_opacity', required=False, help="Circle fill opacity for 0D", type=float, default=0.6)
parser.add_argument("-r", "--color-text", dest='color_text', required=False, help="Color text", type=str, default='white')
parser.add_argument("-n", "--font-size", dest='font_size', required=False, help="Font size", type=str, default='\\scriptsize')
parser.add_argument("-l", "--ncolors", dest='ncolors', required=False, help="Number of colors", type=int, default=None)
parser.add_argument("-q", "--color-suffix", dest='color_suffix', required=False, help="Suffix for color names", type=str, default='')

parser.add_argument("-g", "--show-topology", dest='show_topology', required=False, help="Show grain topology", action="store_true", default=False)
parser.add_argument("-m", "--show-simplified", dest='show_simplified', required=False, help="Show simplified representation", action="store_true", default=False)
parser.add_argument("-d", "--hide-headers", dest='hide_headers', required=False, help="Hide headers", action="store_true", default=False)

group = parser.add_mutually_exclusive_group(required=False)
group.add_argument("--label-grain-ids", dest='label_grain_ids', required=False, help="Show grain ids as labels", action="store_true", default=False)
group.add_argument("--label-order-params", dest='label_order_params', required=False, help="Show order params as labels", action="store_true", default=False)

args = parser.parse_args()

show_labels = args.label_grain_ids or args.label_order_params

def build_grain(pts, dim):
    if dim == 2:
        center = pts.mean(0)
        angle = np.arctan2(*(pts - center).T[::-1])
        index = np.argsort(angle)
    else:
        hull = ConvexHull(pts)
        index = hull.vertices
    
    return index[0::args.coarsening]

# Get all files to process
files_list = library.get_solutions(args.files, do_sort = False)

if args.output:
    ofname = args.output
else:
    fname_without_ext = os.path.splitext(args.files)[0]
    ofname = fname_without_ext + ".tex"

if not files_list:
    raise Exception("The files list is empty, nothing to process")

# Deal with block headers
headers = None
if not args.hide_headers:
    if len(files_list) == 1:
        fname_without_ext = os.path.splitext(os.path.basename(files_list[0]))[0]
        headers = [fname_without_ext]
    else:
        headers = library.generate_short_labels(files_list)

    headers = [library.tex_escape(h) for h in headers]

    if args.headers:
        for i in range(min(len(headers), len(args.headers))):
            headers[i] = args.headers[min(i, len(args.headers) - 1)]

# Get color scheme based on the order params number
if args.ncolors is None:
    n_op = 0
    for file in files_list:
        src_file = open(file, 'rb')
        src_file.readline() # skip first line - dim
        src_file.readline() # skip second line - number of grains
        data = src_file.readline()
        n_op = max(n_op, int(data))

    ncolors = n_op
else:
    ncolors = args.ncolors

colors = library.get_hex_colors(ncolors)

with open(ofname, 'w') as f:

    for ic, clr in enumerate(colors):
        rgb = library.hex_to_rgb(colors[ic])
        f.write("\\definecolor{clr%d%s}{RGB}{%d,%d,%d}\n" % (ic, args.color_suffix, rgb[0], rgb[1], rgb[2]))
    
    f.write("\n")

    f.write("\\begin{tikzpicture}[scale=%f]\n" % args.scale)

    for j, file in enumerate(files_list):

        src_file = open(file, 'rb')
        data = src_file.readlines()
        data = [d.decode('ascii') for d in data]
        data = [d.replace(" \n", "").replace("\n", "") for d in data]

        dim         = int(data[0])
        n_grains    = int(data[1])
        n_op        = int(data[2])
        grain_ids   = [int(i) for i in data[3].split()]
        grain_to_op = [int(i) for i in data[4].split()]
        point0      = [float(i) for i in data[5].split()]
        point1      = [float(i) for i in data[6].split()]

        properties = [float(i) for i in data[7].split()]
        centers = [[float(i) for i in properties[(g*(dim + 1)) : (g*(dim + 1) + dim)]] for g in range(0, n_grains)]
        radii = [properties[g*(dim + 1) + dim] for g in range(0, n_grains)]

        points = [[float(i) for i in data[8 + g].split()] for g in range(0, n_grains)]
        points = [np.array([[p[g*dim + d] for d in range(0, dim)] for g in range(0, int(len(p)/dim))]) for p in points]

        width = point1[0] - point0[0]
        height = point1[1] - point0[1]

        normalized_width = 1.
        normalized_height = height/width

        jcol = j % args.row_limit
        jrow = j // args.row_limit

        xs = args.xshift + jcol*normalized_width + jcol*args.xinterval
        ys = args.yshift - jrow*normalized_height - jrow*args.yinterval

        if jcol == 0:
            f.write("\n")

        def normalize_point(p):
            point = [0] * dim

            for d in range(0, dim):
                point[d] = (p[d] - point0[0]) / width
            
            return point

        def normalize_radius(r):
            return r / width

        center_tick_radius_normalized = args.center_tick_ratio * normalize_radius(min(radii))

        f.write("\\draw [] (%f,%f) rectangle (%f,%f);\n" % (xs, ys, xs + normalized_width, ys + normalized_height))
        f.write("\n")

        if not args.hide_headers:
            f.write("\\node[] at (%f,%f) {%s %s};\n" % (0.5 + xs, normalized_height + 0.05 + ys, args.font_size, headers[j]))

        for g in range(0, n_grains):
            if len(points[g]) > 0 and (args.order_params is None or grain_to_op[g] in args.order_params):
                indices = build_grain(points[g], dim)

                color = "clr{}{}".format(grain_to_op[g], args.color_suffix)

                lbl = None
                if args.label_grain_ids:
                    lbl = grain_ids[g]
                if args.label_order_params:
                    lbl = grain_to_op[g]

                if args.show_topology:
                    f.write("\\draw [fill=%s]\n" % color)
                    for p in points[g][indices]:
                        pp = normalize_point(p)

                        f.write("(%f, %f) --\n" % (pp[0] + xs, pp[1] + ys))
                    f.write("cycle;\n")

                    if show_labels:
                        c = normalize_point(centers[g])
                        f.write("\\node[anchor=center] at (%f,%f) {%s\color{%s} %d};\n" % (c[0] + xs, c[1] + ys, args.font_size, args.color_text, lbl))

                if args.show_simplified:
                    c = normalize_point(centers[g])
                    r = normalize_radius(radii[g])
                    f.write("\\draw[dashed, color=%s, fill=%s!10!white, fill opacity=%f] (%f,%f) circle (%f);\n" % (color, color, args.circle_opacity, c[0] + xs, c[1] + ys, r))

                    if not args.show_topology:
                        if show_labels:
                            f.write("\\node[anchor=center] at (%f,%f) {%s\color{%s} %d};\n" % (c[0] + xs, c[1] + ys, args.font_size, color, lbl))
                        else:
                            f.write("\\fill [color=%s] (%f, %f) circle (%f);\n" % (color, c[0] + xs, c[1] + ys, center_tick_radius_normalized))
                    elif not show_labels:
                        f.write("\\fill [color=%s] (%f, %f) circle (%f);\n" % (args.color_text, c[0] + xs, c[1] + ys, center_tick_radius_normalized))

                f.write("\n")

    f.write("\\end{tikzpicture}\n")
