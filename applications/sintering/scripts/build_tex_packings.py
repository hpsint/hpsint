import os
import numpy as np
import argparse
import library
import alphashape
import psutil
import queue
import multiprocessing
from random import randrange
from scipy.spatial import ConvexHull

parser = argparse.ArgumentParser(description='Build tex packing from special hpsint output')
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

parser.add_argument("-t", "--show-topology", dest='show_topology', required=False, help="Show grain topology", action="store_true", default=False)
parser.add_argument("-m", "--show-simplified", dest='show_simplified', required=False, help="Show simplified representation", action="store_true", default=False)
parser.add_argument("-d", "--hide-headers", dest='hide_headers', required=False, help="Hide headers", action="store_true", default=False)

parser.add_argument("-g", "--rand-range", dest='rand_range', required=False, help="Randomization magnitude divided by 1000", type=int, default=2)
parser.add_argument("-z", "--contour-ordering", default='convex', const='iterate', nargs='?', choices=['iterate', 'optimize', 'convex'], help="Choose the contour points ordering")

group = parser.add_mutually_exclusive_group(required=False)
group.add_argument("--label-grain-ids", dest='label_grain_ids', required=False, help="Show grain ids as labels", action="store_true", default=False)
group.add_argument("--label-order-params", dest='label_order_params', required=False, help="Show order params as labels", action="store_true", default=False)

args = parser.parse_args()

show_labels = args.label_grain_ids or args.label_order_params

def build_grain(pts_in, dim):

    if args.contour_ordering != 'convex':
        pts = []
        for i in range(len(pts_in)):
            pts.append((pts_in[i][0] + randrange(-args.rand_range, args.rand_range)/1000., pts_in[i][1] + randrange(-args.rand_range, args.rand_range)/1000.))

        last_alpha = 0

        if args.contour_ordering == 'optimize':
            last_alpha = 0.9 * alphashape.optimizealpha(pts)

        elif args.contour_ordering == 'iterate':
            alpha0 = 0.0
            alpha_step = 0.01
            alpha = alpha0

            hull = alphashape.alphashape(pts, alpha)

            last_na_points = 0
            last_length = 0

            do_iterate = True

            while do_iterate:
                alpha += alpha_step
                hull = alphashape.alphashape(pts, alpha)
                if hasattr(hull, 'exterior'):
                    na_points = len(hull.exterior.coords.xy[0])
                    dl = (hull.length - last_length) / hull.length

                    # This is to detect if inner intersections have appeared or not
                    if alpha > 0.2 and dl > 0.1:
                        do_iterate = False
                        break

                    last_length = hull.length
                    if (na_points > last_na_points):
                        last_na_points = na_points
                        last_alpha = alpha
                else:
                    do_iterate = False

        if last_alpha > 0:
            hull = alphashape.alphashape(pts, last_alpha)

            if hasattr(hull, 'exterior'):
                hull_pts = hull.exterior.coords.xy
                ordered_points = [[hull_pts[0][i], hull_pts[1][i]] for i in range(len(hull_pts[0]))]

                return ordered_points

    # If we reached here, then either 'convex' option was chosen for ordering or we failed with the previous methods
    if dim == 2:
        center = pts_in.mean(0)
        angle = np.arctan2(*(pts_in - center).T[::-1])
        index = np.argsort(angle)
    else:
        hull = ConvexHull(pts_in)
        index = hull.vertices

    return pts_in[index]

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def axes_rotation(x):
    return angle_between(np.array([1, 0]), np.array(x)) * 180. / np.pi

# Multiprocessing job
def do_job(tasks_to_execute, tasks_completed):
    while True:
        try:
            grain_index = tasks_to_execute.get_nowait()
        except queue.Empty:
            break
        else:
            print("processing grain {} ...".format(grain_index))
            ordered_points = build_grain(points[grain_index][0::args.coarsening], dim)
            print("completed grain {} ...".format(grain_index))
            
            tasks_completed.put((grain_index, ordered_points))

    return True

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

# Main colors
colors = library.get_hex_colors(ncolors)

# Multiprocessing settings
number_of_processes = psutil.cpu_count(logical=False)

print("cpu_count = {}".format(number_of_processes))

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

        print("n_grains = {}".format(n_grains))
        print("n_op = {}".format(n_op))

        n_props_to_grain_types = {(dim + 1): 'spherical', (2*dim + dim*dim): 'elliptical', (dim): 'wavefront'}

        properties = [float(i) for i in data[7].split()]

        # Read grains properties
        min_radius = 1e16
        grains = []
        n_props = len(properties)
        i = 0
        while i < n_props:
            n_data = int(properties[i])
            if n_data not in n_props_to_grain_types:
                raise Exception("Unknown grain type for n_data = {}".format(n_data))
            
            gtype = n_props_to_grain_types[n_data]
            center = [float(j) for j in properties[(i + 1) : (i + 1 + dim)]]

            grain = {'type': gtype, 'center': center}

            if gtype == 'spherical':
                grain['radius'] = float(properties[i + 1 + dim])
                min_radius = min(min_radius, grain['radius'])
            elif gtype == 'elliptical':
                grain['axes'] = [[float(j) for j in properties[(i + 1 + (1 + d)*dim) : (i + 1 + (2 + d)*dim)]] for d in range(0, dim)]
                grain['radii'] = [float(j) for j in properties[(i + 1 + (1 + dim)*dim) : (i + 1 + (2 + dim)*dim)]]
                min_radius = min(min_radius, min(grain['radii']))

            grains.append(grain)
            
            i += n_data + 1

        points = [[] for i in range(n_grains)]
        for g in range(8, len(data), 2):
            grain_num = int(data[g])
            points[grain_num] += [float(i) for i in data[g + 1].split()]

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
                point[d] = (p[d] - point0[d]) / width
            
            return point

        def normalize_radius(r):
            return r / width

        min_radius = min(min_radius, width)

        center_tick_radius_normalized = args.center_tick_ratio * normalize_radius(min_radius)

        f.write("\\draw [] (%f,%f) rectangle (%f,%f);\n" % (xs, ys, xs + normalized_width, ys + normalized_height))
        f.write("\n")

        if not args.hide_headers:
            f.write("\\node[] at (%f,%f) {%s %s};\n" % (0.5 + xs, normalized_height + 0.05 + ys, args.font_size, headers[j]))

        # Precompute contours if we need them
        tasks_to_execute = multiprocessing.Queue()
        tasks_completed = multiprocessing.Queue()
        processes = []

        grain_contours = [[] for _ in range(n_grains)]
        if args.show_topology:

            for g in range(0, n_grains):
                if len(points[g]) > 0 and (args.order_params is None or grain_to_op[g] in args.order_params):
                    tasks_to_execute.put(g)

            print("n_grains to process = {}".format(tasks_to_execute.qsize()))

            # creating processes
            for w in range(number_of_processes):
                p = multiprocessing.Process(target=do_job, args=(tasks_to_execute, tasks_completed))
                processes.append(p)
                p.start()

            # completing process
            for p in processes:
                p.join()

            # handle results
            while not tasks_completed.empty():
                res = tasks_completed.get()
                grain_contours[res[0]] = res[1]

        for g in range(0, n_grains):
            if len(grain_contours[g]) > 0:

                color = "clr{}{}".format(grain_to_op[g], args.color_suffix)

                lbl = None
                if args.label_grain_ids:
                    lbl = grain_ids[g]
                if args.label_order_params:
                    lbl = grain_to_op[g]

                c = normalize_point(grains[g]['center'])
                cx = c[0] + xs
                cy = c[1] + ys

                if args.show_topology:
                    f.write("\\draw [fill=%s]\n" % color)
                    for p in grain_contours[g]:
                        pp = normalize_point(p)

                        f.write("(%f, %f) --\n" % (pp[0] + xs, pp[1] + ys))
                    f.write("cycle;\n")

                    if show_labels:
                        f.write("\\node[anchor=center] at (%f,%f) {%s\color{%s} %d};\n" % (cx, cy, args.font_size, args.color_text, lbl))

                if args.show_simplified:
                    c = normalize_point(grains[g]['center'])

                    if grains[g]['type'] == 'spherical':
                        r = normalize_radius(grains[g]['radius'])
                        f.write("\\draw[dashed, color=%s, fill=%s!10!white, fill opacity=%f] (%f,%f) circle (%f);\n"
                            % (color, color, args.circle_opacity, cx, cy, r))
                    elif grains[g]['type'] == 'elliptical':
                        radii = [normalize_radius(r) for r in grains[g]['radii']]
                        angle = axes_rotation(grains[g]['axes'][0])
                        f.write("\\draw[dashed, color=%s, fill=%s!10!white, fill opacity=%f, rotate around={%f:(%f,%f)}] (%f,%f) ellipse (%f and %f);\n"
                            % (color, color, args.circle_opacity, -angle, cx, cy, cx, cy, radii[0], radii[1]))

                    if not args.show_topology:
                        if show_labels:
                            f.write("\\node[anchor=center] at (%f,%f) {%s\color{%s} %d};\n" % (cx, cy, args.font_size, color, lbl))
                        else:
                            f.write("\\fill [color=%s] (%f, %f) circle (%f);\n" % (color, cx, cy, center_tick_radius_normalized))
                    elif not show_labels:
                        f.write("\\fill [color=%s] (%f, %f) circle (%f);\n" % (args.color_text, cx, cy, center_tick_radius_normalized))

                f.write("\n")

    f.write("\\end{tikzpicture}\n")
