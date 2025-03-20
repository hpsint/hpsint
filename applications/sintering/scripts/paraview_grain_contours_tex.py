import argparse
import numpy as np
import os
import sys

from paraview.simple import *

tol = 1e-12

# Build local axes
def build_local_axes(origin, ez, candidates):

    ex_orient = None

    for current_orientation in candidates:
        ex_temp = current_orientation - origin
        
        if np.linalg.norm(ex_temp) < tol:
            continue

        ex_temp /= np.linalg.norm(ex_temp)
        if (np.abs(np.dot(ez, ex_temp) - 1.) < tol):
            continue

        ex_orient = ex_temp
        break

    if ex_orient is None:
        raise Exception("None of the candidates fitted as orientation point")
    
    ey = np.cross(ez, ex_orient)
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, ez)

    return ex, ey


# Script arguments
parser = argparse.ArgumentParser(description='Extract grain controus projections in TEX format')
parser.add_argument("-f", "--file", type=str, help="Solution file", required=True)
parser.add_argument("-o", "--output", type=str, help="Output file", required=False, default=None)
parser.add_argument("-n", "--normal", type=float, nargs=3, help="Z-axis normal (not used if --zaxis is defined)", required=False, default=[0., 0., 1.])
parser.add_argument("-r", "--origin", type=float, nargs=3, help="Origin of the plan", required=False, default=[0., 0., 0.])
parser.add_argument("-x", "--xaxis", type=float, nargs=3, help="Orentation point of the x-axis", required=False, default=None)
parser.add_argument("-z", "--zaxis", type=float, nargs=3, help="Orentation point of the z-axis", required=False, default=None)
parser.add_argument("-b", "--bottom-left", type=float, nargs=2, help="Bottom left point of the bounding box", required=False, default=None)
parser.add_argument("-t", "--top-right", type=float, nargs=2, help="Top right point of the bounding box", required=False, default=None)

args = parser.parse_args()

# Get initial assembly length and particle diameter
reader = XMLUnstructuredGridReader(FileName=args.file)
reader.UpdatePipeline()

# Build local axes
origin = np.array(args.origin)
if args.xaxis:
    xaxis = np.array(args.xaxis)

    if np.linalg.norm(xaxis - origin) < tol:
        raise Exception("X-axis orientation point should not coincide with the plane origin")
    
    candidates = [xaxis]
else:
    [x_min, x_max, y_min, y_max, z_min, z_max] = reader.GetDataInformation().GetBounds()

    candidates = [
        np.array([x_min, y_min, z_min]),
        np.array([x_max, y_min, z_min]),
        np.array([x_min, y_max, z_min]),
        np.array([x_max, y_max, z_min]),
        np.array([x_min, y_min, z_max]),
        np.array([x_max, y_min, z_max]),
        np.array([x_min, y_max, z_max]),
        np.array([x_max, y_max, z_max])
    ]

# Build z-axis
if args.zaxis:
    zaxis = np.array(args.zaxis)
    if np.linalg.norm(zaxis - origin) < tol:
        raise Exception("Z-axis orientation point should not coincide with the plane origin")
    
    if args.xaxis:
        if np.linalg.norm(zaxis - xaxis) < tol:
            raise Exception("Orientation points for x- and z-axes should not coincide")

    ez = zaxis - origin
else:
    ez = np.array(args.normal)

    if np.linalg.norm(ez) < tol:
        raise Exception("Normal vector defining z-axis should not be equal to 0")

ez /= np.linalg.norm(ez)

# Build ex and ey    
ex, ey = build_local_axes(origin, ez, candidates)

# Build rotation matrix
ex0 = np.array([1, 0, 0])
ey0 = np.array([0, 1, 0])
ez0 = np.array([0, 0, 1])

R = np.tensordot(ex0, ex, 0) + np.tensordot(ey0, ey, 0) + np.tensordot(ez0, ez, 0)

print("Local axes: {}, {}, {}".format(ex, ey, ez))

slice = Slice(reader)

slice.SliceType = 'Plane'
slice.HyperTreeGridSlicer = 'Plane'
slice.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice.SliceType.Origin = args.origin

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice.HyperTreeGridSlicer.Origin = args.origin

# Properties modified on slice1.SliceType
slice.SliceType.Normal = ez.tolist()

slice.UpdatePipeline()

# Raw Paraview data
number_of_points = slice.GetDataInformation().GetNumberOfPoints()
fullData = servermanager.Fetch(slice)
grain_ids = fullData.GetPointData().GetScalars('grain_id')
order_params = fullData.GetPointData().GetScalars('order_parameter_id')

# Extract data from the VTK file
grains_data = {}
used_order_params = set()
for i in range(fullData.GetNumberOfPoints()):
    grain_id = int(grain_ids.GetValue(i))

    if not grain_id in grains_data:
        order_param = int(order_params.GetValue(i))

        used_order_params.add(order_param)

        grains_data[grain_id] = {
            'grain_index': grain_id,
            'order_param': order_param,
            'radius': 0,
            'center': None,
            'points': []
        }

    p_global = np.array(fullData.GetPoint(i))
    p_local = R.dot(p_global)

    grains_data[grain_id]['points'].append(np.array([p_local[0], p_local[1]]))

# Find center and particle radius
for gid, grain in grains_data.items():
    grain['center'] = np.array([0., 0.])
    for p in grain['points']:
        grain['center'] += p
    grain['center'] /= len(grain['points'])

    grain['radius'] = 0
    for p in grain['points']:
        grain['radius'] = max(grain['radius'], np.linalg.norm(grain['center'] - p))

# Define bounding box
bbox = {'min': None, 'max': None}
if not args.bottom_left:
    bbox['min'] = np.array([sys.float_info.max, sys.float_info.max])
if not args.top_right:
    bbox['max'] = np.array([sys.float_info.min, sys.float_info.min])

# Calculate bounding boxes if not predefined bu the user
for gid, grain in grains_data.items():
    if bbox['min'] is not None:
        bbox['min'][0] = min(bbox['min'][0], grain['center'][0] - grain['radius'])
        bbox['min'][1] = min(bbox['min'][1], grain['center'][1] - grain['radius'])

    if bbox['max'] is not None:
        bbox['max'][0] = max(bbox['max'][0], grain['center'][0] + grain['radius'])
        bbox['max'][1] = max(bbox['max'][1], grain['center'][1] + grain['radius'])

# Set the bounding box if predefined by the user
if args.bottom_left:
    bbox['min'] = np.array(args.bottom_left)
if args.top_right:
    bbox['max'] = np.array(args.top_right)

# File format (data and length):
# problem dimensionality         - dim
# number of grains               - N
# number of order_parameters     - M
# grain indices                  - array[N]
# grain order parameters         - array[N]
# BB bottom left point           - array[dim]
# BB top right point             - array[dim]
# properties (center and radius) - array[(dim+1)*N]
# particle_0                     - 1
# points_0                       - array[...]
# ...
# particle_N                     - 1
# points_N                       - array[...]

# Output save filename
if args.output:
    save_fname = args.output
else:
    fname_without_ext = os.path.splitext(args.file)[0]
    save_fname = fname_without_ext + "_origin=" + ",".join([str(v) for v in args.origin]) + "_normal=" + ",".join([str(v) for v in ez.tolist()]) + ".txt"

print("Saving result to {}".format(save_fname))

with open(save_fname,'w') as f:
    # Write dimension
    f.write("2\n")

    # Write number of grains
    f.write("{:d}\n".format(len(grains_data)))

    # Write number of order parameters
    f.write("{:d}\n".format(len(used_order_params)))

    # Write grain indices
    for gid in grains_data.keys():
        f.write("{:d} ".format(gid))
    f.write("\n")

    # Write order parameters
    for gid, grain in grains_data.items():
        f.write("{:d} ".format(grain['order_param']))
    f.write("\n")

    # Write bouding box
    f.write("{} {}\n".format(bbox['min'][0], bbox['min'][1]))
    f.write("{} {}\n".format(bbox['max'][0], bbox['max'][1]))

    # Write centers and radii
    for gid, grain in grains_data.items():
        f.write("{} {} {} ".format(grain['center'][0], grain['center'][1], grain['radius']))
    f.write("\n")

    # Write particle points
    segment_counter = 0
    for gid, grain in grains_data.items():
        f.write("{:d}\n".format(segment_counter))
        for p in grain['points']:
            f.write("{} {} ".format(p[0], p[1]))
        f.write("\n")
        segment_counter += 1
