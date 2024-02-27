from vtk import vtkXMLUnstructuredGridReader
from functools import reduce
import glob
import re
import numpy as np
import argparse

from paraview.simple import *

# Build measurement line
def build_line(reader, axis):
    [x_min, x_max, y_min, y_max, z_min, z_max] = reader.GetDataInformation().GetBounds()

    # Default values
    x_start = (x_max + x_min) / 2
    y_start = (y_max + y_min) / 2
    z_start = (z_max + z_min) / 2
    x_end = x_start
    y_end = y_start
    z_end = z_start

    if axis == "x":
        x_start = x_min
        x_end = x_max
        length = x_max - x_min
    elif axis == "y":
        y_start = y_min
        y_end = y_max
        length = y_max - y_min
    elif axis == "z":
        z_start = z_min
        z_end = z_max
        length = z_max - z_min

    line = PlotOverLine(reader)
    line.Point1 = [x_start, y_start, z_start]
    line.Point2 = [x_end, y_end, z_end]
    line.Resolution = args.resolution
    line.UpdatePipeline()

    return [line, length]

# Measure certain value along the line
def measure_over_line(pline, quantity, threshold, length):
    nbp = pline.GetDataInformation().GetNumberOfPoints()
    size_step = length/nbp

    active_points_count = 0

    fullData = servermanager.Fetch(pline)
    vals = fullData.GetPointData().GetScalars(quantity)

    for jpoint in range(0, nbp):
        q_val = vals.GetValue(jpoint)
        if (q_val > threshold):
            active_points_count += 1

    magnitude = size_step * active_points_count

    return magnitude

# Script arguments
parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-m", "--mask", type=str, help="File mask", required=True)
parser.add_argument("-o", "--output", type=str, help="Output csv file", required=False, default="")
parser.add_argument("-r", "--resolution", type=int, help="Number of points for the line filters", required=False, default=10000)
parser.add_argument("-t", "--threshold", type=float, help="Minimum value for the quantity", required=False, default=0.5)
parser.add_argument("-q", "--quantity", type=str, help="Quantity data name to analyze", required=False, default="c")
parser.add_argument("-a", "--alignment", type=str, help="Axis along which the assembly is aligned", required=False, default="x")
parser.add_argument("-d", "--direction", type=str, help="Axis along which the neck is measured", required=False, default="y")

args = parser.parse_args()

# Get vtk files according to the mask and sort it by number
vtk_files_list = glob.glob(args.mask)
vtk_files_list.sort(key=lambda f: int(re.sub('\D', '', f)))

# Build reader
#reader = vtkXMLUnstructuredGridReader()

# CSV data
csv_header = ["t", "neck_diameter", "neck_growth", "length", "shrinkage"]
n_rows = len(vtk_files_list)
n_cols = len(csv_header)
csv_data = np.zeros(shape=(n_rows, n_cols))

# Get initial assembly length and particle diameter
reader = XMLUnstructuredGridReader(FileName=vtk_files_list[0])
reader.UpdatePipeline()

[line_length, domain_length] = build_line(reader, args.alignment)
length0 = measure_over_line(line_length, args.quantity, args.threshold, domain_length)

diameter0 = length0 / 2

for idx, vtk_file in enumerate(vtk_files_list):

    print("Parsing file {} ({}/{})".format(vtk_file, idx + 1, n_rows))

    reader = XMLUnstructuredGridReader(FileName=vtk_file)
    reader.UpdatePipeline()

    # Measurement lines
    [line_length, domain_length] = build_line(reader, args.alignment)
    [line_neck, domain_width] = build_line(reader, args.direction)

    neck_diameter = measure_over_line(line_neck, args.quantity, args.threshold, domain_width)
    neck_growth = neck_diameter / diameter0

    length = measure_over_line(line_length, args.quantity, args.threshold, domain_length)
    shrinkage = (length0 - length) / length0

    csv_data[idx, 0] = 0
    csv_data[idx, 1] = neck_diameter
    csv_data[idx, 2] = neck_growth
    csv_data[idx, 3] = length
    csv_data[idx, 4] = shrinkage

# Save to csv
if len(args.output) > 0:
    np.savetxt(args.output, csv_data, header=','.join(csv_header), comments='', delimiter=',')

print(csv_data)
