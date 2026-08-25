import argparse
import library
import glob
import numpy as np
import pathlib
import os
import re

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

# Measure certain value along the line as the distance between the first and
# the last points exceeding the given threshold
def measure_over_line_ends(pline, quantity, threshold, length):
    nbp = pline.GetDataInformation().GetNumberOfPoints()

    fullData = servermanager.Fetch(pline)
    vals = fullData.GetPointData().GetScalars(quantity)

    first_point = None
    last_point = None

    for jpoint in range(0, nbp):
        q_val = vals.GetValue(jpoint)
        if (q_val > threshold):
            if first_point is None:
                first_point = fullData.GetPoint(jpoint)
            last_point = fullData.GetPoint(jpoint)

    if first_point is None or last_point is None:
        return 0

    magnitude = np.sqrt(sum((a - b) ** 2 for a, b in zip(first_point, last_point)))

    return magnitude

# Script arguments
parser = argparse.ArgumentParser(description='Extract neck and packing size for the 2 particle case from Paraview')
parser.add_argument("-m", "--mask", type=str, help="File mask", required=False, default="solution.*.vtu")
parser.add_argument("-f", "--file", type=str, help="Solution file", required=False, default="solution.log")
parser.add_argument("-p", "--path", type=str, help="Common path, can be defined as mask too", required=False, default=None)
parser.add_argument("-o", "--output", type=str, help="Output csv file", required=False, default=None)
parser.add_argument("-r", "--resolution", type=int, help="Number of points for the line filters", required=False, default=10000)
parser.add_argument("-t", "--threshold", type=float, help="Minimum value for the quantity", required=False, default=0.5)
parser.add_argument("-q", "--quantity", type=str, help="Quantity data name to analyze", required=False, default="c")
parser.add_argument("-a", "--alignment", type=str, help="Axis along which the assembly is aligned", required=False, default="x")
parser.add_argument("-d", "--direction", type=str, help="Axis along which the neck is measured", required=False, default="y")
parser.add_argument("-e", "--extend-to", dest='extend_to', required=False, help="Extend labels when shortening to", type=str, default=None)
parser.add_argument("-u", "--suffix", dest='suffix', required=False, help="Suffix to append to the save file", type=str, default="_vtk_postproc")
parser.add_argument("--time-start", dest='time_start', required=False, help="Start of the time range to parse", type=float, default=None)
parser.add_argument("--time-end", dest='time_end', required=False, help="End of the time range to parse", type=float, default=None)
parser.add_argument("--first-last-only", dest='first_last_only', required=False, help="Parse only the first and the last files within the time range", action='store_true')
parser.add_argument("--endpoints", dest='endpoints', required=False, help="Measure the distance between the first and the last points above the threshold instead of the integral length", action='store_true')

args = parser.parse_args()

# Select the measurement function to be used
measure_func = measure_over_line_ends if args.endpoints else measure_over_line

# Deal with path names
if args.path is not None:
    list_solution_files = library.get_solutions([os.path.join(args.path, args.file)])
    list_vtk_folders = [os.path.dirname(s) for s in list_solution_files]
    print("")

    if not list_solution_files:
        raise Exception("No files detected that would fit the provided masks")

else:
    list_solution_files = [args.file]
    list_vtk_folders = [os.path.dirname(args.file)]

    if not os.path.isfile(args.file):
        raise Exception("The provided solution file does not exist")

# Read vtk data
list_vtks = []
for f in list_vtk_folders:
    files_list = glob.glob(os.path.join(f, args.mask))
    files_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    list_vtks.append(files_list)

# CSV data
csv_header = ["time", "dt", "neck_diameter", "neck_growth", "length", "shrinkage"]
n_cols = len(csv_header)
csv_header = " ".join(csv_header)

# Build CSV format
csv_format = ["%g"] * (n_cols)
csv_format = " ".join(csv_format)

# Save csv names
if args.output is not None:
    file_names = library.generate_short_labels(list_solution_files, args.extend_to)
    csv_names = [os.path.join(args.output, n.replace(os.sep, "_") + args.suffix + ".csv")  for n in file_names]
else:
    csv_names = [os.path.splitext(f)[0] + args.suffix + ".csv" for f in list_solution_files]

f_counter = 0
n_folders = len(list_solution_files)
for file_solution, files_list in zip(list_solution_files, list_vtks):

    print("Parsing folder {} ({}/{})".format(os.path.dirname(file_solution), f_counter + 1, n_folders))

    if not len(files_list):
        print("The folder does not contain any suitable data to parse, skipping")
        print("")
        continue

    # Read solution file
    fdata = np.genfromtxt(file_solution, dtype=None, names=True)

    # Total number of vtk files
    n_rows = len(files_list)

    # Restrict the files to be parsed to the requested time range. Indices
    # beyond the data available in the solution file are dropped as they
    # cannot be matched against a time value.
    selected_indices = [idx for idx in range(min(n_rows, fdata.shape[0]))
                         if (args.time_start is None or fdata["time"][idx] >= args.time_start)
                         and (args.time_end is None or fdata["time"][idx] <= args.time_end)]

    if not selected_indices:
        print("No files match the specified time range, skipping")
        print("")
        continue

    # Keep only the first and the last files of the selected range if requested
    if args.first_last_only and len(selected_indices) > 2:
        selected_indices = [selected_indices[0], selected_indices[-1]]

    n_selected = len(selected_indices)

    csv_data = np.zeros(shape=(n_selected, n_cols))

    # Get initial assembly length and particle diameter from the first file of the selected range
    reader = XMLUnstructuredGridReader(FileName=files_list[selected_indices[0]])
    reader.UpdatePipeline()

    [line_length, domain_length] = build_line(reader, args.alignment)
    length0 = measure_func(line_length, args.quantity, args.threshold, domain_length)

    diameter0 = length0 / 2

    for pos, idx in enumerate(selected_indices):

        vtk_file = files_list[idx]

        prefix = "|--" if pos + 1 < n_selected else "`--"

        print("{} Parsing file {} ({}/{})".format(prefix, vtk_file, pos + 1, n_selected))

        reader = XMLUnstructuredGridReader(FileName=vtk_file)
        reader.UpdatePipeline()

        # Measurement lines
        [line_length, domain_length] = build_line(reader, args.alignment)
        [line_neck, domain_width] = build_line(reader, args.direction)

        neck_diameter = measure_func(line_neck, args.quantity, args.threshold, domain_width)
        neck_growth = neck_diameter / diameter0

        length = measure_func(line_length, args.quantity, args.threshold, domain_length)
        shrinkage = (length0 - length) / length0

        csv_data[pos, 0] = fdata["time"][idx]
        csv_data[pos, 1] = fdata["dt"][idx]
        csv_data[pos, 2] = neck_diameter
        csv_data[pos, 3] = neck_growth
        csv_data[pos, 4] = length
        csv_data[pos, 5] = shrinkage
        

    file_path = csv_names.pop(0)
    pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    np.savetxt(file_path, csv_data, delimiter=' ', header=csv_header, fmt=csv_format, comments='')
    print("   Saving result to {}".format(file_path))
    print("")
