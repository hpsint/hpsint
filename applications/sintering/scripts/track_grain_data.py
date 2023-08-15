import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import re
import os
import math
import pathlib

parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-m", "--mask", type=str, help="File mask", required=True)
parser.add_argument("-f", "--file", type=str, help="Solution file", required=True)
parser.add_argument("-p", "--path", type=str, help="Common path", required=False, default=None)
parser.add_argument("-g", "--grains", type=int, help="Grain indices", nargs='+', required=False)
parser.add_argument("-q", "--quantities", type=str, help="Grain indices", nargs='+', required=False)
parser.add_argument("-o", "--output", type=str, required=False, help="Destination csv file", default=None)
parser.add_argument("-n", "--replace-nans", action='store_true', help="Replace nans", required=False, default=False)

# Parse arguments
args = parser.parse_args()

# Deal with path names
file_solution = args.file
file_distribution = args.mask
if args.path is not None:
    file_solution = os.path.join(args.path, file_solution)
    file_distribution = os.path.join(args.path, file_distribution)

# Read solution file
fdata = np.genfromtxt(file_solution, dtype=None, names=True)

# Read distribution data
files_list = glob.glob(file_distribution)
files_list.sort(key=lambda f: int(re.sub('\D', '', f)))

has_missing_arguments = False

# If no grain id provided to track then show available indices
if not args.grains:
    print("You did not specify any grain index to track. These are the grain indices available at the start and end of the simulation:")
    
    print(" -- start t = {}:".format(fdata["time"][0]))
    qdata = np.genfromtxt(files_list[0], dtype=None, names=True)
    print(qdata["id"])

    print(" -- end t = {}:".format(fdata["time"][-1]))
    qdata = np.genfromtxt(files_list[-1], dtype=None, names=True)
    print(qdata["id"])

    print("Rerun the script specifying at least one of them")
    print("")
    
    has_missing_arguments = True

if not args.quantities:

    print("You did not specify any quantity to plot. These are the quantities available:")

    available_fields = None
    for f in files_list:
        cdata = np.genfromtxt(f, dtype=None, names=True)
        if available_fields is None:
            available_fields = set(cdata.dtype.names)
        else:
            available_fields = available_fields.intersection(set(cdata.dtype.names))

    sorted_available_fields = sorted(list(available_fields))
        
    for name in list(sorted_available_fields):
        print(" -- {}".format(name))
    
    print("Rerun the script specifying at least one of them")
    print("")

    has_missing_arguments = True
    
if has_missing_arguments:
    exit()

# Total number of stats files
n_rows = len(files_list)

# Build a CSV header, format and init the data array
csv_header = ["time"]
for gid in args.grains:
    for qty in args.quantities:
        csv_header.append("{}_{}".format(qty, gid))

csv_format = ["%g"] * len(csv_header)
csv_format = " ".join(csv_format)

csv_data = np.empty((len(fdata["time"]), len(csv_header)), float)
csv_header = " ".join(csv_header)

# Flag indentifying that we had nans at the beginning
n_qtys = len(args.grains) * len(args.quantities)
fill_nans_up_to = [None] * n_qtys

for idx, log_file in enumerate(files_list):

    csv_data[idx, 0] = fdata["time"][idx]

    qdata = np.genfromtxt(log_file, dtype=None, names=True)

    qty_counter = 0
    for gid in args.grains:

        itemindex = np.where(qdata["id"] == gid)

        for qty in args.quantities:

            qty_name = "{}_{}".format(qty, gid)
            
            csv_data[idx, qty_counter+1] = qdata[qty][itemindex] if itemindex and qty in qdata.dtype.names else math.nan

            if args.replace_nans:
                if math.isnan(csv_data[idx, qty_counter+1]):
                    if idx == 0 or math.isnan(csv_data[idx-1, qty_counter+1]):
                        fill_nans_up_to[qty_counter] = idx
                    else:
                        csv_data[idx, qty_counter+1] = csv_data[idx-1, qty_counter+1]

                elif fill_nans_up_to[qty_counter] is not None:
                    while fill_nans_up_to[qty_counter] >= 0:
                        csv_data[fill_nans_up_to[qty_counter], qty_counter+1] = csv_data[idx, qty_counter+1]
                        fill_nans_up_to[qty_counter] -= 1
            
            qty_counter += 1

    print("Parsing file {} ({}/{})".format(log_file, idx + 1, n_rows))

if args.output:
    csv_filename = args.output
else:
    csv_filename = os.path.splitext(file_solution)[0] + "_grains_quantities.csv"

pathlib.Path(os.path.dirname(csv_filename)).mkdir(parents=True, exist_ok=True)
np.savetxt(csv_filename, csv_data, delimiter=' ', header=csv_header, fmt=csv_format, comments='')
