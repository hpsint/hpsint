import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import re
import os
import math
import pathlib
import library

parser = argparse.ArgumentParser(description='Process shrinkage data')
parser.add_argument("-m", "--mask", type=str, help="File mask", required=True)
parser.add_argument("-f", "--file", type=str, help="Solution file", required=True)
parser.add_argument("-p", "--path", type=str, help="Common path, can be defined as mask too", required=False, default=None)
parser.add_argument("-g", "--grains", type=int, help="Grain indices", nargs='+', required=False)
parser.add_argument("-q", "--quantities", type=str, help="Grain indices", nargs='+', required=False)
parser.add_argument("-o", "--output", type=str, required=False, help="Destination csv file", default=None)
parser.add_argument("-n", "--replace-nans", action='store_true', help="Replace nans", required=False, default=False)

# Parse arguments
args = parser.parse_args()

# Deal with path names
if args.path is not None:
    list_solution_files = library.get_solutions([os.path.join(args.path, args.file)])
    list_distribution_folders = [os.path.dirname(s) for s in list_solution_files]
    print("")

    if not list_solution_files:
        raise Exception("No files detected that would fit the provided masks")

else:
    list_solution_files = [args.file]
    list_distribution_folders = [os.path.dirname(args.file)]

    if not os.path.isfile(args.file):
        raise Exception("The provided solution file does not exist")

# Read distribution data
list_distributions = []
for f in list_distribution_folders:
    files_list = glob.glob(os.path.join(f, args.mask))
    files_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    list_distributions.append(files_list)

def intersect_ids(file, available_ids):
    qdata = np.genfromtxt(file, dtype=None, names=True)

    if available_ids is None:
        available_ids = set(qdata["id"])
    else:
        available_ids = available_ids.intersection(set(qdata["id"]))

    return available_ids

def intersect_qtys(files_list, available_qtys):
    for f in files_list:
        cdata = np.genfromtxt(f, dtype=None, names=True)
        if available_qtys is None:
            available_qtys = set(cdata.dtype.names)
        else:
            available_qtys = available_qtys.intersection(set(cdata.dtype.names))

    return available_qtys

has_missing_arguments = False

# If no grain id provided to track then show available indices
if not args.grains:
    print("You did not specify any grain index to track. These are the grain indices available at the start and end of the simulations:")

    available_ids_start = None
    available_ids_end = None
    for files_list in list_distributions:
        available_ids_start = intersect_ids(files_list[0], available_ids_start)
        available_ids_end = intersect_ids(files_list[-1], available_ids_end)

    sorted_available_ids_start = sorted(list(available_ids_start))
    sorted_available_ids_end = sorted(list(available_ids_end))
    
    print(" -- start:")
    print(sorted_available_ids_start)

    print(" -- end:")
    print(sorted_available_ids_end)

    print("Rerun the script specifying at least one of them")
    print("")
    
    has_missing_arguments = True

if not args.quantities:

    print("You did not specify any quantity to track. These are the quantities available:")

    available_qtys = None
    for files_list in list_distributions:
        available_qtys = intersect_qtys(files_list, available_qtys)

    sorted_available_qtys = sorted(list(available_qtys))
        
    for name in sorted_available_qtys:
        print(" -- {}".format(name))
    
    print("Rerun the script specifying at least one of them")
    print("")

    has_missing_arguments = True
    
if has_missing_arguments:
    exit()

# Total number of quantities
n_qtys = len(args.grains) * len(args.quantities)

# Build a CSV header
csv_header = ["time"]
for gid in args.grains:
    for qty in args.quantities:
        csv_header.append("{}_{}".format(qty, gid))
csv_header = " ".join(csv_header)

# Build CSV format
csv_format = ["%g"] * (n_qtys + 1)
csv_format = " ".join(csv_format)

# Save csv names
if args.output is not None:
    file_names = library.generate_short_labels(list_solution_files)
    csv_names = [os.path.join(args.output, n.replace(os.sep, "_") + "_grains_quantities.csv")  for n in file_names]
else:
    csv_names = [os.path.splitext(f)[0] + "_grains_quantities.csv" for f in list_solution_files]

f_counter = 0
n_folders = len(list_solution_files)
for file_solution, files_list in zip(list_solution_files, list_distributions):

    print("Parsing folder  {} ({}/{})".format(os.path.dirname(file_solution), f_counter + 1, n_folders))

    # Read solution file
    fdata = np.genfromtxt(file_solution, dtype=None, names=True)

    # Total number of stats files
    n_rows = len(files_list)

    # Flag indentifying that we had nans at the beginning
    fill_nans_up_to = [None] * n_qtys

    # Init CSV data
    csv_data = np.empty((len(fdata["time"]), n_qtys + 1), float)

    for idx, log_file in enumerate(files_list):

        if idx >= fdata.shape[0]:
            prefix = "├" if idx + 1 < n_rows else "└"
            print("{}─ Skipping file {} ({}/{}) due to data inconsistency".format(prefix, log_file, idx + 1, n_rows))
            continue

        csv_data[idx, 0] = fdata["time"][idx]

        qdata = np.genfromtxt(log_file, dtype=None, names=True)

        qty_counter = 0
        for gid in args.grains:

            itemindex = np.where(qdata["id"] == gid)
            if not len(itemindex[0]):
                exit("It seems you provided the wrong grain id = {}, check the list of those available for analysis".format(gid))

            for qty in args.quantities:

                if qty not in qdata.dtype.names:
                    exit("It seems you provided the wrong quantity name \"{}\", check the list of those available for analysis".format(qty))

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

        prefix = "├" if idx + 1 < n_rows else "└"
        print("{}─ Parsing file {} ({}/{})".format(prefix, log_file, idx + 1, n_rows))

    file_path = csv_names.pop(0)
    pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    np.savetxt(file_path, csv_data, delimiter=' ', header=csv_header, fmt=csv_format, comments='')
    print("   Saving result to {}".format(file_path))
    print("")

    f_counter += 1
