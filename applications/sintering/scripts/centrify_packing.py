import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Centrify packing')
parser.add_argument("-f", "--file", dest="file", type=str, required=True, help="Packing file to process")
parser.add_argument("-s", "--suffix", dest="suffix", type=str, required=False, default="center", help="New file suffix")

args = parser.parse_args()

fdata = np.genfromtxt(args.file, dtype=None, names=True, delimiter=',')

smax = 1e16
xlim = [smax, -smax]
ylim = [smax, -smax]
zlim = [smax, -smax]

for p in fdata:
    xlim[0] = min(p["x"] - p["r"], xlim[0])
    xlim[1] = max(p["x"] + p["r"], xlim[1])
    ylim[0] = min(p["y"] - p["r"], ylim[0])
    ylim[1] = max(p["y"] + p["r"], ylim[1])
    zlim[0] = min(p["z"] - p["r"], zlim[0])
    zlim[1] = max(p["z"] + p["r"], zlim[1])

print("Packing dimensions:")
print("  x = {}".format(xlim))
print("  y = {}".format(ylim))
print("  x = {}".format(zlim))

xc = (xlim[1] - xlim[0]) / 2.
yc = (ylim[1] - ylim[0]) / 2.
zc = (zlim[1] - zlim[0]) / 2.

print("Packing center = {}".format([xc, yc, zc]))

for p in fdata:
    p["x"] -= xc
    p["y"] -= yc
    p["z"] -= zc

fname, fext = os.path.splitext(args.file)

fnew = fname + "_" + args.suffix + fext

print("File to be saved: {}".format(fnew))

header = ",".join(fdata.dtype.names)

np.savetxt(fnew, fdata, delimiter=",", header=header)
