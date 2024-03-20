import numpy as np
import math as m
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Transform packing: 1) centirfy (offset) -> 2) scale -> 3) rotate')
parser.add_argument("-f", "--file", dest="file", type=str, required=True, help="Packing file to process")
parser.add_argument("-x", "--suffix", dest="suffix", type=str, required=False, default="mod", help="New file suffix")
parser.add_argument("-c", "--centrify", dest="centrify", action='store_true', required=False, default=False, help="Centrify packing")
parser.add_argument("-s", "--scale", dest="scale", type=float, required=False, default=None, help="Scale packing")
parser.add_argument("-r", "--rotate", dest="rotate", type=float, nargs=3, required=False, default=None, help="Rotate packing using Euler angles (order: Rx->Ry->Rz)")
parser.add_argument("-o", "--offset", dest="offset", type=float, nargs=3, required=False, default=None, help="Packing offset (ignored if --centrify option is provided)")

args = parser.parse_args()

fdata = np.genfromtxt(args.file, dtype=None, names=True, delimiter=',')

smax = sys.float_info.max
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

# Operation 1 - centrify
if args.centrify:
    xc = (xlim[1] - xlim[0]) / 2.
    yc = (ylim[1] - ylim[0]) / 2.
    zc = (zlim[1] - zlim[0]) / 2.

    print("Centrifying packing to its center {}".format([xc, yc, zc]))

    for p in fdata:
        p["x"] -= xc
        p["y"] -= yc
        p["z"] -= zc

elif args.offset:
    print("Offsetting packing with vector {}".format(args.offset))

    for p in fdata:
        p["x"] += args.offset[0]
        p["y"] += args.offset[1]
        p["z"] += args.offset[2]

if args.scale:
    print("Scaling packing with scale {}".format(args.scale))

    for p in fdata:
        p["x"] *= args.scale
        p["y"] *= args.scale
        p["z"] *= args.scale
        p["r"] *= args.scale

if args.rotate:
    print("Rotating packing with Euler angles {}".format(args.rotate))

    def Rx(theta_deg):
        theta = m.radians(theta_deg)
        return np.matrix([[ 1, 0           , 0           ],
                        [ 0, m.cos(theta),-m.sin(theta)],
                        [ 0, m.sin(theta), m.cos(theta)]])
    
    def Ry(theta_deg):
        theta = m.radians(theta_deg)
        return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                        [ 0           , 1, 0           ],
                        [-m.sin(theta), 0, m.cos(theta)]])
    
    def Rz(theta_deg):
        theta = m.radians(theta_deg)
        return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                        [ m.sin(theta), m.cos(theta) , 0 ],
                        [ 0           , 0            , 1 ]])

    R = Rz(args.rotate[2]) * Ry(args.rotate[1]) * Rx(args.rotate[0])

    for p in fdata:
        v1 = np.array([[p["x"]], [p["y"]], [p["z"]]])
        v2 = R * v1
        p["x"] = v2[0]
        p["y"] = v2[1]
        p["z"] = v2[2]

fname, fext = os.path.splitext(args.file)

fnew = fname + "_" + args.suffix + fext

print("File to be saved: {}".format(fnew))

header = ",".join(fdata.dtype.names)

np.savetxt(fnew, fdata, delimiter=",", header=header)
