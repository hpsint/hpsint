import os
import numpy as np
from argparse import ArgumentParser

def extract(filename, scale=1., limits_x=None, limits_y=None, limits_z=None, save_path=None, suffix="particles.cloud"):
    cdata = np.genfromtxt(filename, delimiter=',', dtype=None, names=True)

    X = np.empty((0,4), float)
    count_particles = 0
    for row in cdata:

        new_x = float(row['x'])
        new_y = float(row['y'])
        new_z = float(row['z'])
        new_r = float(row['r'])

        add_particle = True

        if limits_x is not(None):
            add_particle = add_particle and (new_x - new_r > limits_x[0] and new_x + new_r < limits_x[1])
        if limits_y is not(None):
            add_particle = add_particle and (new_y - new_r > limits_y[0] and new_y + new_r < limits_y[1])
        if limits_z is not(None):
            add_particle = add_particle and (new_z - new_r > limits_z[0] and new_z + new_r < limits_z[1])

        if add_particle:
            X = np.append(X, np.array([[scale * new_x, scale * new_y, scale * new_z, scale * new_r]]), axis=0)
            count_particles += 1

    fname_particles = str(count_particles) + suffix
    if save_path:
        fname_particles = os.path.join(save_path, fname_particles)

    header_particles = "x,y,z,r"
    fmt_particles = "%g,%g,%g,%g"
    np.savetxt(fname_particles, X, delimiter=' ', header=header_particles, fmt=fmt_particles)

    if limits_x is not(None):
        print("lim_x: {} .. {}".format(limits_x[0], limits_x[1]))
    if limits_y is not(None):
        print("lim_y: {} .. {}".format(limits_y[0], limits_y[1]))
    if limits_z is not(None):
        print("lim_z: {} .. {}".format(limits_z[0], limits_z[1]))

    print("Total number of particles extracted: {} / {}".format(count_particles, len(cdata)))

if __name__ == '__main__':

    parser = ArgumentParser(description='Extract a part of a large packing bounded by a box')
    parser.add_argument("-f", "--file", dest="filename", required=True, help="Filename")
    parser.add_argument("-s", "--scale", dest='scale', required=False, help="Scale dimensions", default=1.0, type=float)
    parser.add_argument("-x", "--limits-x", dest='limits_x', nargs=2, required=False, help="Limits in x-direction", type=float)
    parser.add_argument("-y", "--limits-y", dest='limits_y', nargs=2, required=False, help="Limits in y-direction", type=float)
    parser.add_argument("-z", "--limits-z", dest='limits_z', nargs=2, required=False, help="Limits in z-direction", type=float)
    parser.add_argument("-u", "--suffix", dest='suffix', required=False, help="Suffix to append to the save file", default="particles.cloud", type=str)
    parser.add_argument("-p", "--path", dest='suffix', required=False, help="Save path", default=None, type=str)

    args = parser.parse_args()

    extract(args.filename, args.scale, args.limits_x, args.limits_y, args.limits_z, args.path, args.suffix)
