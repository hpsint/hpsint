import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser
from scipy.stats import norm
from matplotlib.ticker import PercentFormatter

def to_degs(rads):
    return rads * 180. / np.pi

def to_dia_mkm(val):
    return val * 2 * 1e6

def cap_volume(r, b):

    inside_sqrt = r*r - b*b

    with warnings.catch_warnings():
        warnings.filterwarnings('error')

        if inside_sqrt < 0 and inside_sqrt < np.abs(r)*1e-10:
            inside_sqrt = 0.0

        try:
            a = np.sqrt(inside_sqrt)
        except FloatingPointError:
            print('Warning was raised as an exception!')

    h = r - b

    V = 1./6. * np.pi * h * (3*a*a + h*h)

    return V

# Now create segments
def distribute_in_segments(particles, lims, divs, direction):
    
    coord_min = lims[direction][0]
    coord_max = lims[direction][1]

    all_dirs = [0, 1, 2]
    d_id = all_dirs.index(direction)
    del all_dirs[d_id]

    width = lims[all_dirs[0]][1] - lims[all_dirs[0]][0]
    height = lims[all_dirs[1]][1] - lims[all_dirs[1]][0]
    area_cross_section = width*height

    step = (coord_max - coord_min) / divs

    segments = []
    densities = []
    for i in range(divs):

        s = {
            'start': coord_min + i*step,
            'end': coord_min + (i+1)*step,
            'particles': [],
            'v_particles': 0,
            'v_solid': step * area_cross_section,
            'rel_density': 0
        }

        for p_id, p in particles.items():

            r = p['radius']

            s_lower = s['start']
            s_upper = s['end']

            pt_lower = p['center'][direction] - r
            pt_upper = p['center'][direction] + r

            V = 0

            if pt_lower > s_lower and pt_lower < s_upper:

                if pt_upper < s_upper: # whole sphere
                    V = 4./3. * np.pi * r**3

                elif p['center'][direction] > s_upper:
                    V = cap_volume(r, p['center'][direction] - s_upper)

                elif p['center'][direction] < s_upper:
                    V0 = 4./3. * np.pi * r**3
                    V1 = cap_volume(r, s_upper - p['center'][direction])
                    V = V0 - V1

            elif pt_upper > s_lower and pt_upper < s_upper:

                if p['center'][direction] < s_lower:
                    V = cap_volume(r, s_lower - p['center'][direction])

                elif p['center'][direction] > s_lower:
                    V0 = 4./3. * np.pi * r**3
                    V1 = cap_volume(r, p['center'][direction] - s_lower)
                    V = V0 - V1

            elif pt_lower < s_lower and pt_upper > s_upper:
                if s_lower < p['center'][direction] and p['center'][direction] < s_upper:
                    V0 = 4./3. * np.pi * r**3
                    V1 = cap_volume(r, p['center'][direction] - s_lower)
                    V2 = cap_volume(r, s_upper - p['center'][direction])

                    V = V0 - V1 - V2

                elif s_upper < p['center'][direction]:
                    V1 = cap_volume(r, p['center'][direction] - s_lower)
                    V2 = cap_volume(r, p['center'][direction] - s_upper)

                    V = V2 - V1

                elif p['center'][direction] < s_lower:
                    V1 = cap_volume(r, s_upper - p['center'][direction])
                    V2 = cap_volume(r, s_lower - p['center'][direction])

                    V = V2 - V1

            if V > 0:
                s['particles'].append(p)
                s['v_particles'] += V

        s['rel_density'] = s['v_particles'] / s['v_solid']

        segments.append(s)
        densities.append(s['rel_density'])

    return densities

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename", required=True, help="Filename without extension")
parser.add_argument("-d", "--divisions", dest="divisions", required=False,  help="Number of divisions", default=10, type=int)
parser.add_argument("-s", "--save", dest="save", required=False,  help="Save plots", action='store_true', default=False)

args = parser.parse_args()

# Filenames
pfname = args.filename + ".particles"
cfname = args.filename + ".contacts"

# Read nodes data first
pdata = np.genfromtxt(pfname, dtype=None, delimiter=',', skip_header=1)

ax1 = plt.subplot(2, 1, 1)
ax1.set_xlabel('r')
ax1.set_ylabel('ratio')

# Read all paticles to the dict
particles = {}
for particle in pdata:

    idx = particle[0]
    x = particle[1]
    y = particle[2]
    z = particle[3]
    r = particle[4]

    particles[idx] = {'center': np.array([x, y, z]), 'radius': r}

angle0xy = 0
angle0xz = 0
angle0yz = 0

nx = np.array([1, 0, 0])
ny = np.array([0, 1, 0])
nz = np.array([0, 0, 1])

coordination_number = {}
for ip, p in particles.items():
    coordination_number[ip] = 0

# Read contacts
cdata = np.genfromtxt(cfname, delimiter=',', dtype=None, names=True)
for row in cdata:
    id1 = row['p1']
    id2 = row['p2']

    p1 = particles[id1]
    p2 = particles[id2]

    v1 = p1['center']
    v2 = p2['center']

    d = v2 - v1
    fac = np.linalg.norm(d)

    cosTxy = np.abs(np.dot(d, nz) / fac)
    cosTxz = np.abs(np.dot(d, ny) / fac)
    cosTyz = np.abs(np.dot(d, nx) / fac)

    Txy = np.arccos(cosTxy)
    Txz = np.arccos(cosTxz)
    Tyz = np.arccos(cosTyz)

    angle0xy += Txy
    angle0xz += Txz
    angle0yz += Tyz

    coordination_number[id1] += 1
    coordination_number[id2] += 1

x_min = 1e9
x_max = -1e9
y_min = 1e9
y_max = -1e9
z_min = 1e9
z_max = -1e9

V_particles = 0
radii = []
for ip, p in particles.items():
    radii.append(p['radius'])

    V_particles += 4./3. * np.pi * p['radius']**3

    x_min = min(x_min, p['center'][0] - p['radius'])
    x_max = max(x_max, p['center'][0] + p['radius'])
    y_min = min(y_min, p['center'][1] - p['radius'])
    y_max = max(y_max, p['center'][1] + p['radius'])
    z_min = min(z_min, p['center'][2] - p['radius'])
    z_max = max(z_max, p['center'][2] + p['radius'])

x_sz = x_max - x_min
y_sz = y_max - y_min
z_sz = z_max - z_min

lims = [
    [x_min, x_max],
    [y_min, y_max],
    [z_min, z_max]
]

V_solid = x_sz * y_sz * z_sz

density = V_particles/V_solid

coord_avg = sum(coordination_number.values()) / len(particles)

radii.sort()
r_min = min(radii)
r_max = max(radii)
r_avg = (r_min + r_max) / 2.
dr = (r_max - r_avg)

n_particles = len(particles)
n_contacts = len(cdata)

print("Average particle radius:     {} +- {} (+- {}%)".format(r_avg, dr, dr/r_avg*100))
print("Average coordination number: {}".format(coord_avg))
print("Initial densification:       {}%".format(density * 100))
print("Packing dimensions:          {} x {} x {}".format(x_sz, y_sz, z_sz))
print("Particles radii interval:    {} ... {}".format(r_min, r_max))
print("Total number of particles:   {}".format(n_particles))
print("Total number of contacts:    {}".format(n_contacts))
print("x dim:                       {} .. {}".format(x_min, x_max))
print("y dim:                       {} .. {}".format(y_min, y_max))
print("z dim:                       {} .. {}".format(z_min, z_max))

# Fit a normal distribution to the data:
mu, std = norm.fit(radii)

# Plot the histogram.
n_bins = 18
counts, bins = np.histogram(radii, bins=n_bins)
bins = to_dia_mkm(bins)
counts_save = np.append(counts, 0)
counts_save = counts_save/n_particles*100
if args.save:
    np.savetxt("size_distribution_hist.csv", np.column_stack((bins, counts_save)))
#ax1.hist(radii, bins=n_bins, weights=counts, density=True, alpha=0.6, color='g')
ax1.hist(bins[:-1], bins, weights=counts, density=True, alpha=0.6, color='g')

r_arr = np.linspace(r_min, r_max, 100)
r_arr = to_dia_mkm(r_arr)
p_arr = norm.pdf(r_arr, to_dia_mkm(mu), to_dia_mkm(std))
ax1.plot(r_arr, p_arr, 'k', linewidth=2)

#p_arr = p_arr/n_particles*100
if args.save:
    np.savetxt("size_distribution_curve.csv", np.column_stack((r_arr, p_arr)))

#ax1.gca().yaxis.set_major_formatter(PercentFormatter(1))

print("")
print("Average contact angles for different planes:")
print("0xy = {}".format(to_degs(angle0xy / n_contacts)))
print("0xz = {}".format(to_degs(angle0xz / n_contacts)))
print("0yz = {}".format(to_degs(angle0yz / n_contacts)))

# Now analyze homogeneity
ax2 = plt.subplot(2, 1, 2)

# Define number of segments
sizes = [x_sz, y_sz, z_sz]
dim_min = np.min(sizes)

divs = [args.divisions]*3

lables = ["dens_x", "dens_y", "dens_z"]
dens = [[]]*3
rels = [[]]*3

for i in range(3):
    divs[i] = int(round(divs[i] * sizes[i]/dim_min))
    dens[i] = distribute_in_segments(particles, lims, divs[i], i)
    rels[i] = np.linspace(0. + 0.5*sizes[i]/divs[i], 1. - 0.5*sizes[i]/divs[i], divs[i])

    ax2.plot(rels[i], dens[i], linewidth=2, label=lables[i])

    if args.save:
        np.savetxt(lables[i] + ".csv", np.column_stack((rels[i], dens[i])))

ax2.grid()
ax2.legend(title='Relative density:')
ax2.set_xlabel('segment')
ax2.set_ylabel('density')

plt.show()