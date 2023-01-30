import os
import sys
import pathlib

this_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(this_path, '../../scripts'))

from extract_subpacking import extract

# settings
scale = 1e3
limits = [2.6e-4, 3.2e-4, 4.0e-4, 4.5e-4, 5.5e-4, 6.5e-4, 9.2e-4, 1.15e-3, 1e-1]
packing = '10245.particles'

packing_path = os.path.join(this_path, packing)
save_path = this_path

for upper in limits:
    lims = [-0.1, upper]
    extract(packing_path, scale, lims, lims, lims, save_path)
