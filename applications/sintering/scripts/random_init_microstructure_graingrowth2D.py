#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt
import argparse

import pyvoro2.planar as pv2

from scipy.spatial import Voronoi

from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.geometry.polygon import orient

# fmt: off
parser = argparse.ArgumentParser(description="Initialize random microstructure for grain growth simulation.")
parser.add_argument("-n","--n_grains", dest="n_grains", required = False,type=int,help="Number of grains")
parser.add_argument("-D","--domain_size", dest="domain_size", required = True, type=float,nargs='+' , help="Size of the domain")
parser.add_argument("-d","--distribution_type", dest="distribution_type", required = False, type=str, default="lognormal", help="Initialization distribution type")
parser.add_argument("-s","--seed", dest="seed", type=int, required = False, default=42, help="Random seed")
parser.add_argument("-q","--quantity", dest="quantity", required = False, type=str, default="radius", help="Quantity type for initial distribution (radius or area)")
parser.add_argument("-m","--mean", dest="mean", required = True, default = 5.0, type=float, help="Mean of the distribution")
parser.add_argument("-v","--std", dest="std", required = False, type=float, default = 0.8, help="Standard deviation of the distribution")
parser.add_argument("-f", "--spacing_factor", dest="spacing_factor", required = False, type=float, default=0.6, help="Spacing factor between grains")
parser.add_argument("-l", "--max_attempts", dest="max_attempts", required = False, type=int, default=200, help="Maximum attempts to place grains")
parser.add_argument("-t", "--tolerance", dest="tolerance", required = False, type=float, default=1e-3, help="Tolerance for grain placement")
parser.add_argument("-r", "--relaxation", dest="relaxation", required = False, type=float, default=0.8, help="Relaxation factor for grain placement")
parser.add_argument("-o", "--output_file", dest="output_file", required = False, type=str, default=None, help="Output file name")
parser.add_argument("-nl","--no-lloyd", dest="lloyd", action="store_false", help="Disable Lloyd relaxation")
parser.add_argument("-np","--no-plot", dest="plot", action="store_false", help="Disable plotting of the microstructure")
args = parser.parse_args()
# fmt: on


def sample_target_areas(domain_area, rng, quantity, distribution, mean, std, n_grains):
    """
    Return target grain areas that exactly fill domain_area.

    A distribution ("lognormal", "normal", "uniform") with `mean`/`std` of the chosen quantity must be specified.
    Without `n_grains`, grains are sampled until the domain is filled. The result is uniformly rescaled to sum to
    domain_area.
    """

    if quantity not in ("radius", "area"):
        raise ValueError("quantity must be 'radius' or 'area'")

    def to_area(v):
        v = np.asarray(v, dtype=float)
        return np.pi * v**2 if quantity == "radius" else v

    def draw(n):
        if distribution == "lognormal":
            s2 = np.log(1.0 + (std / mean) ** 2)
            return rng.lognormal(np.log(mean) - 0.5 * s2, np.sqrt(s2), n)
        if distribution == "normal":
            out = rng.normal(mean, std, n)
            # resample non-positive values (truncated normal)
            while np.any(out <= 0.0):
                bad = out <= 0.0
                out[bad] = rng.normal(mean, std, bad.sum())
            return out
        if distribution == "uniform":
            half = np.sqrt(3.0) * std
            if mean - half <= 0.0:
                raise ValueError("uniform distribution must stay positive")
            return rng.uniform(mean - half, mean + half, n)
        if distribution == "random":
            if mean is None:
                raise ValueError("mean is required for random distribution")
            return rng.random(n) * mean
        raise ValueError(f"Unknown distribution '{distribution}'")

    # Sample areas based on the specified distribution and quantity
    if n_grains is not None:
        areas = to_area(draw(n_grains))
    else:
        areas = []
        total = 0.0
        while total < domain_area:
            a = float(to_area(draw(1))[0])
            areas.append(a)
            total += a
        # keep the last grain only if it brings the sum closer to the domain
        if len(areas) > 1 and abs(total - areas[-1] - domain_area) < abs(total - domain_area):
            areas.pop()
        areas = np.array(areas)

    if np.any(areas <= 0.0):
        raise ValueError("All grain sizes must be positive")

    return areas * (domain_area / areas.sum())


def place_periodic_seeds(target_radii, lengths, rng, spacing_factor, max_attempts_per_grain):
    """
    Random sequential placement (largest first) with periodic distance d_ij > spacing_factor * (r_i + r_j).
    spacing_factor is relaxed on failure.
    """

    lengths = np.asarray(lengths, dtype=float)
    n = len(target_radii)
    seeds = np.zeros((n, 2))
    placed = np.zeros(n, dtype=bool)

    for gid in np.argsort(target_radii)[::-1]:

        factor = spacing_factor

        while True:
            cand = rng.uniform(0.0, 1.0, (max_attempts_per_grain, 2)) * lengths

            if not placed.any():
                seeds[gid] = cand[0]
                break

            d = np.abs(cand[:, None, :] - seeds[placed][None, :, :])
            d = np.minimum(d, lengths - d)
            dist = np.linalg.norm(d, axis=2)

            ok = np.all(dist > factor * (target_radii[gid] + target_radii[placed]), axis=1)

            if ok.any():
                seeds[gid] = cand[np.argmax(ok)]
                break

            factor *= 0.9

        placed[gid] = True

    return seeds


def _ordered_cell_polygon(cell):
    # pyvoro2 returns unordered (unwrapped) vertices; power cells are convex
    v = np.asarray(cell["vertices"], dtype=float)
    c = v.mean(axis=0)
    v = v[np.argsort(np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0]))]
    return Polygon(v)


def write_periodic_file(segments, filename, lengths):
    """
    Write segments in the .periodic format read by InitialValuesMicrostructureVoronoi:
      n_vertices max_vertices_per_segment n_segments
      vertex_id x y
      segment_id v_1 ... v_max (0-padded) grain_id
    """

    lengths = np.asarray(lengths, dtype=float)
    tol = 1e-9 * lengths.max()

    vertex_map = {}
    vertex_coords = []
    segment_vertex_ids = []

    for seg in segments:
        ids = []
        for x, y in seg["polygon"].exterior.coords[:-1]:
            key = (int(round(x / tol)), int(round(y / tol)))

            if key not in vertex_map:
                vertex_map[key] = len(vertex_coords) + 1
                vertex_coords.append((x, y))

            vid = vertex_map[key]
            if not ids or ids[-1] != vid:
                ids.append(vid)

        if len(ids) > 1 and ids[0] == ids[-1]:
            ids.pop()

        segment_vertex_ids.append(ids)

    max_vertices = max(len(ids) for ids in segment_vertex_ids)

    with open(filename, "w") as f:
        f.write(f"{len(vertex_coords)} {max_vertices} {len(segments)}\n")

        for vid, (x, y) in enumerate(vertex_coords, start=1):
            f.write(f"{vid:6d} {x:15.8e} {y:15.8e}\n")

        for seg, ids in zip(segments, segment_vertex_ids):
            padded = ids + [0] * (max_vertices - len(ids))
            f.write(f"{seg['segment_id']:6d} " + " ".join(f"{v:6d}" for v in padded) + f" {seg['grain_id']:6d}\n")

    return len(vertex_coords), max_vertices


def generate_periodic_voronoi_microstructure(
    domain_size,
    quantity,
    distribution,
    mean,
    std,
    n_grains,
    seed,
    spacing_factor,
    max_attempts,
    area_tolerance,
    relaxation,
    lloyd,
    output_file,
    plot):
    """
    Create a periodic 2D power (Laguerre-Voronoi) microstructure with pyvoro2 whose grain areas follow a prescribed
    radius or area distribution, and optionally write it as a .periodic file for the sintering operator.

    Parameters
    ----------
    domain_size : float or (Lx, Ly)
        Size of the periodic domain [0, Lx] x [0, Ly].
    quantity : {"radius", "area"}
        Whether `mean` / `std` refer to equivalent radii or areas.
    distribution : {"lognormal", "normal", "uniform"}
    mean, std : float
        Mean and standard deviation of the chosen quantity.
    n_grains : int, optional
        Fix the number of grains instead of sampling until the domain is full.
    seed : int, optional
        RNG seed.
    spacing_factor : float
        Initial seed separation d_ij > spacing_factor * (r_i + r_j).
    max_attempts, area_tolerance, relaxation :
        Weight iteration controls; stops when max relative area error is below `area_tolerance`.
    lloyd : bool
        Move seeds to cell centroids each iteration (more regular grains).
    output_file : str, optional
        Path of the .periodic file to write. Default "<N>grains.periodic". Pass False to skip writing.
    plot : bool
        Plot the microstructure and target/actual area histograms.
    """

    lengths = np.broadcast_to(np.asarray(domain_size, dtype=float), (2,)).copy()
    domain_area = lengths.prod()

    rng = np.random.default_rng(seed)

    target_areas = sample_target_areas(
        domain_area, rng, quantity=quantity, distribution=distribution, mean=mean, std=std, n_grains=n_grains
    )
    n = len(target_areas)
    target_radii = np.sqrt(target_areas / np.pi)

    seeds = place_periodic_seeds(target_radii, lengths, rng, spacing_factor=spacing_factor,max_attempts_per_grain=max_attempts)

    domain = pv2.RectangularCell(bounds=((0.0, lengths[0]), (0.0, lengths[1])), periodic=(True, True))

    weights = target_radii**2

    if distribution == "random":
        cells = pv2.compute(
            seeds, domain=domain, include_empty=True, output="cells", return_adjacency=False, return_edges=False
        )
        cells = sorted(cells, key=lambda c: c["id"])
        areas = np.array([c.get("area", 0.0) for c in cells])
    else:
        for it in range(max_attempts + 1):

            cells = pv2.compute(
                seeds,
                domain=domain,
                mode="power",
                weights=weights,
                include_empty=True,
                output="cells",
                return_adjacency=False,
                return_edges=False,
            )
            cells = sorted(cells, key=lambda c: c["id"])

            areas = np.array([c.get("area", 0.0) for c in cells])
            rel_err = np.abs(areas - target_areas) / target_areas

            if rel_err.max() < area_tolerance or it == max_attempts:
                break

            if lloyd:
                for i, c in enumerate(cells):
                    if c.get("empty", False) or len(c["vertices"]) < 3:
                        continue
                    cen = _ordered_cell_polygon(c).centroid
                    seeds[i] = np.mod([cen.x, cen.y], lengths)

            weights = weights + relaxation * (target_areas - areas) / np.pi

        if any(c.get("empty", False) for c in cells):
            raise RuntimeError(
                "Some grains vanished; try a narrower distribution, more iterations or a smaller relaxation."
            )

        print(f"Power diagram: {n} grains, {it} iterations, max rel. area error {rel_err.max():.3e}")

    # Write every unclipped periodic image of a cell that overlaps the domain
    domain_box = box(0.0, 0.0, lengths[0], lengths[1])
    min_overlap = 1e-10 * domain_area

    primary, extra = [], []

    for i, c in enumerate(cells):
        poly = orient(_ordered_cell_polygon(c), 1.0)
        coords = np.asarray(poly.exterior.coords)

        # primary image: the one whose centroid lies inside [0, L)
        cen = np.array([poly.centroid.x, poly.centroid.y])
        base = -np.floor(cen / lengths)

        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                shift = base + np.array([sx, sy])
                shifted = Polygon(coords + shift * lengths)

                if shifted.intersection(domain_box).area < min_overlap:
                    continue

                target = primary if (sx, sy) == (0, 0) else extra
                target.append({"grain_id": i + 1, "polygon": shifted})

    segments = primary + extra
    for k, seg in enumerate(segments, start=1):
        seg["segment_id"] = k

    if output_file is None:
        output_file = f"{n}grains.periodic"

    if output_file:
        n_vertices, max_vertices = write_periodic_file(segments, output_file, lengths)
        print(
            f"Created {output_file}: {n_vertices} vertices, {len(segments)} segments, {n} grains, max polygon order {max_vertices}"
        )

    if plot:
        if distribution != "random":
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 7))
            cmap = plt.cm.tab20
            for seg in segments:
                x, y = seg["polygon"].exterior.xy
                ax0.fill(x, y, facecolor=cmap(seg["grain_id"] % 20), edgecolor="black", linewidth=0.8)
            ax0.plot(seeds[:, 0], seeds[:, 1], "k.", markersize=3)
            ax0.set_xlim(0, lengths[0])
            ax0.set_ylim(0, lengths[1])
            ax0.set_aspect("equal")

            bins = np.histogram_bin_edges(np.concatenate([target_areas, areas]), bins=15)
            ax1.hist(target_areas, bins=bins, color="lightskyblue", edgecolor="white", label="Target area")
            ax1.hist(areas, bins=bins, histtype="step", color="darkred", linewidth=2, label="Actual area")
            ax1.set_xlabel("Area")
            ax1.set_ylabel("Count")
            ax1.legend()

            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=(7, 7))
            cmap = plt.cm.tab20
            for seg in segments:
                x, y = seg["polygon"].exterior.xy
                ax.fill(x, y, facecolor=cmap(seg["grain_id"] % 20), edgecolor="black", linewidth=0.8)
            ax.plot(seeds[:, 0], seeds[:, 1], "k.", markersize=3)
            ax.set_xlim(0, lengths[0])
            ax.set_ylim(0, lengths[1])
            ax.set_aspect("equal")
            plt.tight_layout()
            plt.show()

generate_periodic_voronoi_microstructure(
    domain_size=args.domain_size,
    quantity=args.quantity,
    distribution=args.distribution_type,
    mean=args.mean,
    std=args.std,
    n_grains=args.n_grains,
    seed=args.seed,
    spacing_factor=args.spacing_factor,
    max_attempts=args.max_attempts,
    area_tolerance=args.tolerance,
    relaxation=args.relaxation,
    lloyd=args.lloyd,
    output_file=args.output_file,
    plot=args.plot,
)
