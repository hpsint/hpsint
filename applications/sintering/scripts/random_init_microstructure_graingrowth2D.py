#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt
import argparse

import pyvoro2 as pv

import pyvoro2.planar as pv2

from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch

from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.geometry.polygon import orient
from pyvoro2.viz3d import VizStyle, view_tessellation

# fmt: off
parser = argparse.ArgumentParser(description="Initialize random microstructure for grain growth simulation.")
parser.add_argument("-dim", "--dimension", dest="dimension", required = True, type=int, choices=[2, 3], help="Dimension of the microstructure (e.g., 2 for 2D, 3 for 3D)")
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


def sample_target_measures(
    dimension, domain_measure, rng, quantity, distribution, mean, std, n_grains
):
    """
    Return target grain measures(area in 2D or volume in 3D) that exactly fill domain_measure.

    A distribution ("lognormal", "normal", "uniform") with `mean`/`std` of the chosen quantity must be specified.
    Without `n_grains`, grains are sampled until the domain is filled. The result is uniformly rescaled to sum to
    domain_measure.
    """

    if quantity not in ("radius", "area"):
        raise ValueError("quantity must be 'radius' or 'area'")

    def to_measure(v):
        if quantity == "radius":
            if dimension == 2:
                return np.pi * v**2
            elif dimension == 3:
                return 4.0 / 3.0 * np.pi * v**3
            else:
                raise ValueError("Unsupported dimension")

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
            return 2 * rng.random(n) * mean
        raise ValueError(f"Unknown distribution '{distribution}'")

    # Sample measures based on the specified distribution and quantity
    if n_grains is not None:
        measures = to_measure(draw(n_grains))
    else:
        measures = []
        total = 0.0
        while total < domain_measure:
            a = float(to_measure(draw(1))[0])
            measures.append(a)
            total += a
        # keep the last grain only if it brings the sum closer to the domain
        if len(measures) > 1 and abs(total - measures[-1] - domain_measure) < abs(
            total - domain_measure
        ):
            measures.pop()
        measures = np.array(measures)

    if np.any(measures <= 0.0):
        raise ValueError("All grain sizes must be positive")

    return measures * (domain_measure / measures.sum())


def place_periodic_seeds(
    dimension, target_radii, lengths, rng, spacing_factor, max_attempts_per_grain
):
    """
    Random sequential placement (largest first) with periodic distance d_ij > spacing_factor * (r_i + r_j).
    spacing_factor is relaxed on failure.
    """

    lengths = np.asarray(lengths, dtype=float)
    n = len(target_radii)
    seeds = np.zeros((n, dimension))
    placed = np.zeros(n, dtype=bool)

    for gid in np.argsort(target_radii)[::-1]:

        factor = spacing_factor

        while True:
            cand = rng.uniform(0.0, 1.0, (max_attempts_per_grain, dimension)) * lengths

            if not placed.any():
                seeds[gid] = cand[0]
                break

            d = np.abs(cand[:, None, :] - seeds[placed][None, :, :])
            d = np.minimum(d, lengths - d)
            dist = np.linalg.norm(d, axis=2)

            ok = np.all(
                dist > factor * (target_radii[gid] + target_radii[placed]), axis=1
            )

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


def _convex_hull_mesh(vertices):
    vertices = np.asarray(vertices, dtype=float)
    hull = ConvexHull(vertices)
    hull_vertices = hull.vertices
    vertex_indices = {index: i for i, index in enumerate(hull_vertices)}

    faces = []
    for face, equation in zip(hull.simplices, hull.equations):
        face = [vertex_indices[index] for index in face]
        points = vertices[hull_vertices[face]]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        if np.dot(normal, equation[:3]) < 0:
            face[1], face[2] = face[2], face[1]
        faces.append(face)

    return vertices[hull_vertices], np.asarray(faces, dtype=int), hull


def _polyhedron_hull(cell):
    return _convex_hull_mesh(cell["vertices"])


def _polyhedron_box_intersection(hull, translation, lengths):
    halfspaces = hull.equations.copy()
    halfspaces[:, 3] -= halfspaces[:, :3] @ translation

    box_halfspaces = []
    for axis, length in enumerate(lengths):
        upper = np.zeros(4)
        upper[axis] = 1.0
        upper[3] = -length
        lower = np.zeros(4)
        lower[axis] = -1.0
        box_halfspaces.extend((upper, lower))
    halfspaces = np.vstack((halfspaces, box_halfspaces))

    normals = halfspaces[:, :3]
    center_result = linprog(
        [0.0, 0.0, 0.0, -1.0],
        A_ub=np.column_stack((normals, np.linalg.norm(normals, axis=1))),
        b_ub=-halfspaces[:, 3],
        bounds=[(None, None)] * 3 + [(0.0, None)],
        method="highs",
    )
    if not center_result.success or center_result.x[3] <= 0.0:
        return np.empty((0, 3)), 0.0

    intersection = HalfspaceIntersection(halfspaces, center_result.x[:3])
    intersection_vertices = intersection.intersections
    return intersection_vertices, ConvexHull(intersection_vertices).volume


def _plot_periodic_tessellation_3d(cells, seeds, lengths):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = [
        plt.cm.tab20((grain_id - 1) % 20) for grain_id in range(1, len(seeds) + 1)
    ]

    for cell in cells:
        color = colors[(cell["grain_id"] - 1) % len(colors)]
        triangles = [cell["vertices"][face] for face in cell["faces"]]
        ax.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolors=[color] * len(triangles),
                edgecolors="none",
                linewidths=0,
                alpha=0.55,
            )
        )

    lx, ly, lz = lengths
    for x in (0.0, lx):
        for y in (0.0, ly):
            ax.plot((x, x), (y, y), (0.0, lz), color="black", linewidth=1.2)
    for y in (0.0, ly):
        for z in (0.0, lz):
            ax.plot((0.0, lx), (y, y), (z, z), color="black", linewidth=1.2)
    for x in (0.0, lx):
        for z in (0.0, lz):
            ax.plot((x, x), (0.0, ly), (z, z), color="black", linewidth=1.2)

    ax.scatter(*seeds.T, color="black", s=18, depthshade=False)
    ax.set(xlim=(0.0, lx), ylim=(0.0, ly), zlim=(0.0, lz))
    ax.set_box_aspect(lengths)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xticks((0.0, lx))
    ax.set_yticks((0.0, ly))
    ax.set_zticks((0.0, lz))
    ax.grid(False)
    ax.view_init(elev=24, azim=35)
    ax.legend(
        handles=[
            Patch(facecolor=color, label=f"Grain {grain_id}")
            for grain_id, color in enumerate(colors, start=1)
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig("tessellation_3d.png", dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)


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
            f.write(
                f"{seg['segment_id']:6d} "
                + " ".join(f"{v:6d}" for v in padded)
                + f" {seg['grain_id']:6d}\n"
            )

    return len(vertex_coords), max_vertices


def write_periodic_file_3d(polyhedra, filename, lengths):
    """Write periodic polyhedra in the versioned HPSINT_PERIODIC 3D text format.

    The file contains a magic/version header, domain lengths, then one record per
    polyhedron with its grain ID, lattice shift, vertices, and triangular faces.
    Face vertex indices are one-based and local to their polyhedron.
    """
    with open(filename, "w") as f:
        f.write("HPSINT_PERIODIC 3 1\n")
        f.write(" ".join(f"{length:.17g}" for length in lengths) + "\n")
        f.write(f"{len(polyhedra)}\n")

        for polyhedron in polyhedra:
            vertices = polyhedron["vertices"]
            faces = polyhedron["faces"]
            shift = polyhedron["image_shift"]
            f.write(
                f"{polyhedron['segment_id']} {polyhedron['grain_id']} "
                f"{shift[0]} {shift[1]} {shift[2]} {len(vertices)} {len(faces)}\n"
            )
            for vertex in vertices:
                f.write(" ".join(f"{coordinate:.17g}" for coordinate in vertex) + "\n")
            for face in faces:
                f.write(" ".join(str(int(index) + 1) for index in face) + "\n")

    return sum(len(polyhedron["vertices"]) for polyhedron in polyhedra), sum(
        len(polyhedron["faces"]) for polyhedron in polyhedra
    )


def generate_periodic_voronoi_microstructure(
    dimension,
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
    plot,
):
    """
    Create a periodic 2D or 3D power (Laguerre-Voronoi) microstructure with pyvoro2 whose grain measures follow a prescribed
    radius or measure(area in 2D or volume in 3D) distribution, and optionally write it as a .periodic file for the sintering operator.

    Parameters
    ----------
    dimension : {2, 3}
            Spatial dimension of the microstructure (2D or 3D).
    domain_size : float or (Lx, Ly, Lz)
        Size of the periodic domain [0, Lx] x [0, Ly] x [0, Lz] for 3D, or [0, Lx] x [0, Ly] for 2D.
    quantity : {"radius", "measure"}
        Whether `mean` / `std` refer to equivalent radii or measures.
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
        Output path. Defaults to "<N>grains.periodic" in 2D and
        "<N>grains.periodic3d" in 3D. Pass False to skip writing.
    plot : bool
        Plot 2D microstructures and histograms, or write a 3D tessellation preview.
    """

    lengths = np.broadcast_to(np.asarray(domain_size, dtype=float), (dimension,)).copy()
    domain_measure = lengths.prod()

    rng = np.random.default_rng(seed)

    target_measures = sample_target_measures(
        dimension,
        domain_measure,
        rng,
        quantity=quantity,
        distribution=distribution,
        mean=mean,
        std=std,
        n_grains=n_grains,
    )
    n = len(target_measures)
    if dimension == 2:
        target_radii = np.sqrt(target_measures / np.pi)
    elif dimension == 3:
        target_radii = (3 * target_measures / (4 * np.pi)) ** (1 / 3)
    else:
        raise ValueError("Dimension must be 2 or 3.")

    seeds = place_periodic_seeds(
        dimension,
        target_radii,
        lengths,
        rng,
        spacing_factor=spacing_factor,
        max_attempts_per_grain=max_attempts,
    )

    if dimension == 2:
        domain = pv2.RectangularCell(
            bounds=((0.0, lengths[0]), (0.0, lengths[1])), periodic=(True, True)
        )
    elif dimension == 3:
        domain = pv.OrthorhombicCell(
            bounds=((0.0, lengths[0]), (0.0, lengths[1]), (0.0, lengths[2])),
            periodic=(True, True, True),
        )
    else:
        raise ValueError("Dimension must be 2 or 3.")

    weights = target_radii**2

    if distribution == "random":
        if dimension == 2:
            cells = pv2.compute(
                seeds,
                domain=domain,
                include_empty=True,
                output="cells",
                return_adjacency=False,
                return_edges=False,
            )
            cells = sorted(cells, key=lambda c: c["id"])
            measures = np.array([c.get("area", 0.0) for c in cells])
        elif dimension == 3:
            cells, diagnostics = pv.compute(
                seeds,
                domain=domain,
                output="cells",
                return_faces=True,
                return_face_shifts=True,
                return_diagnostics=True,
                tessellation_check="raise",
                tessellation_require_reciprocity=True,
            )
            print(diagnostics)
            cells = sorted(cells, key=lambda c: c["id"])
            measures = np.array([c.get("volume", 0.0) for c in cells])
    else:
        for it in range(max_attempts + 1):

            if dimension == 2:
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

                measures = np.array([c.get("area", 0.0) for c in cells])

            elif dimension == 3:
                cells, diagnostics = pv.compute(
                    seeds,
                    domain=domain,
                    mode="power",
                    weights=weights,
                    include_empty=True,
                    output="cells",
                    return_adjacency=False,
                    return_faces=True,
                    return_face_shifts=True,
                    return_diagnostics=True,
                    tessellation_check="raise",
                    tessellation_require_reciprocity=True,
                )
                cells = sorted(cells, key=lambda c: c["id"])

                measures = np.array([c.get("volume", 0.0) for c in cells])
            rel_err = np.abs(measures - target_measures) / target_measures

            if rel_err.max() < area_tolerance or it == max_attempts:
                break

            if lloyd:
                for i, c in enumerate(cells):
                    if dimension == 2:
                        if c.get("empty", False) or len(c["vertices"]) < 3:
                            continue
                        cen = _ordered_cell_polygon(c).centroid
                        seeds[i] = np.mod([cen.x, cen.y], lengths)
                    elif dimension == 3:
                        verts = np.array(c["vertices"])
                        cen = verts.mean(axis=0)
                        seeds[i] = np.mod(cen, lengths)

            weights = weights + relaxation * (target_measures - measures) / np.pi

        if any(c.get("empty", False) for c in cells):
            raise RuntimeError(
                "Some grains vanished; try a narrower distribution, more iterations or a smaller relaxation."
            )

        print(
            f"Power diagram: {n} grains, {it} iterations, max rel. area error {rel_err.max():.3e}"
        )
        if dimension == 3:
            print(diagnostics)

    primary, extra = [], []

    if dimension == 2:
        domain_box = box(0.0, 0.0, lengths[0], lengths[1])
        min_overlap = 1e-10 * domain_measure

        # Write every unclipped periodic image of a cell that overlaps the domain.
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

    elif dimension == 3:
        min_overlap = 1e-10 * domain_measure
        preview_cells = []

        # The domain is already periodic; expand images only for the explicit-image file and preview.
        for i, c in enumerate(cells):
            vertices, faces, hull = _polyhedron_hull(c)
            base = -np.floor(vertices.mean(axis=0) / lengths).astype(int)

            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sz in (-1, 0, 1):
                        image_shift = base + np.array([sx, sy, sz])
                        translation = image_shift * lengths

                        clipped_vertices, overlap = _polyhedron_box_intersection(
                            hull, translation, lengths
                        )
                        if overlap < min_overlap:
                            continue

                        clipped_vertices, clipped_faces, _ = _convex_hull_mesh(
                            clipped_vertices
                        )
                        preview_cells.append(
                            {
                                "id": len(preview_cells) + 1,
                                "grain_id": i + 1,
                                "vertices": clipped_vertices,
                                "faces": clipped_faces,
                            }
                        )

                        target = primary if (sx, sy, sz) == (0, 0, 0) else extra
                        target.append(
                            {
                                "grain_id": i + 1,
                                "image_shift": image_shift,
                                "vertices": vertices + translation,
                                "faces": faces,
                            }
                        )

    segments = primary + extra
    for k, seg in enumerate(segments, start=1):
        seg["segment_id"] = k

    if dimension == 3 and plot:
        display_cells = [
            {
                **cell,
                "faces": [{"vertices": face.tolist()} for face in cell["faces"]],
            }
            for cell in preview_cells
        ]
        viewer = view_tessellation(
            display_cells,
            domain=domain,
            show_sites=False,
            show_vertices=False,
            wrap_cells=False,
            style=VizStyle(edge_color="0x777777", edge_line_width=0.6),
        )
        viewer.write_html("tessellation.html")
        _plot_periodic_tessellation_3d(preview_cells, seeds, lengths)

    if output_file is None:
        output_file = (
            f"{n}grains.periodic" if dimension == 2 else f"{n}grains.periodic3d"
        )

    if output_file:
        if dimension == 2:
            n_vertices, max_vertices = write_periodic_file(
                segments, output_file, lengths
            )
            print(
                f"Created {output_file}: {n_vertices} vertices, {len(segments)} segments, "
                f"{n} grains, max polygon order {max_vertices}"
            )
        else:
            n_vertices, n_faces = write_periodic_file_3d(segments, output_file, lengths)
            print(
                f"Created {output_file}: {n_vertices} vertices, {n_faces} triangular faces, "
                f"{len(segments)} polyhedra, {n} grains"
            )

    if plot and dimension == 2:
        if distribution != "random":
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 7))
            cmap = plt.cm.tab20
            for seg in segments:
                x, y = seg["polygon"].exterior.xy
                ax0.fill(
                    x,
                    y,
                    facecolor=cmap(seg["grain_id"] % 20),
                    edgecolor="black",
                    linewidth=0.8,
                )
            ax0.plot(seeds[:, 0], seeds[:, 1], "k.", markersize=3)
            ax0.set_xlim(0, lengths[0])
            ax0.set_ylim(0, lengths[1])
            ax0.set_aspect("equal")

            bins = np.histogram_bin_edges(
                np.concatenate([target_measures, measures]), bins=15
            )
            ax1.hist(
                target_measures,
                bins=bins,
                color="lightskyblue",
                edgecolor="white",
                label="Target area",
            )
            ax1.hist(
                measures,
                bins=bins,
                histtype="step",
                color="darkred",
                linewidth=2,
                label="Actual area",
            )
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
                ax.fill(
                    x,
                    y,
                    facecolor=cmap(seg["grain_id"] % 20),
                    edgecolor="black",
                    linewidth=0.8,
                )
            ax.plot(seeds[:, 0], seeds[:, 1], "k.", markersize=3)
            ax.set_xlim(0, lengths[0])
            ax.set_ylim(0, lengths[1])
            ax.set_aspect("equal")
            plt.tight_layout()
            plt.show()


generate_periodic_voronoi_microstructure(
    dimension=args.dimension,
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
