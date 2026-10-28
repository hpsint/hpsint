#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi

from shapely.geometry import Polygon
from shapely.geometry import box

# User Inputs
N_GRAINS = 50

DOMAIN_SIZE = 100.0

SEED = 42

OUTPUT_FILE = str(N_GRAINS) + "grains.periodic"

def voronoi_finite_polygons_2d(vor, radius=None):

    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)

    all_ridges = {}

    for (p1, p2), (v1, v2) in zip(
            vor.ridge_points,
            vor.ridge_vertices):

        all_ridges.setdefault(
            p1, []).append((p2, v1, v2))

        all_ridges.setdefault(
            p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):

        vertices = vor.regions[region_idx]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]

        new_region = [
            v for v in vertices if v >= 0
        ]

        for p2, v1, v2 in ridges:

            if v2 < 0:
                v1, v2 = v2, v1

            if v1 >= 0:
                continue

            tangent = (
                vor.points[p2]
                - vor.points[p1]
            )

            tangent /= np.linalg.norm(
                tangent
            )

            normal = np.array(
                [-tangent[1], tangent[0]]
            )

            midpoint = (
                vor.points[p1]
                + vor.points[p2]
            ) / 2.0

            direction = (
                np.sign(
                    np.dot(
                        midpoint - center,
                        normal
                    )
                )
                * normal
            )

            far_point = (
                vor.vertices[v2]
                + direction * radius
            )

            new_vertices.append(
                far_point.tolist()
            )

            new_region.append(
                len(new_vertices) - 1
            )

        vs = np.asarray([
            new_vertices[v]
            for v in new_region
        ])

        centroid = vs.mean(axis=0)

        angles = np.arctan2(
            vs[:, 1] - centroid[1],
            vs[:, 0] - centroid[0]
        )

        new_region = np.array(
            new_region
        )[np.argsort(angles)]

        new_regions.append(
            new_region.tolist()
        )

    return (
        new_regions,
        np.asarray(new_vertices)
    )

# Random centres for grains

rng = np.random.default_rng(SEED)

seeds = rng.uniform(
    0.0,
    DOMAIN_SIZE,
    size=(N_GRAINS, 2)
)

# Periodic tiling (3 x 3) for periodic boundary conditions

points = []

for dx in (-DOMAIN_SIZE, 0, DOMAIN_SIZE):
    for dy in (-DOMAIN_SIZE, 0, DOMAIN_SIZE):

        shifted = (
            seeds
            + np.array([dx, dy])
        )

        points.append(shifted)


points = np.vstack(points)

# Create Voronoi diagram

vor = Voronoi(points)

regions, vertices = \
    voronoi_finite_polygons_2d(vor)

domain_box = box(
    0.0,
    0.0,
    DOMAIN_SIZE,
    DOMAIN_SIZE
)

# Selecting central tile only

central_start = 4 * N_GRAINS
central_end = 5 * N_GRAINS

segments = []

segment_id = 1

for point_idx in range(
        central_start,
        central_end):

    grain_id = (
        point_idx
        - central_start
        + 1
    )

    polygon = Polygon(
        vertices[
            regions[point_idx]
        ]
    )

   # create periodic copies

    for dx in (
        -DOMAIN_SIZE, 0, DOMAIN_SIZE
    ):
        for dy in (
            -DOMAIN_SIZE, 0, DOMAIN_SIZE
        ):

            shifted = Polygon(
                [
                    (
                        x + dx,
                        y + dy
                    )
                    for x, y in
                    polygon.exterior.coords
                ]
            )

            piece = shifted.intersection(
                domain_box
            )

            if piece.is_empty:
                continue

            if piece.geom_type == "Polygon":

                segments.append(
                    {
                        "segment_id":
                            segment_id,
                        "grain_id":
                            grain_id,
                        "polygon":
                            piece
                    }
                )

                segment_id += 1

            elif piece.geom_type == \
                    "MultiPolygon":

                for p in piece.geoms:

                    segments.append(
                        {
                            "segment_id":
                                segment_id,
                            "grain_id":
                                grain_id,
                            "polygon":
                                p
                        }
                    )

                    segment_id += 1

# Remove duplicate segments

unique_segments = []

for seg in segments:

    duplicate = False

    for existing in unique_segments:

        if (
            seg["grain_id"]
            == existing["grain_id"]
            and
            seg["polygon"]
            .symmetric_difference(
                existing["polygon"]
            )
            .area
            < 1e-8
        ):
            duplicate = True
            break

    if not duplicate:
        unique_segments.append(seg)

segments = unique_segments

# Global vertex table

vertex_map = {}
vertex_coords = []

for seg in segments:

    ids = []

    coords = list(
        seg["polygon"]
        .exterior.coords[:-1]
    )

    for x, y in coords:

        key = (
            round(x, 10),
            round(y, 10)
        )

        if key not in vertex_map:

            vertex_map[key] = (
                len(vertex_coords) + 1
            )

            vertex_coords.append(
                key
            )

        ids.append(
            vertex_map[key]
        )

    seg["vertex_ids"] = ids

# Format width for output file

max_vertices = max(
    len(seg["vertex_ids"])
    for seg in segments
)

# Write output file

with open(
    OUTPUT_FILE,
    "w"
) as f:

    f.write(
        f"{len(vertex_coords)} "
        f"{max_vertices} "
        f"{len(segments)}\n"
    )

    for vid, (x, y) in enumerate(
            vertex_coords,
            start=1):

        f.write(
            f"{vid:6d}"
            f" {x:15.8e}"
            f" {y:15.8e}\n"
        )

    for seg in segments:

        verts = seg["vertex_ids"]

        padded = (
            verts +
            [0] *
            (
                max_vertices
                - len(verts)
            )
        )

        line = (
            f"{seg['segment_id']:6d} "
            + " ".join(
                f"{v:6d}"
                for v in padded
            )
            + f" {seg['grain_id']:6d}\n"
        )

        f.write(line)

# Plot the microstructure

fig, ax = plt.subplots(
    figsize=(10, 10)
)

cmap = plt.cm.tab20

for seg in segments:

    x, y = \
        seg["polygon"].exterior.xy

    color = cmap(
        seg["grain_id"] % 20
    )

    ax.fill(
        x,
        y,
        facecolor=color,
        edgecolor="black",
        linewidth=0.8
    )

ax.set_xlim(
    0,
    DOMAIN_SIZE
)

ax.set_ylim(
    0,
    DOMAIN_SIZE
)

ax.set_aspect("equal")

plt.tight_layout()
plt.show()

print()
print("Created:")
print(OUTPUT_FILE)
print()
print(
    "Vertices :",
    len(vertex_coords)
)
print(
    "Segments :",
    len(segments)
)
print(
    "Grains   :",
    N_GRAINS
)
print(
    "Max polygon order :",
    max_vertices
)