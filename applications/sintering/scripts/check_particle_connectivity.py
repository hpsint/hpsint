"""
check_packing_connectivity.py
─────────────────────────────
Reads a CSV file of spherical particles (columns: x, y, z, r) and:
  1. Builds a contact graph (two spheres are "in contact" when the
     distance between their centres ≤ r_i + r_j + tolerance).
  2. Reports the number of connected components.
  3. If more than one component exists, enlarges the MINIMUM NUMBER of
     particles needed to achieve full connectivity:
       - Computes the MST of the inter-component gap graph.
       - For each MST edge (a bridge between two components), closes
         the gap by enlarging exactly ONE particle (the larger of the
         two candidates), distributing the required radius increase
         entirely onto that one particle.
       - Re-checks connectivity after each bridge is closed, since
         one enlargement may incidentally connect more components.
  4. Writes the modified packing to a new file.

Usage
─────
    python check_packing_connectivity.py particles.csv [options]

Options
───────
    --tolerance TOL     Extra gap allowed for two spheres to be
                        considered "in contact".  Default: 1e-10.

    --output FILE       Path for the output file (enlarged radii).
                        Default: <input_stem>_connected.csv

    --comment CHAR      Comment character in the input file.
                        Default: '#'.

    --sep SEP           Column separator in the input file.
                        Default: ',' (comma).

    --strategy {larger,smaller,split}
                        Which particle(s) to enlarge to close each gap:
                          larger  – enlarge only the larger of the two  [default]
                          smaller – enlarge only the smaller of the two
                          split   – distribute the gap equally between both
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree


# ─────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────

def load_particles(path: str, comment: str = "#", sep: str = ",") -> pd.DataFrame:
    df = pd.read_csv(
        path,
        comment=comment,
        sep=sep,
        header=None,
        names=["x", "y", "z", "r"],
        dtype=float,
        engine="python",
    )
    if df.empty:
        raise ValueError("Input file is empty or contains only comments.")
    if (df["r"] <= 0).any():
        raise ValueError(f"{int((df['r'] <= 0).sum())} particle(s) have non-positive radius.")
    return df


# ─────────────────────────────────────────────────────────────
# Graph helpers
# ─────────────────────────────────────────────────────────────

def build_contact_graph(centres: np.ndarray, radii: np.ndarray, tolerance: float):
    """Sparse adjacency matrix for the contact graph."""
    n = len(radii)
    max_contact = 2.0 * radii.max() + tolerance
    tree = cKDTree(centres)
    pairs = tree.query_pairs(r=max_contact, output_type="ndarray")

    if pairs.size == 0:
        return csr_matrix((n, n), dtype=np.float64)

    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    dist = np.linalg.norm(centres[i_idx] - centres[j_idx], axis=1)
    gap  = dist - (radii[i_idx] + radii[j_idx] + tolerance)
    mask = gap <= 0

    i_c, j_c = i_idx[mask], j_idx[mask]
    data = np.ones(2 * len(i_c), dtype=np.float64)
    rows = np.concatenate([i_c, j_c])
    cols = np.concatenate([j_c, i_c])
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def get_components(centres: np.ndarray, radii: np.ndarray, tolerance: float):
    adj = build_contact_graph(centres, radii, tolerance)
    n_comp, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    return n_comp, labels


# ─────────────────────────────────────────────────────────────
# Inter-component closest-pair search
# ─────────────────────────────────────────────────────────────

def inter_component_gap_graph(
    centres: np.ndarray,
    radii: np.ndarray,
    labels: np.ndarray,
    n_comp: int,
    tolerance: float,
):
    """
    For every pair of components (A, B) find the pair of particles
    (i in A, j in B) with the smallest gap:
        gap = ||c_i - c_j|| - r_i - r_j - tolerance

    Returns a list of dicts, one per component-pair, with keys:
        comp_a, comp_b  – component indices
        gap             – amount of combined radius growth needed (> 0)
        i, j            – particle indices of the closest pair
        dist            – Euclidean distance between their centres
    """
    comp_indices = [np.where(labels == c)[0] for c in range(n_comp)]

    edges = []
    for a in range(n_comp):
        idx_a = comp_indices[a]
        c_a   = centres[idx_a]
        r_a   = radii[idx_a]

        for b in range(a + 1, n_comp):
            idx_b = comp_indices[b]
            c_b   = centres[idx_b]
            r_b   = radii[idx_b]

            tree_b = cKDTree(c_b)
            dists, nn_b = tree_b.query(c_a, k=1)

            gap_candidates = dists - r_a - r_b[nn_b] - tolerance
            best = int(np.argmin(gap_candidates))

            local_i = best
            local_j = int(nn_b[best])

            edges.append(dict(
                comp_a = a,
                comp_b = b,
                gap    = float(gap_candidates[best]),
                i      = int(idx_a[local_i]),
                j      = int(idx_b[local_j]),
                dist   = float(dists[best]),
            ))

    return edges


# ─────────────────────────────────────────────────────────────
# MST over the component graph
# ─────────────────────────────────────────────────────────────

def mst_bridges(edges, n_comp):
    """
    Build a weighted complete graph on n_comp nodes (components),
    weighted by gap, and return its MST edges.
    A MST on n_comp nodes has exactly n_comp-1 edges — the minimum
    set of inter-component bridges needed.
    """
    W = np.full((n_comp, n_comp), fill_value=np.inf)
    best_edge = {}

    for e in edges:
        a, b = e["comp_a"], e["comp_b"]
        key  = (min(a, b), max(a, b))
        if key not in best_edge or e["gap"] < best_edge[key]["gap"]:
            best_edge[key] = e
            W[a, b] = e["gap"]
            W[b, a] = e["gap"]

    W_sparse = csr_matrix(np.triu(W))
    mst = minimum_spanning_tree(W_sparse)
    mst_coo = mst.tocoo()

    bridges = []
    for a, b in zip(mst_coo.row, mst_coo.col):
        key = (min(a, b), max(a, b))
        bridges.append(best_edge[key])

    # Sort bridges by gap descending so larger gaps are closed first
    bridges.sort(key=lambda e: e["gap"], reverse=True)
    return bridges


# ─────────────────────────────────────────────────────────────
# Apply a single bridge fix
# ─────────────────────────────────────────────────────────────

def close_gap(radii: np.ndarray, edge: dict, strategy: str, tolerance: float):
    """
    Enlarge particle(s) to close the gap for one bridge edge.
    Modifies `radii` in-place.
    Returns the set of particle indices that were enlarged.
    """
    i, j   = edge["i"], edge["j"]
    gap    = edge["gap"]

    if gap <= 0:
        return set()

    required = gap

    if strategy == "larger":
        target = i if radii[i] >= radii[j] else j
        radii[target] += required
        return {target}

    elif strategy == "smaller":
        target = i if radii[i] <= radii[j] else j
        radii[target] += required
        return {target}

    elif strategy == "split":
        radii[i] += required / 2.0
        radii[j] += required / 2.0
        return {i, j}

    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")


# ─────────────────────────────────────────────────────────────
# Main repair loop
# ─────────────────────────────────────────────────────────────

def repair_connectivity(
    df: pd.DataFrame,
    tolerance: float = 1e-10,
    strategy: str = "larger",
) -> tuple:
    """
    Iteratively close inter-component gaps using MST bridges until
    the packing is fully connected.

    Returns (modified_df, log) where log is a list of per-step dicts.
    """
    centres = df[["x", "y", "z"]].to_numpy()
    radii   = df["r"].to_numpy().copy()
    log     = []

    iteration = 0
    while True:
        n_comp, labels = get_components(centres, radii, tolerance)
        if n_comp == 1:
            break

        iteration += 1
        print(f"\n  Iteration {iteration}: {n_comp} component(s) — computing bridges …")

        edges   = inter_component_gap_graph(centres, radii, labels, n_comp, tolerance)
        bridges = mst_bridges(edges, n_comp)
        print(f"    MST has {len(bridges)} bridge(s) to close.")

        enlarged_this_iter = set()
        for bridge in bridges:
            i, j    = bridge["i"], bridge["j"]
            dist    = bridge["dist"]
            gap_now = dist - radii[i] - radii[j] - tolerance

            if gap_now <= 0:
                print(f"    Bridge ({i},{j}) already closed by a previous fix — skipping.")
                continue

            bridge_copy       = dict(bridge)
            bridge_copy["gap"] = gap_now

            enlarged = close_gap(radii, bridge_copy, strategy, tolerance)
            enlarged_this_iter |= enlarged

            print(f"    Closed gap {gap_now:.4e} between particles {i} and {j}: "
                  f"enlarged particle(s) {sorted(enlarged)}")
            log.append({
                "iteration"  : iteration,
                "particle_i" : i,
                "particle_j" : j,
                "gap_closed" : gap_now,
                "enlarged"   : sorted(enlarged),
                "new_radii"  : {k: float(radii[k]) for k in enlarged},
            })

        print(f"    Particles enlarged this iteration: "
              f"{len(enlarged_this_iter)} → {sorted(enlarged_this_iter)}")

    df_out = df.copy()
    df_out["r"] = radii
    return df_out, log


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check and minimally repair connectivity of a sphere packing."
    )
    parser.add_argument("input", help="Path to the input CSV file.")
    parser.add_argument(
        "--tolerance", type=float, default=1e-10,
        help="Gap tolerance for contact detection (default: 1e-10).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: <stem>_connected.csv).",
    )
    parser.add_argument(
        "--comment", type=str, default="#",
        help="Comment character in the CSV (default: '#').",
    )
    parser.add_argument(
        "--sep", type=str, default=",",
        help="Column separator in the CSV (default: ',').",
    )
    parser.add_argument(
        "--strategy", choices=["larger", "smaller", "split"], default="larger",
        help=(
            "Which particle(s) to enlarge to close each gap: "
            "'larger' (default) enlarges only the larger sphere, "
            "'smaller' enlarges only the smaller, "
            "'split' distributes the gap equally between both."
        ),
    )
    args = parser.parse_args()

    # ── load ──────────────────────────────────────────────────
    print(f"Loading '{args.input}' …")
    df = load_particles(args.input, comment=args.comment, sep=args.sep)
    n  = len(df)
    print(f"  {n} particles loaded.")

    # ── analyse original packing ──────────────────────────────
    print("\n── Original packing ─────────────────────────────────────")
    centres = df[["x", "y", "z"]].to_numpy()
    radii   = df["r"].to_numpy()
    n_comp, labels = get_components(centres, radii, args.tolerance)
    sizes = np.bincount(labels)

    print(f"  Connected components : {n_comp}")
    print(f"  Largest component    : {sizes.max()} particles "
          f"({100 * sizes.max() / n:.1f} %)")
    print(f"  Component size distribution:")
    unique_sizes, counts = np.unique(sizes, return_counts=True)
    for sz, cnt in sorted(zip(unique_sizes, counts), reverse=True):
        print(f"    {cnt:6d} component(s)  ×  {sz:6d} particle(s)")

    if n_comp == 1:
        print("\n✓ The packing is already fully connected. No enlargement needed.")
        sys.exit(0)

    # ── repair ───────────────────────────────────────────────
    print(f"\n── Repairing with strategy='{args.strategy}' ────────────")
    df_out, log = repair_connectivity(df, tolerance=args.tolerance, strategy=args.strategy)

    # ── verify ───────────────────────────────────────────────
    c2 = df_out[["x", "y", "z"]].to_numpy()
    r2 = df_out["r"].to_numpy()
    n_comp2, _ = get_components(c2, r2, args.tolerance)
    assert n_comp2 == 1, "BUG: packing still disconnected after repair!"

    # ── summary ──────────────────────────────────────────────
    enlarged_particles = sorted({p for step in log for p in step["enlarged"]})
    n_enlarged = len(enlarged_particles)
    total_dr   = (df_out["r"] - df["r"]).sum()
    max_dr_rel = ((df_out["r"] - df["r"]) / df["r"]).max()

    print(f"\n── Summary ──────────────────────────────────────────────")
    print(f"  Components before    : {n_comp}")
    print(f"  Components after     : {n_comp2}  ✓")
    print(f"  Particles enlarged   : {n_enlarged}  (out of {n})")
    print(f"  Particle indices     : {enlarged_particles}")
    print(f"  Total Δr added       : {total_dr:.6e}")
    print(f"  Max relative Δr      : {max_dr_rel * 100:.4f} %")

    # ── write output ─────────────────────────────────────────
    out_path = args.output or str(
        Path(args.input).parent / f"{Path(args.input).stem}_connected{Path(args.input).suffix}"
    )
    df_out.to_csv(out_path, index=False, header="x,y,z,r", float_format="%.15e")
    print(f"  Output written to    : '{out_path}'")


if __name__ == "__main__":
    main()
