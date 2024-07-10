# The code is taken from
# https://stackoverflow.com/questions/71104627/how-could-i-speed-up-my-written-python-code-spheres-contact-detection-collisio

import numpy as np
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed

def flatten_neighbours(arr):
    sizes = np.fromiter(map(len, arr), count=len(arr), dtype=np.int64)
    values = np.concatenate(arr)
    return sizes, values

@delayed
def find_neighbours(searched_pts, ref_pts, max_dist):
    balltree = BallTree(ref_pts, leaf_size=16, metric='euclidean')
    res = balltree.query_radius(searched_pts, r=max_dist)
    return flatten_neighbours(res)

def vstack_neighbours(top_infos, bottom_infos):
    top_sizes, top_values = top_infos
    bottom_sizes, bottom_values = bottom_infos
    return np.concatenate([top_sizes, bottom_sizes]), np.concatenate([top_values, bottom_values])

def hstack_neighbours(left_infos, right_infos, offset):
    left_sizes, left_values = left_infos
    right_sizes, right_values = right_infos
    n = left_sizes.size
    out_sizes = np.empty(n, dtype=np.int64)
    out_values = np.empty(left_values.size + right_values.size, dtype=np.int64)
    left_cur, right_cur, out_cur = 0, 0, 0
    right_values += offset
    for i in range(n):
        left, right = left_sizes[i], right_sizes[i]
        full = left + right
        out_values[out_cur:out_cur+left] = left_values[left_cur:left_cur+left]
        out_values[out_cur+left:out_cur+full] = right_values[right_cur:right_cur+right]
        out_sizes[i] = full
        left_cur += left
        right_cur += right
        out_cur += full
    return out_sizes, out_values

def reorder_neighbours(in_sizes, in_values, index, reverse_index):
    n = reverse_index.size
    out_sizes = np.empty_like(in_sizes)
    out_values = np.empty_like(in_values)
    in_offsets = np.empty_like(in_sizes)
    s, cur = 0, 0

    for i in range(n):
        in_offsets[i] = s
        s += in_sizes[i]

    for i in range(n):
        in_ind = reverse_index[i]
        size = in_sizes[in_ind]
        in_offset = in_offsets[in_ind]
        out_sizes[i] = size
        for j in range(size):
            out_values[cur+j] = index[in_values[in_offset+j]]
        cur += size

    return out_sizes, out_values

def small_inplace_sort(arr):
    if len(arr) < 80:
        # Basic insertion sort
        i = 1
        while i < len(arr):
            x = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > x:
                arr[j+1] = arr[j]
                j = j - 1
            arr[j+1] = x
            i += 1
    else:
        arr.sort()

def compute(poss, radii, neighbours_sizes, neighbours_values):
    n, m = neighbours_sizes.size, np.max(neighbours_sizes)

    # Big buffers allocated with the maximum size.
    # Thank to virtual memory, it does not take more memory can actually needed.
    particle_corsp_overlaps = np.empty(neighbours_values.size, dtype=np.float64)
    ends_ind_org = np.empty((neighbours_values.size, 2), dtype=np.float64)

    in_offset = 0
    out_offset = 0

    buff1 = np.empty(m, dtype=np.int64)
    buff2 = np.empty(m, dtype=np.float64)
    buff3 = np.empty(m, dtype=np.float64)

    for particle_idx in range(n):
        size = neighbours_sizes[particle_idx]
        cur = 0

        for i in range(size):
            value = neighbours_values[in_offset+i]
            if value != particle_idx:
                buff1[cur] = value
                cur += 1

        nears_i_ind = buff1[0:cur]
        small_inplace_sort(nears_i_ind)  # Note: bottleneck of this function
        in_offset += size

        if len(nears_i_ind) == 0:
            continue

        x1, y1, z1 = poss[particle_idx]
        cur = 0

        for i in range(len(nears_i_ind)):
            index = nears_i_ind[i]
            x2, y2, z2 = poss[index]
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            contact_check = dist - (radii[index] + radii[particle_idx])
            if contact_check <= 0.0:
                buff2[cur] = contact_check
                buff3[cur] = index
                cur += 1

        particle_corsp_overlaps[out_offset:out_offset+cur] = buff2[0:cur]

        contacts_sec_ind = buff3[0:cur]
        small_inplace_sort(contacts_sec_ind)
        sphere_olps_ind = contacts_sec_ind

        for i in range(cur):
            ends_ind_org[out_offset+i, 0] = particle_idx
            ends_ind_org[out_offset+i, 1] = sphere_olps_ind[i]

        out_offset += cur

    # Truncate the views to their real size
    particle_corsp_overlaps = particle_corsp_overlaps[:out_offset]
    ends_ind_org = ends_ind_org[:out_offset]

    assert len(ends_ind_org) % 2 == 0
    size = len(ends_ind_org)//2
    ends_ind = np.empty((size,2), dtype=np.int64)
    ends_ind_idx = np.empty(size, dtype=np.int64)
    gap = np.empty(size, dtype=np.float64)
    cur = 0

    # Find efficiently duplicates (replace np.unique+np.sort)
    for i in range(len(ends_ind_org)):
        left, right = ends_ind_org[i]
        if left < right:
            ends_ind[cur, 0] = left
            ends_ind[cur, 1] = right
            ends_ind_idx[cur] = i
            gap[cur] = particle_corsp_overlaps[i]
            cur += 1

    return gap, ends_ind, ends_ind_idx, ends_ind_org

def ends_gap(poss, radii):
    assert poss.size >= 1

    # Sort the balls
    index = np.argsort(radii)
    reverse_index = np.empty(index.size, np.int64)
    reverse_index[index] = np.arange(index.size, dtype=np.int64)
    sorted_poss = poss[index]
    sorted_radii = radii[index]

    # Split them in two groups: the small and the big ones
    split_ind = len(radii) * 3 // 4
    small_poss, big_poss = np.split(sorted_poss, [split_ind])
    small_radii, big_radii = np.split(sorted_radii, [split_ind])
    max_small_radii = sorted_radii[max(split_ind, 0)]
    max_big_radii = sorted_radii[-1]

    # Find the neighbours in parallel
    result = Parallel(n_jobs=4, backend='threading')([
        find_neighbours(small_poss, small_poss, small_radii+max_small_radii),
        find_neighbours(small_poss, big_poss,   small_radii+max_big_radii  ),
        find_neighbours(big_poss,   small_poss, big_radii+max_small_radii  ),
        find_neighbours(big_poss,   big_poss,   big_radii+max_big_radii    )
    ])
    small_small_neighbours = result[0]
    small_big_neighbours = result[1]
    big_small_neighbours = result[2]
    big_big_neighbours = result[3]

    # Merge the (segmented) arrays in a big one
    neighbours_sizes, neighbours_values = vstack_neighbours(
        hstack_neighbours(small_small_neighbours, small_big_neighbours, split_ind),
        hstack_neighbours(big_small_neighbours, big_big_neighbours, split_ind)
    )

    # Reverse the indices.
    # Note that the results in `neighbours_values` associated to 
    # `neighbours_sizes[i]` are subsets of `query_radius([poss[i]], r=dia_max)`
    # on a `BallTree(poss)`.
    res = reorder_neighbours(neighbours_sizes, neighbours_values, index, reverse_index)
    neighbours_sizes, neighbours_values = res

    # Finally compute the neighbours with a method similar to the 
    # previous one, but using a much faster optimized code.
    return compute(poss, radii, neighbours_sizes, neighbours_values)