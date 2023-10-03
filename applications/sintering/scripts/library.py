import colorsys
import glob
import os
import re

def get_hex_colors(N):
    hsv_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in hsv_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

def get_solutions(files, do_print = True):

    if type(files) is not list:
        raise Exception("Variable files={} is not list".format(files))

    files_list = []
    for fm in files:
        current_files_list = glob.glob(fm, recursive=True)
        current_files_list.sort()
        files_list += current_files_list

    files_list = sorted(list(set(files_list)))

    if do_print:
        print("The complete list of files to process:")
        for f in files_list:
            print(f)

    return files_list

def generate_short_labels(files_list):
    min_len = min([len(f) for f in files_list])

    s_start = 0
    s_end = 0
    
    for i in range(min_len):
        string_equal = True
        for f in files_list:
            if not(f[i] == files_list[0][i]):
                string_equal = False
                break

        if string_equal:
            s_start += 1
        else:
            break

    for i in range(min_len):
        string_equal = True
        for f in files_list:
            if not(f[-1-i] == files_list[0][-1-i]):
                string_equal = False
                break

        if string_equal:
            s_end += 1
        else:
            break

    labels = [f[s_start:-s_end] for f in files_list]

    # If there is a single empty label then append everything that is before and/or after os.sep
    if '' in labels:
        labels = []

        do_start = any(os.sep != f[s_start] for f in files_list)
        do_end = any(os.sep != f[-s_end] for f in files_list)

        for f in files_list:

            s_start_current = s_start
            s_end_current = s_end

            if do_start:
                s_start_current = s_start - 1
                while f[s_start_current] is not os.sep and s_start_current > 0:
                    s_start_current -= 1
                if f[s_start_current] is os.sep:
                    s_start_current += 1

            if do_end:
                s_end_current = s_end + 1
                while f[-s_end_current] is not os.sep and s_end_current < 0:
                    s_end_current += 1
                if f[-s_end_current] is os.sep:
                    s_end_current -= 1
        
            labels.append(f[s_start_current:-s_end_current])

    return labels

def get_markers(n_plot, n_points, n_total_markers, available_markers = None):

    if available_markers and n_total_markers > 0:
        n_every = round(n_points / n_total_markers)
        n_every = max(1, n_every)

        m_type = available_markers[n_plot % len(available_markers)]
    else:
        n_every = 1
        m_type = None

    return m_type, n_every
