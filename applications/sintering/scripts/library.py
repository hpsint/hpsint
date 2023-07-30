import colorsys
import glob
import re

def get_hex_colors(N):
    hsv_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in hsv_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

def get_solutions(files, do_print = True):

    files_list = []
    for fm in files:
        current_files_list = glob.glob(fm, recursive=True)
        current_files_list.sort(key=lambda f: int(re.sub('\D', '', f)))
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

    return labels
