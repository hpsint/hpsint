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
