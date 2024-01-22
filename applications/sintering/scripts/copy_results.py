import shutil
import argparse
import glob
import re
import os
import pathlib
import library

parser = argparse.ArgumentParser(description='Mass copy the results:')
parser.add_argument("-s", "--source", type=str, help="Source path, can be a mask", required=True)
parser.add_argument("-d", "--destination", type=str, help="Destination path", required=True)
parser.add_argument("-f", "--file", nargs='+', help="What files to copy", required=False, default="solution.log")
parser.add_argument("-j", "--job-extension", type=str, help="Job file extension", required=False, default="out")
parser.add_argument("-a", "--job-all", action='store_true', help="Copy all job files", required=False, default=False)
parser.add_argument("-c", "--clean", action='store_true', help="Clean destination folder", required=False, default=False)

args = parser.parse_args()

# Get files according to the mask and sort them by number
folders_list = glob.glob(args.source)
folders_list.sort(key=lambda f: int(re.sub('\D', '', f)))

# List of destionation folders, can be a single one
all_dst_folders = []

for folder in folders_list:

    print("Copying data from folder {}".format(folder))

    if not os.path.isdir(folder):
        raise Exception('The provided path mask should capture folders only')
    
    folder_this = pathlib.PurePath(folder).name
    folder_parent = pathlib.PurePath(folder).parent.name
    
    # %0 - immediate parent dir
    # %1 - parent of the parent dir
    output_path = args.destination
    output_path = output_path.replace("%0", folder_this)
    output_path = output_path.replace("%1", folder_parent)

    folder_dst = os.path.dirname(output_path + "dummy.txt")

    if folder_dst not in all_dst_folders:
        all_dst_folders.append(folder_dst)
        if args.clean and os.path.isdir(folder_dst):
            library.clean_folder(folder_dst)
    
    for mask in args.file:
        ff = os.path.join(folder, mask)
        files_list = glob.glob(os.path.join(folder, mask))
        files_list.sort(key=os.path.getmtime)

        for file_src in files_list:
            filename, file_extension = os.path.splitext(file_src)
            file_extension = file_extension[1:]
            base_name = os.path.basename(file_src)

            skip_the_rest = False

            # If we copy all job files, then we do not rename it
            if file_extension == args.job_extension:
                if args.job_all:
                    file_dst = os.path.join(folder_dst, base_name)
                else:
                    parts = base_name.split('.')
                    parts.pop(0)
                    file_dst = output_path + ".".join(parts)

                    # Skip the rest of the job files
                    skip_the_rest = True
            else:
                file_dst = output_path + base_name
            
            pathlib.Path(os.path.dirname(file_dst)).mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_src, file_dst)
            print("  {} -> {}".format(base_name, file_dst))

            if skip_the_rest:
                break
