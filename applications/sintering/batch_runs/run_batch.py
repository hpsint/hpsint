import json
import argparse
import os
import copy
import time
import shutil
import re
from collections import abc

def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def traverse(dic, path=None):
    if not path:
        path=[]
    if isinstance(dic,dict):
        for x in dic.keys():
            local_path = path[:]
            local_path.append(x)
            for b in traverse(dic[x], local_path):
                yield b
    else: 
        yield path,dic

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

# Script arguments
parser = argparse.ArgumentParser(description="Run multiple tests")
parser.add_argument("-f", "--file", type=str, help="Batch file to process", required=True)
parser.add_argument("-d", "--debug", action='store_true', help="Debug and print jobs only without submission", required=False, default=False)
parser.add_argument("-c", "--clear", action='store_true', help="Clean folders even we perform only debug prints", required=False, default=False)
parser.add_argument("-a", "--account", type=str, help="User account", required=False)
parser.add_argument("-e", "--email", type=str, help="Email for notification", required=False)
parser.add_argument("-x", "--extra", type=str, help="Extra settings passed to the solver", required=False)
parser.add_argument("-s", "--suffix", type=str, help="Suffix for output folder", required=False)
parser.add_argument("-p", "--partition", type=str, help="HPC partition", required=False)
parser.add_argument("-n", "--nodes", type=str, help="HPC number of nodes", required=False)
parser.add_argument("-t", "--time", type=str, help="HPC max time", required=False)
parser.add_argument("-r", "--restart", type=str, help="Perform case restart", required=False)

# Options for overriding runs
parser.add_argument("-i", "--dimensions", type=str, nargs='+', help="Runs dimensions", required=False, default=[])
parser.add_argument("-y", "--physics", type=str, nargs='+', help="Runs physics", required=False, default=[])
parser.add_argument("-m", "--mobility", type=str, nargs='+', help="Runs mobility", required=False, default=[])

args = parser.parse_args()

# Current directory
cwd = os.path.split(os.getcwd())[1]
cwd_short = cwd[:3]

with open(args.file) as json_data:
    data = json.load(json_data)

    common_options = {}

    # Set default slurm options if any
    default_slurm_options = None
    if "SlurmOptions" in data["Simulation"].keys():
        default_slurm_options = data["Simulation"]["SlurmOptions"]

    # User related info is sensitive so can be defined via command line
    if args.account:
        common_options["--account"] = args.account
    elif "User" in data.keys() and "Account" in data["User"].keys():
        common_options["--account"] = data["User"]["Account"]
    else:
        raise Exception('User account has to be provided')

    if args.email:
        common_options["--mail-user"] = args.email
    elif "User" in data.keys() and "Account" in data["User"].keys():
        common_options["--mail-user"] = data["User"]["Email"]

    # If cluster information is provided then use it
    if args.partition:
        common_options["--partition"] = args.partition
    elif "Cluster" in data.keys() and "Partition" in data["Cluster"].keys():
        common_options["--partition"] = data["Cluster"]["Partition"]
    else:
        raise Exception('HPC partition has to be provided')

    if args.nodes:
        common_options["--nodes"] = args.nodes
    elif "Cluster" in data.keys() and "Nodes" in data["Cluster"].keys():
        common_options["--nodes"] = data["Cluster"]["Nodes"]
    else:
        raise Exception('HPC number of nodes has to be provided')

    if args.time:
        common_options["--time"] = args.time
    elif "Cluster" in data.keys() and "Time" in data["Cluster"].keys():
        common_options["--time"] = data["Cluster"]["Time"]
    else:
        raise Exception('HPC max evaluation time has to be provided')

    # What cases are to run
    runs = {}

    if args.dimensions:
        runs["dimensions"] = args.dimensions
    elif "Runs" in data.keys() and "Dimensions" in data["Runs"].keys():
        runs["dimensions"] = data["Runs"]["Dimensions"]
    else:
        raise Exception('Runs dimensions have to be provided')
    
    if args.physics:
        runs["physics"] = args.physics
    elif "Runs" in data.keys() and "Physics" in data["Runs"].keys():
        runs["physics"] = data["Runs"]["Physics"]
    else:
        raise Exception('Runs physics have to be provided')
    
    if args.mobility:
        runs["mobility"] = args.mobility
    elif "Runs" in data.keys() and "Mobility" in data["Runs"].keys():
        runs["mobility"] = data["Runs"]["Mobility"]
    else:
        raise Exception('Runs mobilities have to be provided')

    # Default study folder
    # Main study root
    study_root = data["Settings"]["StudyRoot"]
    study_output = os.path.join(study_root, "output")
    study_clouds = os.path.join(study_root, "clouds")
    study_settings = os.path.join(study_root, "settings")

    # Default output directory
    if "OutputRoot" in data["Settings"].keys():
        default_output_root = data["Settings"]["OutputRoot"]
    else:
        default_output_root = study_output

    # Executable name
    common_options["executable"] = data["Simulation"]["BuildRoot"]

    # Settings file
    settings_file = data["Settings"]["File"]
    if not(settings_file[0] == "/" or settings_file[0] == "~"):
        settings_file = os.path.join(study_settings, settings_file)
    common_options["settings_file"] = settings_file

    # Extra settings
    common_options["settings_extra"] = ""
    if "Extra" in data["Settings"].keys():
        common_options["settings_extra"] = data["Settings"]["Extra"]

    # Casewise options
    counter = 0
    for case in data["Runs"]["Cases"]:

        # Copy common options
        case_common_options = copy.deepcopy(common_options)

        # Check if it is a string or dict with more data
        if isinstance(case, abc.Mapping):
            simulation_mode_params = case["Name"]

            if "Cluster" in case.keys():
                if "Partition" in case["Cluster"].keys():
                    case_common_options["--partition"] = case["Cluster"]["Partition"]
                if "Nodes" in case["Cluster"].keys():
                    case_common_options["--nodes"] = case["Cluster"]["Nodes"]
                if "Time" in case["Cluster"].keys():
                    case_common_options["--time"] = case["Cluster"]["Time"]
        else:
            simulation_mode_params = case

        # Simulation case to run
        if data["Simulation"]["Mode"] == "--cloud" and not(simulation_mode_params[0] == "/" or simulation_mode_params[0] == "~"):
            simulation_mode_params = os.path.join(study_clouds, simulation_mode_params)

        case_common_options["simulation_case"] = data["Simulation"]["Mode"] + " " + simulation_mode_params

        # Generate output directory name
        default_job_folder = data["Simulation"]["Mode"].split('--', 1)[1] + "_"
        if data["Simulation"]["Mode"] == "--cloud" or data["Simulation"]["Mode"] == "--restart":
            default_job_folder += os.path.splitext(os.path.basename(simulation_mode_params))[0]
        elif data["Simulation"]["Mode"] == "--circle" or data["Simulation"]["Mode"] == "--hypercube":
            default_job_folder += simulation_mode_params.replace(' ', 'x')

        default_job_root = os.path.join(default_output_root, default_job_folder)

        for dim in runs["dimensions"]:
            for phys in runs["physics"]:
                for mobility in runs["mobility"]:

                    # Copy common options
                    case_options = copy.deepcopy(case_common_options)

                    # Possible special options per case
                    special_names = []

                    # Output folder suffix
                    suffix = ""
                    if args.suffix:
                        suffix += "_" + args.suffix

                    # Also build up a meaningful job name
                    job_name = cwd_short + "_" + default_job_folder + suffix + "_" + dim

                    if phys == "generic":
                        physics = "generic"
                        case_options["settings_extra"] += " " + "--Advection.Enable=false"
                        job_name += '_generic'
                        special_names.append("Generic")

                    elif phys == "generic_wang":
                        physics = "generic"
                        case_options["settings_extra"] += " " + "--Advection.Enable=true"
                        job_name += '_genwang'
                        special_names.append("Generic")

                    elif phys == "coupled_wang":
                        physics = "wang"
                        case_options["settings_extra"] += " " + "--Advection.Enable=true"
                        job_name += '_couwang'
                        special_names.append("Coupled")

                    job_name += "_" + mobility[0]
                    case_options["--job-name"] = job_name

                    case_options["executable"] = os.path.join(case_options["executable"], "sintering-{}-{}-{}".format(physics, dim.upper(), mobility))

                    # Other options
                    special_names.append(special_names[0] + dim.upper())
                    special_names.append(special_names[1] + mobility.capitalize())

                    # Check if special options were defined
                    if "Special" in data.keys():
                        for special_name in special_names:
                            if special_name in data["Special"].keys():
                                for x in traverse(data["Special"][special_name]):
                                    opt = ('.'.join(x[0])).encode('ascii', 'ignore').decode("utf-8") 
                                    val = x[1]

                                    if isinstance(val, str):
                                        val = val.encode('ascii', 'ignore').decode("utf-8")
                                    else:
                                        val = str(val)
                                    
                                    case_options["settings_extra"] += " --" + opt + "=" + val

                    # Disable 3D VTK output 
                    if "DisableRegularOutputFor3D" in data["Settings"].keys() and data["Settings"]["DisableRegularOutputFor3D"] == "true" and dim == '3d':
                        case_options["settings_extra"] += " --Output.Regular=false"
                        case_options["settings_extra"] += " --Output.Porosity=false"

                    # Append options defined via cmd extra params, they have the highest priority
                    if args.extra:
                        case_options["settings_extra"] += " " + args.extra

                    # Output directory
                    job_dir = os.path.join(default_job_root, dim + "_" + phys + suffix)
                    if not os.path.isdir(job_dir):
                        try:
                            os.makedirs(job_dir)
                        except OSError:
                            pass
                    case_options["--chdir"] = job_dir

                    # Override simulation case info if we request restart
                    if args.restart:
                        restart_file = args.restart
                        if not(os.path.isabs(restart_file)):
                            restart_file = os.path.join(job_dir, args.restart)

                        # If not a specific file has been provided, try to take the latest one
                        if not(os.path.isfile(restart_file)):

                            pattern = re.escape(args.restart) + r'_\d+_driver'
                            restart_files = [f for f in os.listdir(job_dir) if re.match(pattern, f)]
                            if not restart_files:
                                raise Exception('The provided restart file does not exist')

                            restart_file = ''
                            restart_mtime = 0
                            for restart_candidate in restart_files:
                                c_full_path = os.path.join(job_dir, restart_candidate)
                                c_mtime = os.path.getmtime(c_full_path)
                                if c_mtime > restart_mtime:
                                    restart_mtime = c_mtime
                                    restart_file = c_full_path

                            restart_file = restart_file.replace('_driver', '')

                        case_options["simulation_case"] = "--restart " + restart_file

                    # Print options
                    print("\nRunning case #{}:".format(counter))
                    counter += 1
                    for key, value in case_options.items():
                        print("  {}: {}".format(key.rjust(20), value))

                    # Can we proceed - check if there is something using the folder
                    lock_file = os.path.join(job_dir, "job.lock")
                    print("")
                    if os.path.isfile(lock_file):
                        print("NOTICE: there is another job using this folder -> ", end='')
                        if "IgnoreLocks" in data["Settings"].keys() and data["Settings"]["IgnoreLocks"] == "true":
                            print("WARNING: the lock file is ignored")
                        else:
                            print("ERROR: the job is ignored")
                            continue
                    else:
                        print("NOTICE: no other job is using this folder, we are going to take it")
                    touch(lock_file)
                    print("")

                    # Generate command
                    slurm_params = []
                    for key, value in case_options.items():
                        if key.startswith("--"):
                            slurm_params.append(key + "=" + value)
                    slurm_options = ' '.join(slurm_params)

                    if default_slurm_options:
                        slurm_options = default_slurm_options + " " + slurm_options

                    exec_options = case_options["simulation_case"] + " " + case_options["settings_file"] + " " + case_options["settings_extra"]

                    cmd = data["Simulation"]["CmdTemplate"].format(slurm=slurm_options, executable=case_options["executable"], 
                                                                options=exec_options, lock_file=lock_file)
                    
                    print(cmd)

                    # Clean folders if needed
                    do_clear = "CleanOutputFolder" in data["Settings"].keys() and data["Settings"]["CleanOutputFolder"] == "true"
                    if do_clear and not(args.restart) and (not(args.debug) or args.clear):
                        clean_folder(job_dir)

                    # Run real job after all options are set
                    if not(args.debug):
                        os.system(cmd)

                        # Wait for 3 seconds
                        time.sleep(3)
