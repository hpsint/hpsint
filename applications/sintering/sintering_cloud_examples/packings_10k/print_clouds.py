import os
import subprocess
import pathlib

this_path = pathlib.Path(__file__).parent.resolve()

# Adjust prefix and executable path accordingly
executable = "./applications/sintering/sintering-print-particles"
clouds_path = this_path

cmd_params = [executable]
for cloud in os.listdir(clouds_path):
    if cloud.endswith(".cloud"):
        print("Creating VTU ouput for cloud {}".format(cloud))
        cmd_params.append(os.path.join(clouds_path, cloud))

subprocess.run(cmd_params)