import os
import subprocess
import pathlib

clouds_path = pathlib.Path(__file__).parent.resolve()

# Adjust executable path accordingly
executable = "./applications/sintering/sintering-print-particles"

cmd_params = [executable]
for cloud in os.listdir(clouds_path):
    if cloud.endswith(".cloud"):
        print("Creating VTU ouput for cloud {}".format(cloud))
        cmd_params.append(os.path.join(clouds_path, cloud))

subprocess.run(cmd_params)
