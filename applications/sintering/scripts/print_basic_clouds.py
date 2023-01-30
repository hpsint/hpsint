import os
import subprocess
import pathlib

this_path = pathlib.Path(__file__).parent.resolve()

# Adjust prefix and executable path accordingly
executable = "./applications/sintering/sintering-print-particles"
prefix = "../sintering_cloud_examples/"

clouds_path = os.path.join(this_path, prefix)
clouds_names = ["49particles.cloud", "108particles.cloud", "186particles.cloud", "290particles.cloud", "608particles.cloud", "1089particles.cloud"]

cmd_params = [executable]
for cloud in clouds_names:
    cmd_params.append(os.path.join(clouds_path, cloud))

subprocess.run(cmd_params)