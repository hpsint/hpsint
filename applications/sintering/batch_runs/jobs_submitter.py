import argparse
import subprocess
import re
import os
import time

def get_current_jobs_num():
    cmd = "squeue -h --me -o '%t' | uniq -c"

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    result = re.findall("(\d+)", out)
    num_jobs = sum([int(x) for x in result])
   
    return num_jobs

parser = argparse.ArgumentParser(description="Jobs submitter")
parser.add_argument("-c", "--cmd", type=str, help="Command to get jobs", required=True)
parser.add_argument("-l", "--limit", type=int, help="Number of active jobs", required=False, default=2)
parser.add_argument("-t", "--timeout", type=int, help="Default timeout", required=False, default=60)

args = parser.parse_args()

proc_jobs = subprocess.Popen(args.cmd, stdout=subprocess.PIPE, shell=True)
(out_jobs, err) = proc_jobs.communicate()
jobs = re.findall("sbatch.*", out_jobs)

n_jobs = len(jobs)

print("")
print("Total number of jobs fetched: {}".format(n_jobs))

while True:
    n_active_jobs = get_current_jobs_num()
    time_fromatted = time.strftime("%d/%m/%Y %I:%M:%S %p", time.localtime())
    print("{}: #running = {}, #queued = {}".format(time_fromatted, n_active_jobs, len(jobs)))

    for i in range(0, args.limit - n_active_jobs):
        if not jobs:
            break

        job_new = jobs.pop(0)

        time_fromatted = time.strftime("%d/%m/%Y %I:%M:%S %p", time.localtime())
        print("{}: submitting a new job".format(time_fromatted))
        print(job_new)

        os.system(job_new)

        time.sleep(3)

    if not jobs:
        print("{}: all {} jobs from the queue have been submitted".format(time_fromatted, n_jobs))
        break

    time.sleep(args.timeout)

print("")
