import argparse
import subprocess
import re
import os
import time

def get_current_jobs_num(partition):
    cmd = "squeue -h --me -o '%t %P' | uniq -c"

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    # Python 3 returns utf-8 output
    try:
        out = out.decode('utf-8')
    except:
        pass

    result = re.findall("(\d+) ([A-Z]+) (\w+)", out)
    num_jobs = 0
    for tpl in result:
        if tpl[2] == partition or partition == '':
            num_jobs += int(tpl[0])
   
    return num_jobs

parser = argparse.ArgumentParser(description="Jobs submitter")
parser.add_argument("-c", "--cmd", type=str, help="Command to get jobs", required=True)
parser.add_argument("-l", "--limit", type=int, help="Number of active jobs", required=False, default=2)
parser.add_argument("-t", "--timeout", type=int, help="Default timeout", required=False, default=60)
parser.add_argument("-p", "--partition", type=str, help="Partition", required=False, default='test')

args = parser.parse_args()

proc_jobs = subprocess.Popen(args.cmd, stdout=subprocess.PIPE, shell=True)
(out_jobs, err) = proc_jobs.communicate()

# Python 3 returns utf-8 output
try:
    out_jobs = out_jobs.decode('utf-8')
except:
    pass

jobs = re.findall("sbatch.*", out_jobs)

n_jobs = len(jobs)

print("")
print("Total number of jobs fetched: {}".format(n_jobs))

while True:
    n_active_jobs = get_current_jobs_num(args.partition)
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
