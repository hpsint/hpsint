### On gold nodes

```
mkdir configurations-new
cd configurations-new
python3 ../../applications/sintering/scripts_performance/configurations.py

qsub job_0.cmd
qsub job_1.cmd
qsub job_2.cmd
qsub job_3.cmd
qsub job_4.cmd
qsub job_5.cmd
qsub job_6.cmd
qsub job_7.cmd
qsub job_8.cmd
qsub job_9.cmd

```

```
mkdir preconditioners-new
cd preconditioners-new
python3 ../../applications/sintering/scripts_performance/preconditioners.py

qsub job_0.cmd
qsub job_1.cmd
qsub job_2.cmd
qsub job_3.cmd
qsub job_4.cmd
qsub job_5.cmd

```

```
mkdir preconditioners-long-new
cd preconditioners-long-new
python3 ../../applications/sintering/scripts_performance/preconditioners.py 1

qsub job_0.cmd
qsub job_1.cmd
qsub job_2.cmd

```

```
array=($(ls -v ../applications/sintering/sintering_cloud_examples/packings_10k/*.cloud))
./applications/sintering/sintering-print-particles "${array[@]}" 
```


### On SuperMUC-NG

```

```


