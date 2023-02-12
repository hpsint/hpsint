### On gold nodes

```
mkdir preconditioners
cd preconditioners
python3 ../../applications/sintering/scripts_performance/configurations.py

qsub job_0.cmd
qsub job_1.cmd
qsub job_2.cmd
qsub job_3.cmd
qsub job_4.cmd
qsub job_5.cmd
qsub job_6.cmd

```

```
mkdir configurations
cd configurations
python3 ../../applications/sintering/scripts_performance/preconditioners.py

qsub job_0.cmd
qsub job_1.cmd
qsub job_2.cmd
qsub job_3.cmd
qsub job_4.cmd
qsub job_5.cmd
qsub job_6.cmd

```

```
mkdir configurations-long
cd configurations-long
python3 ../../applications/sintering/scripts_performance/preconditioners.py 1

qsub job_0.cmd
qsub job_1.cmd
qsub job_2.cmd

```


### On SuperMUC-NG

```

```


