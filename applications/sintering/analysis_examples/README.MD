### Small 2D benchmrk (5 particles)

```bash
mpirun -np 40 ./applications/sintering/sintering-2D-generic-scalar --cloud ../applications/sintering/sintering_cloud_examples/5particles.cloud ../applications/sintering/analysis_examples/5particles.json
```

#### Advection

as proposed in https://github.com/peterrum/pf-applications/pull/324#issuecomment-1365277045:

```bash
./applications/sintering/sintering-2D-generic-scalar --circle 15 2 ../applications/sintering/analysis_examples/no_advection.json
./applications/sintering/sintering-2D-generic-scalar --circle 15 2 ../applications/sintering/analysis_examples/wang_generic_advection.json
./applications/sintering/sintering-2D-wang-scalar    --circle 15 2 ../applications/sintering/analysis_examples/wang_coupled_advection.json
```

### Wang studies: 2 particles

as proposed in https://github.com/peterrum/pf-applications/issues/402#issue-1548624389

```bash
./applications/sintering/sintering-2D-generic-scalar --circle 15 2 ../applications/sintering/analysis_examples/coupled_wang_studies_0.json
./applications/sintering/sintering-2D-generic-scalar --circle 15 2 ../applications/sintering/analysis_examples/coupled_wang_studies_1.json
./applications/sintering/sintering-2D-wang-scalar --circle 15 2 ../applications/sintering/analysis_examples/coupled_wang_studies_2.json
```

### Wang studies: 5 particles

as proposed in https://github.com/peterrum/pf-applications/issues/402#issuecomment-1396932830

```bash
./applications/sintering/sintering-2D-generic-scalar --cloud ../applications/sintering/sintering_cloud_examples/5particles.cloud ../applications/sintering/analysis_examples/coupled_wang_studies_0.json
./applications/sintering/sintering-2D-generic-scalar --cloud ../applications/sintering/sintering_cloud_examples/5particles.cloud ../applications/sintering/analysis_examples/coupled_wang_studies_1.json
./applications/sintering/sintering-2D-wang-scalar --cloud ../applications/sintering/sintering_cloud_examples/5particles.cloud ../applications/sintering/analysis_examples/coupled_wang_studies_2.json
```

### 49 particle case

```bash
mpirun -np 320 ./applications/sintering/sintering-3D-generic-scalar --cloud ../applications/sintering/sintering_cloud_examples/49particles.cloud ../applications/sintering/analysis_examples/49particles.json
```

Variations with different interface thicknesses `delta_min=0.5, 1.0, 2.0, 3.0, 4.0` (the thickness value is controled by the free energy parameters), as proposed in https://github.com/peterrum/pf-applications/issues/313:

```bash
mpirun -np 320 ./applications/sintering/sintering-3D-generic-scalar --cloud ../applications/sintering/sintering_cloud_examples/49particles.cloud ../applications/sintering/analysis_examples/49particles_delta_min_05.json
mpirun -np 320 ./applications/sintering/sintering-3D-generic-scalar --cloud ../applications/sintering/sintering_cloud_examples/49particles.cloud ../applications/sintering/analysis_examples/49particles_delta_min_10.json
mpirun -np 320 ./applications/sintering/sintering-3D-generic-scalar --cloud ../applications/sintering/sintering_cloud_examples/49particles.cloud ../applications/sintering/analysis_examples/49particles_delta_min_20.json
mpirun -np 320 ./applications/sintering/sintering-3D-generic-scalar --cloud ../applications/sintering/sintering_cloud_examples/49particles.cloud ../applications/sintering/analysis_examples/49particles_delta_min_30.json
mpirun -np 320 ./applications/sintering/sintering-3D-generic-scalar --cloud ../applications/sintering/sintering_cloud_examples/49particles.cloud ../applications/sintering/analysis_examples/49particles_delta_min_40.json
```

... `delta_min=2.0` results in 16MDoFs (comparable to https://github.com/peterrum/pf-applications/issues/308), `delta_min=3.0` in 6MDoFs, and `delta_min=4.0` in 4MDoFs (comparable to https://github.com/peterrum/pf-applications/issues/203).

Variation with `FE_Q_iso_Q1` (to be recompiled!!!):

```bash
mpirun -np 320 ./applications/sintering/sintering-3D-generic-scalar --cloud ../applications/sintering/sintering_cloud_examples/49particles.cloud ../applications/sintering/analysis_examples49particles_iso.json
```

... as propsed in https://github.com/peterrum/pf-applications/pull/329.

Test case to run the simulation up to time `T=100000` when only 2 massive grains are remaining. This case also outputs grains contours and various scalar quantities to control the results:
```
mpirun -np 320 ./applications/sintering/sintering-3D-generic-scalar --cloud ../applications/sintering/sintering_cloud_examples/49particles.cloud ../applications/sintering/analysis_examples/49particles_delta_min_40_ultimate.json
```
