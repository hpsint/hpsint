# Simple phase-field applications based on deal.II

## Authors (in alphabetic order)

- Peter Munch
- Daniel Paukner 

## Setup and run

Build the applications:
```bash
git clone https://github.com/peterrum/dg-applications.git
mkdir dg-applications-build
cd dg-applications-build
cmake ../dg-applications -DDEAL_II_DIR=../dealii-build
make -j10
```

Run the `cahn_hilliard` application:
```bash
./cahn_hilliard
```
