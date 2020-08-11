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

## Matrix-free cell integrals

### Allen-Cahn equation (forward Euler)

We solve the following problem:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\left(w,  \eta^{n %2B 1}\right) = \left(w,  \eta^{n} - \Delta t M f_{,\eta}^{n}  \right)  %2B \left(\nabla w,  (-\Delta t M \kappa) \cdot \nabla \eta^{n}\right)" width="600">
</p>

with

- matrix-free cell integral of the **right-hand side**:

```cpp
for (unsigned int cell = cells.first; cell < cells.second; ++cell)
  {
    phi.reinit(cell);
    phi.gather_evaluate(src, true, true, false);
    for (unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        phi.submit_value(phi.get_value(q) - dt * M * df_dphi(phi.get_value(q)), q);
        phi.submit_gradient(-dt * M * kappa * phi.get_gradient(q), q);
      }
    phi.integrate_scatter(true, true, dst);
  }
```

- matrix-free cell integral of the **mass matrix**:

```cpp
for (auto cell = range.first; cell < range.second; ++cell)
  {
    phi.reinit(cell);
    phi.gather_evaluate(src, true, false, false);
    for (unsigned int q = 0; q < phi.n_q_points; ++q)
      phi.submit_value(phi.get_value(q), q); 
    phi.integrate_scatter(true, false, dst);
  }
```

### Allen-Cahn equation (backwards Euler)

We solve the following problem:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\left(w,  \eta^{n %2B 1}   %2B  \Delta t M f_{,\eta}^{n %2B 1}\right) %2B \left(\nabla w,  (\Delta t M \kappa) \cdot \nabla \eta^{n %2B 1}\right) = \left(w,  \eta^{n} \right)  "  width="600">
</p>

### Allen-Cahn equation (2 components - FESystems)

```cpp
for (unsigned int cell = cells.first; cell < cells.second; ++cell)
  {
    phi.reinit(cell);
    phi.gather_evaluate(src, true, true, false);
    for (unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        const auto value    = phi.get_value(q);
        const auto gradient = phi.get_gradient(q);

        Tensor<1, 2, VectorizedArrayType> value_result;
        value_result[0] = value[0] - dt * M * df_dphi(value[0]);
        value_result[1] = value[1] - dt * M * df_dphi(value[1]);

        Tensor<1, 2, Tensor<1, dim, VectorizedArrayType>>
          gradient_result;
        gradient_result[0] = -dt * M * kappa * gradient[0];
        gradient_result[1] = -dt * M * kappa * gradient[1];

        phi.submit_value(value_result, q);
        phi.submit_gradient(gradient_result, q);
      }
    phi.integrate_scatter(true, true, dst);
  }
```

### Allen-Cahn equation (2 components - block vector)

```cpp
for (unsigned int cell = cells.first; cell < cells.second; ++cell)
  {
    phi_1.reinit(cell);
    phi_1.gather_evaluate(src.block(0), true, true, false);

    phi_2.reinit(cell);
    phi_2.gather_evaluate(src.block(1), true, true, false);

    for (unsigned int q = 0; q < phi_1.n_q_points; ++q)
      {
        phi_1.submit_value(phi_1.get_value(q) - dt * M * df_dphi(phi_1.get_value(q)), q);
        phi_2.submit_value(phi_2.get_value(q) - dt * M * df_dphi(phi_2.get_value(q)), q);

        phi_1.submit_gradient(-dt * M * kappa * phi_1.get_gradient(q), q);
        phi_2.submit_gradient(-dt * M * kappa * phi_2.get_gradient(q), q);
      }

    phi_1.integrate_scatter(true, true, dst.block(0));
    phi_2.integrate_scatter(true, true, dst.block(1));
  }
```

### Cahn-Hilliard equation

```cpp
for (unsigned int cell = cells.first; cell < cells.second; ++cell)
  {
    phi.reinit(cell);
    phi.gather_evaluate(src, true, true, false);
    for (unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        const auto value    = phi.get_value(q);
        const auto gradient = phi.get_gradient(q);

        Tensor<1, 2, VectorizedArrayType> value_result;
        value_result[0] = value[0];
        value_result[1] = df_dc(value[0]);

        Tensor<1, 2, Tensor<1, dim, VectorizedArrayType>> gradient_result;
        gradient_result[0] = -dt * M * gradient[1];
        gradient_result[1] = kappa * gradient[0];

        phi.submit_value(value_result, q);
        phi.submit_gradient(gradient_result, q);
      }
    phi.integrate_scatter(true, true, dst);
  }
```
