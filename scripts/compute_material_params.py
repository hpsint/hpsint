#!/usr/bin/env python3
"""
Compute the dimensionless coefficients that actually enter
MobilityScalar/MobilityTensorial::apply_M() (Mvol, Mvap, Msurf, Mgb, L) from
the "Realistic" mobility input parameters (see
Sintering::ProviderRealistic::calculate() in
applications/sintering/include/pf-applications/sintering/mobility.h), and the
energy coefficients (A, B, kappa_c, kappa_p) that enter the free energy
expression (see Sintering::compute_energy_params() in
applications/sintering/src/tools.cc).

Usage:
    python3 compute_material_params.py config.json [--time T [--time T2 ...]]
    python3 compute_material_params.py config.json [--temperature T [--temperature T2 ...]]
    python3 compute_material_params.py --demo

`config.json` is expected to follow the same structure as the application
input files (e.g. applications/sintering/analysis_examples/2d_titanium.json),
i.e. it must contain a "Material" section with "MobilityRealistic" (and,
optionally, "EnergyRealistic") sub-sections, "TimeScale", "LengthScale",
"EnergyScale" and "Temperature", plus a "Geometry" section with
"InterfaceWidth".
"""

from argparse import ArgumentParser

import json
import math


# Physical constants used by ProviderRealistic (mobility.h)
KB = 8.617343e-5  # Boltzmann constant, eV/K
R = 8.314         # Gas constant, J/(mol K)


def parse_temperature(value):
    """Parse a "x0: y0, x1: y1, ..." string (or a {x: y} dict already
    decoded from JSON) into a sorted list of (x, y) pairs, mirroring
    Function1DPiecewise's storage."""

    if isinstance(value, dict):
        pairs = [(float(k), float(v)) for k, v in value.items()]
    else:
        pairs = []
        for token in str(value).split(','):
            token = token.strip()
            if not token:
                continue
            x_str, y_str = token.split(':')
            pairs.append((float(x_str), float(y_str)))

    pairs.sort(key=lambda p: p[0])
    return pairs


def piecewise_linear(x, pairs, extrapolate_linear=False):
    """Re-implementation of Function1DPiecewise<Number>::value() from
    include/pf-applications/numerics/functions.h."""

    if not pairs:
        return 0.0

    if x <= pairs[0][0]:
        if not extrapolate_linear or len(pairs) < 2:
            return pairs[0][1]
        (x1, y1), (x2, y2) = pairs[0], pairs[1]
    elif x >= pairs[-1][0]:
        if not extrapolate_linear or len(pairs) < 2:
            return pairs[-1][1]
        (x1, y1), (x2, y2) = pairs[-2], pairs[-1]
    else:
        i2 = next(i for i, (px, _) in enumerate(pairs) if px >= x)
        i1 = i2 - 1
        x1, y1 = pairs[i1]
        x2, y2 = pairs[i2]

    k = (y1 - y2) / (x1 - x2)
    b = y1 - k * x1
    return k * x + b


def calc_diffusion_coefficient(D0, Q, T, omega_dmls, time_scale,
                               length_scale, energy_scale,
                               arrhenius_factor):
    """Mirrors ProviderRealistic::calc_diffusion_coefficient()."""

    D0_dmls = D0 * time_scale / (length_scale * length_scale)
    D_dmls = D0_dmls * math.exp(-Q / (arrhenius_factor * T))
    M = D_dmls * omega_dmls / (arrhenius_factor * T) * energy_scale

    return M


def calc_gb_mobility_coefficient(D0, Q, T, interface_width, time_scale,
                                 length_scale, energy_scale,
                                 arrhenius_factor):
    """Mirrors ProviderRealistic::calc_gb_mobility_coefficient()."""

    D0_dmls = D0 * time_scale * energy_scale / (length_scale ** 4)
    D_dmls = D0_dmls * math.exp(-Q / (arrhenius_factor * T))
    L = 4.0 / 3.0 * D_dmls / interface_width

    return L


def calc_advection_k(k0, activation_energy, T, arrhenius_unit="Boltzmann"):
    """Mirrors Sintering::ArrheniusEvaluator::eval() from
    applications/sintering/include/pf-applications/sintering/arrhenius.h,
    used to recompute the advection force prefactor K when
    Advection.Qk is non-zero. k0 is used as the Arrhenius
    prefactor and activation_energy as the activation energy."""

    arrhenius_factor = KB if arrhenius_unit == "Boltzmann" else R
    return k0 * math.exp(-activation_energy / (arrhenius_factor * T))


def compute_mobility_coefficients(mobility_realistic, interface_width,
                                  time_scale, length_scale, energy_scale,
                                  T, arrhenius_unit="Boltzmann"):
    """Mirrors ProviderRealistic::calculate(). Returns a dict with the
    Mvol, Mvap, Msurf, Mgb, L values that are stored in Mobility and then
    used by MobilityScalar/MobilityTensorial::apply_M()."""

    omega = float(mobility_realistic["Omega"])
    D_vol0 = float(mobility_realistic["DVol0"])
    D_vap0 = float(mobility_realistic["DVap0"])
    D_surf0 = float(mobility_realistic["DSurf0"])
    D_gb0 = float(mobility_realistic["DGb0"])
    Q_vol = float(mobility_realistic["QVol"])
    Q_vap = float(mobility_realistic["QVap"])
    Q_surf = float(mobility_realistic["QSurf"])
    Q_gb = float(mobility_realistic["QGb"])
    # Not present in the user-supplied snippet -> falls back to the
    # MobilityRealisticData default (0), same as parameters.h.
    D_gb_mob0 = float(mobility_realistic.get("DGbMob0", 0.0))
    Q_gb_mob = float(mobility_realistic.get("QGbMob", 0.0))

    arrhenius_factor = KB if arrhenius_unit == "Boltzmann" else R

    omega_dmls = omega / (length_scale ** 3)

    Mvol = calc_diffusion_coefficient(D_vol0, Q_vol, T, omega_dmls,
                                      time_scale, length_scale, energy_scale,
                                      arrhenius_factor)
    Mvap = calc_diffusion_coefficient(D_vap0, Q_vap, T, omega_dmls,
                                      time_scale, length_scale, energy_scale,
                                      arrhenius_factor)
    Msurf = calc_diffusion_coefficient(D_surf0, Q_surf, T, omega_dmls,
                                       time_scale, length_scale, energy_scale,
                                       arrhenius_factor)
    Mgb = calc_diffusion_coefficient(D_gb0, Q_gb, T, omega_dmls,
                                     time_scale, length_scale, energy_scale,
                                     arrhenius_factor)
    L = calc_gb_mobility_coefficient(D_gb_mob0, Q_gb_mob, T, interface_width,
                                     time_scale, length_scale, energy_scale,
                                     arrhenius_factor)

    return {"Mvol": Mvol, "Mvap": Mvap, "Msurf": Msurf, "Mgb": Mgb, "L": L}


def compute_energy_params(surface_energy, gb_energy, interface_width,
                          length_scale, energy_scale):
    """Mirrors Sintering::compute_energy_params() from
    applications/sintering/src/tools.cc."""

    scaled_gb_energy = gb_energy / energy_scale * length_scale ** 2
    scaled_surface_energy = surface_energy / energy_scale * length_scale ** 2

    kappa_c = 3.0 / 4.0 * (2.0 * scaled_surface_energy -
                          scaled_gb_energy) * interface_width
    kappa_p = 3.0 / 4.0 * scaled_gb_energy * interface_width

    A = (12.0 * scaled_surface_energy -
        7.0 * scaled_gb_energy) / interface_width
    B = scaled_gb_energy / interface_width

    return {"A": A, "B": B, "kappa_c": kappa_c, "kappa_p": kappa_p}


def print_energy(energy):
    print("Energy coefficients (A, B, kappa_c, kappa_p):")
    for key in ("A", "B", "kappa_c", "kappa_p"):
        print(f"  {key:<7} = {energy[key]:.6e}")


def build_cases(times, temperatures, temperature_pairs):
    """Build a list of (label, T) cases from either a list of times (mapped
    to temperature via the piecewise-linear function) or a list of explicit
    temperatures. Falls back to a single case at time=0 if neither is
    given."""

    cases = []
    if temperatures:
        for T in temperatures:
            cases.append((f"T={T:g}", T))
    else:
        for t in (times if times else [0.0]):
            T = piecewise_linear(t, temperature_pairs)
            cases.append((f"t={t:g}", T))

    return cases


def print_mobility_table(cases, mobility_realistic, interface_width,
                         time_scale, length_scale, energy_scale,
                         advection=None):
    """Print a table with one row per (label, T) case, showing the
    temperature, the resulting Mvol, Mvap, Msurf, Mgb, L coefficients, and
    the advection force prefactor K (constant, or Arrhenius-evaluated if
    Advection.Qk is non-zero)."""

    advection = advection or {}
    k0 = float(advection.get("K", 0.0))
    activation_energy = float(advection.get("Qk", 0.0))

    columns = ["case", "T [K]", "Mvol", "Mvap", "Msurf", "Mgb", "L", "k"]
    widths = [12, 10, 12, 12, 12, 12, 12, 12]

    header = "".join(f"{name:<{w}}" for name, w in zip(columns, widths))
    print(header)
    print("-" * len(header))

    for label, T in cases:
        mobility = compute_mobility_coefficients(mobility_realistic,
                                                 interface_width, time_scale,
                                                 length_scale, energy_scale,
                                                 T)
        k = (calc_advection_k(k0, activation_energy, T)
             if activation_energy != 0.0 else k0)
        row = [label, f"{T:g}"] + [f"{mobility[k]:.4e}"
                                   for k in ("Mvol", "Mvap", "Msurf", "Mgb",
                                            "L")] + [f"{k:.4e}"]
        print("".join(f"{value:<{w}}" for value, w in zip(row, widths)))


def run_demo(times, temperatures=None):
    """Reproduces the values with the snippet given by the user, using the
    remaining Material/Geometry parameters taken from
    applications/sintering/analysis_examples/2d_titanium.json as reasonable
    defaults (these are NOT part of "MobilityRealistic" itself, but are
    required to compute the dimensionless mobilities)."""

    mobility_realistic = {
        "Omega": "1.81e-29",
        "DVol0": "19e-8",
        "DVap0": "0.0",
        "DSurf0": "2.53e-3",
        "DGb0": "3.22194e-2",
        "QVol": "1.58366",
        "QVap": "100.0",
        "QSurf": "2.1266554",
        "QGb": "2.2902451",
        "QGbMob": "0",
    }

    advection = {"K": "20", "Qk": "2.2902451"}

    interface_width = 3.0
    time_scale = 1e2
    length_scale = 1e-6
    energy_scale = 1e6
    temperature_pairs = parse_temperature("0: 1573, 5000: 1573")

    cases = build_cases(times, temperatures, temperature_pairs)
    print_mobility_table(cases, mobility_realistic, interface_width,
                         time_scale, length_scale, energy_scale, advection)

    energy_realistic = {"SurfaceEnergy": "1.2795e19",
                        "GrainBoundaryEnergy": "1.148e18"}
    energy = compute_energy_params(
        float(energy_realistic["SurfaceEnergy"]),
        float(energy_realistic["GrainBoundaryEnergy"]),
        interface_width, length_scale, energy_scale)

    print()
    print_energy(energy)


def main():
    parser = ArgumentParser(
        description="Compute the coefficients entering apply_M() and the "
                   "free energy from a Realistic material input file.")
    parser.add_argument("config", nargs="?",
                        help="Path to a json input file containing a "
                             "\"Material\" (and \"Geometry\") section.")
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument("--time", type=float, action="append",
                            dest="times",
                            help="Time at which to evaluate the temperature "
                                 "function (default: 0). Can be given "
                                 "multiple times to print a table.")
    time_group.add_argument("--temperature", type=float, action="append",
                            dest="temperatures",
                            help="Temperature (K) to use directly, "
                                 "bypassing the time -> temperature "
                                 "piecewise function. Can be given "
                                 "multiple times to print a table.")
    parser.add_argument("--demo", action="store_true",
                        help="Run with the snippet given in the request, "
                             "filling in the missing scales/interface "
                             "width/temperature from 2d_titanium.json.")

    args = parser.parse_args()

    if args.demo or args.config is None:
        run_demo(args.times, args.temperatures)
        return

    with open(args.config, "r") as f:
        data = json.load(f)

    material = data["Material"]
    interface_width = float(data["Geometry"]["InterfaceWidth"])
    time_scale = float(material["TimeScale"])
    length_scale = float(material["LengthScale"])
    energy_scale = float(material["EnergyScale"])
    temperature_pairs = parse_temperature(material["Temperature"])
    advection = data.get("Advection", {})

    cases = build_cases(args.times, args.temperatures, temperature_pairs)
    print_mobility_table(cases, material["MobilityRealistic"],
                         interface_width, time_scale, length_scale,
                         energy_scale, advection)

    if "EnergyRealistic" in material:
        energy_realistic = material["EnergyRealistic"]
        energy = compute_energy_params(
            float(energy_realistic["SurfaceEnergy"]),
            float(energy_realistic["GrainBoundaryEnergy"]),
            interface_width, length_scale, energy_scale)

        print()
        print_energy(energy)


if __name__ == "__main__":
    main()
