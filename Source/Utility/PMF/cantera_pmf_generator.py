"""Solve a 1D premixed flame in Cantera and save in Pele-readable format."""

###############################################################
# ADIABATIC_FLAME - A freely-propagating, premixed flat flame
#                   of fuel in AIR (O2 + N2)
###############################################################

# import :

import argparse
import csv
import os
import re

import numpy as np
import yaml
from cantera import Solution, FreeFlame

#################################################################
# Parse arguments
#################################################################
parser = argparse.ArgumentParser(
    prog="Cantera PMF Generator",
    description="Use Cantera to solve a 1D premixed flame and save in a Pele-readable format",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    fromfile_prefix_chars='@',
)

parser.add_argument(
    "-m",
    "--mechanism",
    default="drm19",
    help="Name of PelePhysics mechanism from Mechanisms",
)
parser.add_argument(
    "-pp", "--pp_home", default="../../../", help="Path to PelePhysics directory"
)
parser.add_argument(
    "-f", "--fuel", default="CH4:1", help="Fuel stream mole-basis Cantera composition"
)
parser.add_argument(
    "-ox",
    "--oxidizer",
    default="O2:1, N2:3.76",
    help="Oxidizer stream mole-basis Cantera composition",
)
parser.add_argument(
    "-Y",
    "--massfrac",
    default=None,
    help="String of mass fractions (overrides -f and -ox options)",
)
parser.add_argument(
    "-T",
    "--temperature",
    default=300,
    type=float,
    help="Unburned mixture temperature [K]",
)
parser.add_argument(
    "-p", "--pressure", default=101325, type=float, help="Pressure [Pa]"
)
parser.add_argument("-phi", "--phi", default=1.0, type=float, help="Equivalence Ratio")
parser.add_argument("-d", "--domain", default=0.04, type=float, help="Domain width [m]")
parser.add_argument("-v", "--verbose", default=1, type=int, help="Verbosity level")
parser.add_argument(
    "-o",
    "--output",
    default=None,
    help=(
        "Path to directory where flames will be saved, if not specified files are saved to PelePhysics/Mechanisms/<mechanism>/PMFs/"
    ),
)
args = parser.parse_args()

#################################################################
# Prepare your run
#################################################################
# Parameter values :

# Mixture
mechanism = args.mechanism
fuel_species = args.fuel
ox_species = args.oxidizer
Y_species = args.massfrac

# General
p = args.pressure  # pressure [Pa]
tin = args.temperature  # unburned gas temperature [K]
phi = args.phi  # Eq. ratio [-]
transport = "Mix"  # Cantera transport model

# Refined grid at inlet and outlet, 6 points in x-direction :
domain_size = args.domain  # Domain size [m]
initial_grid = (
    domain_size * np.array([0.0, 0.001, 0.01, 0.02, 0.029, 0.03], "d") / 0.03
)  # m

# Set tolerance properties
tol_ss = [1.0e-8, 1.0e-9]  # [rtol atol] for steady-state problem
tol_ts = [1.0e-8, 1.0e-9]  # [rtol atol] for time stepping
loglevel = args.verbose  # amount of diagnostic output
refine_grid = True  # True to enable refinement

# Print information
label_pre = "pmf-" + mechanism
if Y_species is not None:
    num_input_species = len([item for item in Y_species.split(',') if ':' in item])
    label = "Y" + str(num_input_species) + "_T" + str(tin) + "_P" + str(p)
else:
    label = fuel_species.split(":")[0] + "_PHI" + str(phi) + "_T" + str(tin) + "_P" + str(p)

#################
# Find mechanism in PelePhysics
pp_path = os.path.join(args.pp_home, "Mechanisms")
if args.output is None:
    outdir = os.path.join(pp_path, mechanism, "PMFs")
else:
    outdir = args.output
if not os.path.exists(outdir):
    os.makedirs(outdir)

if not (os.path.exists(pp_path)):
    raise RuntimeError("Invalid path to PelePhysics: " + args.pp_home)

mech_paths = [
    re.sub(r'\s--plog=\S+', '', name).strip()
    for name in open(os.path.join(pp_path, "list_mech")).readlines()
    if not name.startswith("#")
]
mech_names = [name.split("/")[0] for name in mech_paths]
mech_paths = dict(zip(mech_names, mech_paths))

# QSS: we will solve with skeletal mechanism, then eliminate QSS species
qss_data = [
    name.split()
    for name in open(os.path.join(pp_path, "list_qss_mech")).readlines()
    if not name.startswith("#")
]
qss_names = [name[0].split("/")[0] for name in qss_data]
qss_paths = dict(zip(qss_names, [name[2] for name in qss_data]))
qss_nonqss = dict(zip(qss_names, [name[3] for name in qss_data]))

if mechanism in mech_names:
    mech_has_qssa = False
    mech_path = mech_paths[mechanism]
elif mechanism in qss_names:
    mech_has_qssa = True
    mech_path = qss_paths[mechanism]
    with open(os.path.join(pp_path, qss_nonqss[mechanism])) as f:
        nonqss_spec_list = yaml.safe_load(f)["species"]
else:
    raise RuntimeError(
        "Requested mechanism ("
        + mechanism
        + ") found in neither list_mech or list_qssa_mech"
    )
mech_path = os.path.join(pp_path, mech_path)

#################
# Create and Run Flame:

# Set gas state to that of the unburned gas
gas = Solution(mech_path, "gas")
if Y_species is None:
    gas.TP = tin, p
    gas.set_equivalence_ratio(phi, fuel_species, ox_species, basis="mole")
else:
    gas.TPY = tin, p, Y_species
    print("\nMass fractions read into Cantera:")
    for spec, massfrac in zip(gas.species_names, gas.Y):
        print(f"  {spec}: {massfrac}")

# Create the free laminar premixed flame
f = FreeFlame(gas, initial_grid)

f.flame.set_steady_tolerances(default=tol_ss)
f.flame.set_transient_tolerances(default=tol_ts)

f.transport_model = transport

# No energy for starters
f.energy_enabled = False

# Refinement criteria
f.set_refine_criteria(ratio=7.0, slope=1, curve=1)

# Max number of times the Jacobian will be used before it must be re-evaluated
f.set_max_jac_age(10, 10)

# Set time steps whenever Newton convergence fails
f.set_time_step(1.0e-06, [1, 2, 5, 10])  # s
f.max_time_step_count = 3000
f.max_grid_points = 1000

# Calculation
f.solve(loglevel, refine_grid)

#################
# Second flame:

# Energy equation enabled
f.energy_enabled = True

# Refinement criteria when energy equation is enabled
f.set_refine_criteria(ratio=5.0, slope=0.5, curve=0.5)

# Calculation
f.solve(loglevel, refine_grid)

#################
# Third flame and so on ...:
f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)

f.solve(loglevel, refine_grid)

##################
# Fourth flame and so on ...:
f.set_refine_criteria(ratio=2.0, slope=0.05, curve=0.05, prune=0.01)

f.solve(loglevel, refine_grid)

##################
# Fifth flame and so on ...
f.set_refine_criteria(ratio=2.0, slope=0.02, curve=0.02, prune=0.01)

f.solve(loglevel, refine_grid)

print("mixture averaged flamespeed = ", f.velocity[0])

#################################################################
# Save your results
#################################################################

# Always save with Mole Fractions of all species and CGS units
nz = f.flame.n_points
csv_file = os.path.join(outdir, str(label_pre + "-" + label + "-X.dat"))
with open(csv_file, "w") as outfile:
    writer = csv.writer(
        outfile, delimiter=" ", quotechar=" ", quoting=csv.QUOTE_MINIMAL
    )
    writer.writerow(
        ['VARIABLES = "X" "temp" "u" "rho"']
        + ['"X_' + x + '"' for x in gas.species_names]
    )
    writer.writerow(["ZONE I=" + str(nz) + " FORMAT=POINT" + " SPECFORMAT=MOLE"])
    for kk in range(len(f.grid)):
        f.set_gas_state(kk)
        writer.writerow(
            [f.grid[kk] * 1e2, gas.T, f.velocity[kk] * 1e2, gas.density * 1e-3]
            + list(gas.X)
        )

# QSS Mechanisms: save again with QSS species removed and others renormalized to unity sum
if mech_has_qssa:
    csv_file = os.path.join(
        outdir, str(label_pre + "-" + label + "-X-qssa-removed.dat")
    )
    with open(csv_file, "w") as outfile:
        writer = csv.writer(
            outfile, delimiter=" ", quotechar=" ", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(
            ['VARIABLES = "X" "temp" "u" "rho"']
            + ['"X_' + x + '"' for x in gas.species_names if x in nonqss_spec_list]
        )
        writer.writerow(["ZONE I=" + str(nz) + " FORMAT=POINT" + " SPECFORMAT=MOLE"])
        max_total_qss = 0.0
        for kk in range(len(f.grid)):
            f.set_gas_state(kk)
            zeroqss_mole_fracs = np.array(
                [
                    X if (gas.species_name(ii) in nonqss_spec_list) else 0.0
                    for ii, X in enumerate(gas.X)
                ]
            )
            total_qss = 1.0 - np.sum(zeroqss_mole_fracs)
            max_total_qss = max(max_total_qss, total_qss)
            # this will modify rho so that it is consistent with the right T/p with the modified mole fractions
            gas.TPX = gas.T, gas.P, zeroqss_mole_fracs
            writer.writerow(
                [f.grid[kk] * 1e2, gas.T, f.velocity[kk] * 1e2, gas.density * 1e-3]
                + [
                    X
                    for ii, X in enumerate(gas.X)
                    if (gas.species_name(ii) in nonqss_spec_list)
                ]
            )
    print("Maximum local total mole fraction of QSS species: ", max_total_qss)
