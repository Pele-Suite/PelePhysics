"""Convert cantera mechanism to C++ files."""

import argparse
import pathlib
import time
from itertools import repeat
from multiprocessing import Pool, cpu_count

import cantera as ct

import ceptr.converter as converter
import ceptr.thermo as cth


def parse_lst_file(lst):
    """Return mechanism paths give a file containing a list of mechanism files."""
    lpath = pathlib.Path(lst)
    fnames = []
    plog_pressures = []
    with open(lst, "r") as f:
        for line in f:
            if not line.startswith("#"):
                parts = line.split("--plog=")
                fnames.append(parts[0])
                if len(parts) > 1:
                    plog_pressures.append(float(parts[1].strip()))
                else:
                    plog_pressures.append(None)
    return [lpath.parents[0] / fn.strip() for fn in fnames], plog_pressures


def parse_qss_lst_file(lst):
    """Return mechanism paths give a file containing a list of qss mechanism files."""
    lpath = pathlib.Path(lst)
    fnames = []
    plog_pressures = []
    with open(lst, "r") as f:
        for line in f:
            if not line.startswith("#"):
                parts = line.split("--plog=")
                fnames.append(parts[0])
                if len(parts) > 1:
                    plog_pressures.append(float(parts[1].strip()))
                else:
                    plog_pressures.append(None)

    mechnames = [lpath.parents[0] / fn.split()[0].strip() for fn in fnames]
    qss_format_inputs = [lpath.parents[0] / fn.split()[1].strip() for fn in fnames]
    return mechnames, qss_format_inputs, plog_pressures


def convert(
    fname,
    jacobian,
    qss_format_input,
    qss_symbolic_jac,
    chemistry,
    gas_name,
    interface_name,
    plog_pressure,
    smooth_thermo=False,
):
    """Convert a mechanism file."""
    print(f"""Converting file {fname}""")

    mechanism_is_homogeneous = chemistry == "homogeneous"
    if not mechanism_is_homogeneous:
        print(f"""\tHomogeneous phase name is '{gas_name}'""")
        print(f"""\tGas-solid interface name is '{interface_name}'""")

    interface = (
        None if mechanism_is_homogeneous else ct.Interface(fname, interface_name)
    )
    mechanism = (
        ct.Solution(fname) if mechanism_is_homogeneous else interface.adjacent[gas_name]
    )

    if smooth_thermo:
        if not mechanism_is_homogeneous:
            raise NotImplementedError(
                "--smooth_thermo does not support heterogeneous chemistry."
            )
        cth.smooth_thermo(mechanism)
        fpath = pathlib.Path(fname)
        smoothed = fpath.with_name(f"{fpath.stem}_smoothed{fpath.suffix}")
        mechanism.write_yaml(smoothed, header=True)
        print(f"Wrote smoothed mechanism to {smoothed}")
    elif mechanism_is_homogeneous:
        cth.check_thermo_continuity(mechanism)

    conv = converter.Converter(
        mechanism,
        interface,
        chemistry,
        jacobian,
        qss_format_input,
        qss_symbolic_jac,
        plog_pressure,
    )
    conv.writer()
    conv.formatter()


def convert_lst(
    lst,
    jacobian,
    qss_format_input,
    qss_symbolic_jac,
    ncpu,
    chemistry,
    gas_name,
    interface_name,
    smooth_thermo=False,
):
    """Convert mechanisms from a file containing a list of directories."""
    mechnames, plog_pressures = parse_lst_file(lst)
    print(f"Using {ncpu} processes")
    with Pool(ncpu) as pool:
        pool.starmap(
            convert,
            zip(
                mechnames,
                repeat(jacobian),
                repeat(qss_format_input),
                repeat(qss_symbolic_jac),
                repeat(chemistry),
                repeat(gas_name),
                repeat(interface_name),
                plog_pressures,
                repeat(smooth_thermo),
            ),
        )


def convert_lst_qss(
    lst,
    jacobian,
    ncpu,
    chemistry,
    gas_name,
    interface_name,
    smooth_thermo=False,
):
    """Convert QSS mechanisms from a file of directories and format input."""
    mechnames, qss_format_inputs, plog_pressures = parse_qss_lst_file(lst)
    print(f"Using {ncpu} processes")
    with Pool(ncpu) as pool:
        pool.starmap(
            convert,
            zip(
                mechnames,
                repeat(jacobian),
                qss_format_inputs,
                repeat(True),
                repeat(chemistry),
                repeat(gas_name),
                repeat(interface_name),
                plog_pressures,
                repeat(smooth_thermo),
            ),
        )


def main():
    """Convert cantera mechanisms to C++ files."""
    start = time.time()
    parser = argparse.ArgumentParser(
        description="Mechanism converter",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--fname", help="Mechanism file", type=str)
    group.add_argument("-l", "--lst", help="Mechanism directory file list", type=str)
    group.add_argument(
        "-lq", "--lst_qss", help="QSS mechanism directory file list", type=str
    )

    parser.add_argument(
        "-c",
        "--chemistry",
        choices=["homogeneous", "heterogeneous"],
        help="Information regarding whether the supplied"
        + " Mechanism file specified Homogeneous or"
        + " heterogeneous chemistry",
        type=str,
        default="homogeneous",
    )

    parser.add_argument(
        "--gas_name",
        type=str,
        default="gas",
        help="Name of the homogeneous phase in the mechanism file",
    )

    parser.add_argument(
        "--interface_name",
        type=str,
        default=None,
        help="Name of the gas-solid interface in the mechanism file",
    )

    parser.add_argument(
        "--qss_format_input",
        help="Input file for QSS Jacobian formatting parameters mechanisms",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "-qsj",
        "--qss_symbolic_jacobian",
        action="store_true",
        help="Compute the QSS Jacobian using symbolic recording",
    )

    parser.add_argument(
        "-nj",
        "--no_jacobian",
        action="store_true",
        help="Do not generate a jacobian",
    )

    parser.add_argument(
        "-n", "--ncpu", help="Number of processes to use", type=int, default=cpu_count()
    )

    parser.add_argument(
        "-p",
        "--plog_pressure",
        help="Pressure in Pascal to evaluate PLOG reactions",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--smooth_thermo",
        action="store_true",
        help="Refit thermodynamic polynomials so cp is C1 and h and s are C2 across"
        + " their temperature breaks, and write the result to <mechanism>_smoothed.yaml",
    )

    args = parser.parse_args()

    if args.chemistry == "heterogeneous":
        assert (
            args.interface_name is not None
        ), f"""Missing --interface_name argument.See 'phases' in {args.fname}"""

    if args.fname:
        convert(
            args.fname,
            not args.no_jacobian,
            args.qss_format_input,
            args.qss_symbolic_jacobian,
            args.chemistry,
            args.gas_name,
            args.interface_name,
            args.plog_pressure,
            args.smooth_thermo,
        )
    elif args.lst:
        convert_lst(
            args.lst,
            not args.no_jacobian,
            args.qss_format_input,
            args.qss_symbolic_jacobian,
            args.ncpu,
            args.chemistry,
            args.gas_name,
            args.interface_name,
            args.smooth_thermo,
        )
    elif args.lst_qss:
        convert_lst_qss(
            args.lst_qss,
            not args.no_jacobian,
            args.ncpu,
            args.chemistry,
            args.gas_name,
            args.interface_name,
            args.smooth_thermo,
        )
    end = time.time()
    print(f"CEPTR run time: {end-start:.2f} s")


if __name__ == "__main__":
    main()
