"""Convert cantera mechanism to C++ files."""

import argparse
import pathlib
import time
from itertools import repeat
from multiprocessing import Pool, cpu_count

import cantera as ct

import ceptr.constants as cc
import ceptr.converter as converter


def parse_lst_lines(lst, n_fields=1, default_pressure=None):
    """Return the entries and PLOG pressures of a list file.

    Blank lines and lines starting with '#' are skipped. Each entry must have
    at least n_fields whitespace-separated fields before an optional
    --plog=<pressure>; an entry without it gets default_pressure.
    """
    entries = []
    plog_pressures = []
    with open(lst, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry, sep, plog = line.partition("--plog=")
            if len(entry.split()) < n_fields:
                raise ValueError(
                    f"{lst}:{lineno}: expected {n_fields} file name(s), got '{line}'"
                )
            pressure = default_pressure
            if sep:
                try:
                    pressure = float(plog)
                except ValueError:
                    raise ValueError(
                        f"{lst}:{lineno}: invalid --plog= pressure '{plog.strip()}'"
                    ) from None
            entries.append(entry.strip())
            plog_pressures.append(pressure)
    return entries, plog_pressures


def parse_lst_file(lst, default_pressure=None):
    """Return mechanism paths give a file containing a list of mechanism files."""
    lpath = pathlib.Path(lst)
    fnames, plog_pressures = parse_lst_lines(lst, default_pressure=default_pressure)
    return [lpath.parents[0] / fn for fn in fnames], plog_pressures


def parse_qss_lst_file(lst, default_pressure=None):
    """Return mechanism paths give a file containing a list of qss mechanism files."""
    lpath = pathlib.Path(lst)
    entries, plog_pressures = parse_lst_lines(lst, 2, default_pressure)
    fields = [entry.split() for entry in entries]
    mechnames = [lpath.parents[0] / f[0] for f in fields]
    qss_format_inputs = [lpath.parents[0] / f[1] for f in fields]
    return mechnames, qss_format_inputs, plog_pressures


def has_plog_reactions(mechanism):
    """Return whether the mechanism has PLOG reactions."""
    return any(
        r.reaction_type == "pressure-dependent-Arrhenius" for r in mechanism.reactions()
    )


def ask_plog_pressure():
    """Ask for the pressure at which to evaluate the PLOG reactions."""
    try:
        return float(
            input(
                "Please specify the pressure, at which you want to evaluate the"
                f" rates in Pascal (1 atm = {cc.Patm_pa} Pa, 1 bar = 1e5 Pa):\n"
            )
        )
    except EOFError:
        raise ValueError(
            "The mechanism has PLOG reactions: give the pressure with"
            " -p/--plog_pressure."
        ) from None


def convert(
    fname,
    jacobian,
    qss_format_input,
    qss_symbolic_jac,
    chemistry,
    gas_name,
    interface_name,
    plog_pressure,
    ask_plog=False,
):
    """Convert a mechanism file.

    With ask_plog, the PLOG pressure is asked for when it is needed and not
    given.
    """
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

    if plog_pressure is None and ask_plog and has_plog_reactions(mechanism):
        plog_pressure = ask_plog_pressure()

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


def convert_many(
    mechnames,
    qss_format_inputs,
    jacobian,
    qss_symbolic_jac,
    ncpu,
    chemistry,
    gas_name,
    interface_name,
    plog_pressures,
):
    """Convert mechanisms in parallel."""
    print(f"Using {ncpu} processes")
    with Pool(ncpu) as pool:
        pool.starmap(
            convert,
            zip(
                mechnames,
                repeat(jacobian),
                qss_format_inputs,
                repeat(qss_symbolic_jac),
                repeat(chemistry),
                repeat(gas_name),
                repeat(interface_name),
                plog_pressures,
            ),
        )


def convert_lst(
    lst,
    jacobian,
    qss_format_input,
    qss_symbolic_jac,
    ncpu,
    chemistry,
    gas_name,
    interface_name,
    plog_pressure=None,
):
    """Convert mechanisms from a file containing a list of directories."""
    mechnames, plog_pressures = parse_lst_file(lst, plog_pressure)
    convert_many(
        mechnames,
        repeat(qss_format_input),
        jacobian,
        qss_symbolic_jac,
        ncpu,
        chemistry,
        gas_name,
        interface_name,
        plog_pressures,
    )


def convert_lst_qss(
    lst,
    jacobian,
    ncpu,
    chemistry,
    gas_name,
    interface_name,
    plog_pressure=None,
):
    """Convert QSS mechanisms from a file of directories and format input."""
    mechnames, qss_format_inputs, plog_pressures = parse_qss_lst_file(
        lst, plog_pressure
    )
    convert_many(
        mechnames,
        qss_format_inputs,
        jacobian,
        True,
        ncpu,
        chemistry,
        gas_name,
        interface_name,
        plog_pressures,
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
            ask_plog=True,
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
            args.plog_pressure,
        )
    elif args.lst_qss:
        convert_lst_qss(
            args.lst_qss,
            not args.no_jacobian,
            args.ncpu,
            args.chemistry,
            args.gas_name,
            args.interface_name,
            args.plog_pressure,
        )
    end = time.time()
    print(f"CEPTR run time: {end-start:.2f} s")


if __name__ == "__main__":
    main()
