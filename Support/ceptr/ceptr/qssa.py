"""Generate QSSA chemistry file."""

import argparse
import pathlib
import time

import cantera as ct
import yaml

import ceptr.qssa_reduction as cqr
import ceptr.reaction_info as cri


def process_qss(fname, nqssa, visualize, method):
    """Apply QSSA reduction to single Cantera yaml file."""
    # Load mechanism
    mechanism = ct.Solution(fname)
    reaction_info = cri.sort_reactions(mechanism, interface=None)
    mechpath = pathlib.Path(mechanism.source)
    qssaname = mechpath.parents[0] / "qssa.yaml"

    # Species
    with open(nqssa) as f:
        f_non_qssa_species = yaml.safe_load(f)
        # Make sure the species are not interpreted as boolean
        if (
            False in f_non_qssa_species["species"]
            or True in f_non_qssa_species["species"]
        ):
            raise ValueError(
                "Some species in non qssa list interpreted as Boolean."
                "Use quotation marks to avoid this issue."
            )
        non_qssa_species = f_non_qssa_species["species"]
    all_species = mechanism.species_names
    qssa_species = sorted(list(set(all_species) - set(non_qssa_species)))
    # Visualize
    if visualize:
        cqr.visualize_qssa(mechanism, reaction_info, qssa_species)

    forward_to_remove = []
    if method == 0:
        qssa_species = cqr.remove_quadratic_method_0(mechanism, qssa_species)
        reactions_to_keep = mechanism.reactions()
    elif method == 1 or method == 2:
        candidates_for_removal = cqr.identify_removals(
            mechanism, reaction_info, qssa_species
        )

        if method == 1:
            reactions_to_keep = cqr.remove_quadratic_method_1(
                mechanism, reaction_info, candidates_for_removal
            )
        elif method == 2:
            (
                reactions_to_keep,
                forward_to_remove,
            ) = cqr.remove_quadratic_method_2(
                mechanism, reaction_info, candidates_for_removal
            )

    qssa = ct.Solution(
        name=f"{mechanism.name}-qssa",
        thermo=mechanism.thermo_model,
        kinetics=mechanism.kinetics_model,
        transport_model=mechanism.transport_model,
        species=mechanism.species(),
        reactions=reactions_to_keep,
    )

    if cqr.qssa_coupling(qssa, qssa_species, forward_to_remove):
        raise ValueError("Failure to generate QSSA mechanism.")

    qssa.update_user_header({"description": f"QSSA of {mechanism.name}"})
    qssa.update_user_data(
        {"qssa_species": qssa_species, "n_qssa_species": len(qssa_species)}
    )
    forward_to_remove_idx = []
    if forward_to_remove:
        for fr in forward_to_remove:
            for idx, reaction in enumerate(qssa.reactions()):
                if fr.equation == reaction.equation:
                    forward_to_remove_idx.append(idx)
                    if not fr.duplicate:
                        break
        # Remove duplicates
        forward_to_remove_idx = list(set(forward_to_remove_idx))
        forward_to_remove_idx.sort()
        qssa.update_user_data({"forward_to_remove_idx": forward_to_remove_idx})
    qssa.write_yaml(qssaname, header=True)

    # Summarize
    for idx in forward_to_remove_idx:
        print(f"Forward reaction to be removed: {qssa.reaction(idx)}")

    mechanism = ct.Solution(fname)  # reread
    for reaction in mechanism.reactions():
        if reaction.equation not in [x.equation for x in qssa.reactions()]:
            # check if the unreversed reaction
            reaction.reversible = False
            if reaction.equation in [x.equation for x in qssa.reactions()]:
                reaction.reversible = True
                print("Removed reverse reaction:", reaction)
            else:
                reaction.reversible = True
                print("Removed reaction:", reaction)


def process_qss_lst(lst, visualize, method):
    """Apply QSSA reduction to list of Cantera yaml files."""
    lpath = pathlib.Path(lst)
    with open(lst, "r") as f:
        for line in f:
            if not line.startswith("#"):
                _, _, skel_file, nqssa_file = line.split()
                skelname = lpath.parents[0] / skel_file.strip()
                nqssa = lpath.parents[0] / nqssa_file.strip()
                print(f"""Converting file {skelname}""")
                process_qss(skelname, nqssa, visualize, method)


def main():
    """Apply QSSA reduction to Cantera yaml files."""
    start = time.time()
    parser = argparse.ArgumentParser(description="Mechanism converter")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--fname", help="Mechanism file", type=str)
    group.add_argument(
        "-lq", "--lst_qss", help="QSS mechanism directory file list", type=str
    )
    parser.add_argument("-n", "--nqssa", help="Non-QSSA species list", type=str)
    parser.add_argument(
        "-m",
        "--method",
        help="QSSA method (default: 2)",
        type=int,
        choices=[0, 1, 2],
        default=2,
    )
    parser.add_argument(
        "-v",
        "--visualize",
        help="Visualize quadratic coupling and QSSA dependencies",
        action="store_true",
    )
    args = parser.parse_args()

    if args.fname:
        process_qss(args.fname, args.nqssa, args.visualize, args.method)
    elif args.lst_qss:
        process_qss_lst(args.lst_qss, args.visualize, args.method)
    end = time.time()
    print(f"CEPTR QSS run time: {end - start:.2f} s")


if __name__ == "__main__":
    main()
