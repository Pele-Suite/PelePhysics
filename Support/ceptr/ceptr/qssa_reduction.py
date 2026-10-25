"""QSSA utilities."""

import itertools

import ceptr.qssa_graphs as cqg
import ceptr.utilities as cu


def identify_removals(mechanism, reaction_info, qssa_species):
    """Identify the reactions that should be removed."""
    reactions_to_remove = {}
    for orig_idx, _ in reaction_info.idxmap.items():
        reaction = mechanism.reaction(orig_idx)

        # Check for quadratic relations
        qssa_species_involved = cu.intersection(
            list(reaction.reactants.keys()), qssa_species
        )
        sum_coeff = sum(
            v for k, v in reaction.reactants.items() if k in qssa_species_involved
        )
        if sum_coeff > 1:
            if orig_idx not in reactions_to_remove:
                reactions_to_remove[orig_idx] = []
            reactions_to_remove[orig_idx].append("f")
        if reaction.reversible:
            qssa_species_involved = cu.intersection(
                list(reaction.products.keys()), qssa_species
            )
            sum_coeff = sum(
                v for k, v in reaction.products.items() if k in qssa_species_involved
            )
            if sum_coeff > 1:
                if orig_idx not in reactions_to_remove:
                    reactions_to_remove[orig_idx] = []
                reactions_to_remove[orig_idx].append("r")

    return reactions_to_remove


def remove_quadratic_method_0(mechanism, qssa_species):
    """Remove species involved in QSSA coupling."""
    qssa_problematic = identify_qssa_coupling(mechanism, qssa_species)

    tryspecies_qssa = []
    qssa_remove_proposal = []
    for lg in range(0, len(qssa_problematic) + 1):
        for subset in itertools.combinations(qssa_problematic, lg):
            # Make a new list of qssa species
            tryspecies_qssa[:] = qssa_species
            for species_remove in list(subset):
                tryspecies_qssa.remove(species_remove)
            # Check if still creates problem
            if not qssa_coupling(mechanism, tryspecies_qssa):
                # This combination works
                # Does this combination contain entirely another successful combination
                # If yes, then we are removing too many species, do not include it as solution
                add = True
                for success_qssa_found in qssa_remove_proposal:
                    if set(success_qssa_found).issubset(list(subset)):
                        add = False
                        break
                if add:
                    qssa_remove_proposal.append(list(subset))

    # Alphabetize, remove smallest set, break ties based on number of length of species name
    [x.sort() for x in qssa_remove_proposal]
    for x in qssa_remove_proposal:
        print(f"Candidate QSSA species for removal: {x}")
    ordered = sorted(
        qssa_remove_proposal, key=lambda x: (len(x), sum([len(y) for y in x]))
    )
    qssa_species_remove = ordered[0]

    # Make new set of qssa species
    new_qssa_species = [x for x in qssa_species if x not in qssa_species_remove]
    print(f"New set of QSSA species: {new_qssa_species}")
    return new_qssa_species


def remove_quadratic_method_1(mechanism, reaction_info, candidates_for_removal):
    """Remove all reactions that generate quadratic coupling.

    Method 1 removes forward and backwards reactions.
    """
    reactions_to_keep = []
    for orig_idx, _ in reaction_info.idxmap.items():
        if orig_idx not in candidates_for_removal:
            reaction = mechanism.reaction(orig_idx)
            reactions_to_keep.append(reaction)
    return reactions_to_keep


def remove_quadratic_method_2(mechanism, reaction_info, candidates_for_removal):
    """Remove all reactions that generate quadratic coupling.

    Method 2 removes forward, backward or both reactions.
    """
    reactions_to_keep = []
    forward_to_remove = []
    for orig_idx, _ in reaction_info.idxmap.items():
        reaction = mechanism.reaction(orig_idx)
        if orig_idx in candidates_for_removal:
            if (
                "f" in candidates_for_removal[orig_idx]
                and "r" in candidates_for_removal[orig_idx]
            ):
                continue
            elif "f" in candidates_for_removal[orig_idx]:
                if reaction.reversible:
                    forward_to_remove.append(reaction)
                    reactions_to_keep.append(reaction)
            elif "r" in candidates_for_removal[orig_idx]:
                reaction.reversible = False
                reactions_to_keep.append(reaction)
        else:
            reactions_to_keep.append(reaction)
    return reactions_to_keep, forward_to_remove


def qssa_coupling(mechanism, qssa_species, forward_to_remove=None):
    """Check for quadratic relations."""
    for reaction in mechanism.reactions():
        if forward_to_remove:
            if reaction.equation in [x.equation for x in forward_to_remove]:
                continue
        qssa_species_involved = cu.intersection(
            list(reaction.reactants.keys()), qssa_species
        )
        sum_coeff = sum(
            v for k, v in reaction.reactants.items() if k in qssa_species_involved
        )
        if sum_coeff > 1:
            return True
        if reaction.reversible:
            qssa_species_involved = cu.intersection(
                list(reaction.products.keys()), qssa_species
            )
            sum_coeff = sum(
                v for k, v in reaction.products.items() if k in qssa_species_involved
            )
            if sum_coeff > 1:
                return True
    return False


def identify_qssa_coupling(mechanism, qssa_species):
    """Find possible problems."""
    qssa_problematic = []
    for reaction in mechanism.reactions():
        qssa_species_involved = cu.intersection(
            list(reaction.reactants.keys()), qssa_species
        )
        sum_coeff = sum(
            v for k, v in reaction.reactants.items() if k in qssa_species_involved
        )
        if sum_coeff > 1:
            qssa_problematic += qssa_species_involved
        if reaction.reversible:
            qssa_species_involved = cu.intersection(
                list(reaction.products.keys()), qssa_species
            )
            sum_coeff = sum(
                v for k, v in reaction.products.items() if k in qssa_species_involved
            )
            if sum_coeff > 1:
                qssa_problematic += qssa_species_involved

    return sorted(set(qssa_problematic))


def visualize_qssa(mechanism, reaction_info, qssa_species):
    """Visualize quadratic coupling and QSSA dependencies."""
    links = cqg.directed_graph(mechanism, qssa_species)
    cqg.plot_directed_graph(mechanism, links)
    links, quadratic_reactions, problematic_species = cqg.qssa_quadratic_graph(
        mechanism, reaction_info, qssa_species
    )
    cqg.plot_quadratic_graph(mechanism, links, quadratic_reactions, problematic_species)
