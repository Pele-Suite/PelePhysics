"""QSSA functions needed for conversion."""

import copy
from collections import Counter, OrderedDict, defaultdict
from math import isclose

import numpy as np
import symengine as sme

import ceptr.constants as cc
import ceptr.formatter as cf
import ceptr.utilities as cu
import ceptr.writer as cw


def set_qssa_reactions(mechanism, species_info, reaction_info):
    """Get list of reaction indices that involve QSSA species."""
    for orig_idx, _ in reaction_info.idxmap.items():
        reaction = mechanism.reaction(orig_idx)
        qssa_reaction = False
        lst_reactants = [(k, v) for k, v in reaction.reactants.items()]
        lst_products = [(k, v) for k, v in reaction.products.items()]
        agents = list(set(lst_reactants + lst_products))
        for symbol, _ in agents:
            if symbol in species_info.qssa_species_list:
                qssa_reaction = True
        if qssa_reaction:
            reaction_info.qssa_reactions.append(orig_idx)
            print(
                "Found a QSSA reaction: ",
                orig_idx,
                reaction.equation,
            )
    reaction_info.n_qssa_reactions = len(reaction_info.qssa_reactions)


def get_qssa_networks(mechanism, species_info, reaction_info):
    """QSSA networks.

    Get networks showing which QSSA species are involved with one
    another and which reactions each QSSA species is involved in

    """
    # Create QSSA networks for mechanism
    create_ss_net(mechanism, species_info, reaction_info)
    create_sr_net(mechanism, species_info, reaction_info)

    # Get non-zero indices of networks to be used for coupling
    # "i" is for row "j" is for column
    species_info.qssa_info.ss_si, species_info.qssa_info.ss_sj = np.nonzero(
        species_info.qssa_info.ssnet
    )
    species_info.sr_si, species_info.qssa_info.sr_rj = np.nonzero(
        species_info.qssa_info.srnet
    )

    print("\n\n SS network for QSSA: ")
    print(species_info.qssa_info.ssnet)
    print(" SR network for QSSA: ")
    print(species_info.qssa_info.srnet)


def create_ss_net(mechanism, species_info, reaction_info):
    """Create the species-species network."""
    for orig_idx, _ in reaction_info.idxmap.items():
        reaction = mechanism.reaction(orig_idx)

        slist = []
        # get a list of species involved in the reactants and products
        for symbol, _ in reaction.reactants.items():
            if symbol in species_info.qssa_species_list:
                slist.append(symbol)
        for symbol, _ in reaction.products.items():
            if symbol in species_info.qssa_species_list:
                slist.append(symbol)

        # if species s1 and species s2 are in the same reaction,
        # denote they are linked in the species-species network
        for s1 in slist:
            for s2 in slist:
                # we should not use the original indices, but the reordered one
                species_info.qssa_info.ssnet[
                    species_info.ordered_idx_map[s1] - species_info.n_species
                ][species_info.ordered_idx_map[s2] - species_info.n_species] = 1


def create_sr_net(mechanism, species_info, reaction_info):
    """Create the species-reac network."""
    # for each reaction in the mechanism
    for orig_idx, _ in reaction_info.idxmap.items():
        reaction = mechanism.reaction(orig_idx)
        reactant_list = []
        product_list = []

        # get a list of species involved in the reactants and products

        for symbol, _ in reaction.reactants.items():
            if symbol in species_info.qssa_species_list:
                reactant_list.append(symbol)
        for symbol, _ in reaction.products.items():
            if symbol in species_info.qssa_species_list:
                product_list.append(symbol)

        # if qssa species s is in reaction number i,
        # denote they are linked in the Species-Reaction network
        # with negative if s is a reactant(consumed) and positive if s is a product(produced)
        for s in reactant_list:
            species_info.qssa_info.srnet[
                species_info.ordered_idx_map[s] - species_info.n_species
            ][orig_idx] = -1
        for s in product_list:
            species_info.qssa_info.srnet[
                species_info.ordered_idx_map[s] - species_info.n_species
            ][orig_idx] = 1


def qssa_validation(mechanism, species_info, reaction_info):
    """Check that the QSSA given are actually "valid" options.

    Exit if species is not valid.
    """
    # Check that QSSA species are all species used in the given mechanism
    for s in species_info.qssa_species_list:
        if s not in species_info.all_species_list:
            raise ValueError(f"species {s} is not in the mechanism")

    # Check that QSSA species are consumed/produced at least once to
    # ensure theoretically valid QSSA option (There is more to it than
    # that, but this is a quick catch based on that aspect)
    for i, symbol in enumerate(species_info.qssa_species_list):
        consumed = 0
        produced = 0
        for orig_idx in species_info.qssa_info.sr_rj[species_info.sr_si == i]:
            reaction = mechanism.reaction(orig_idx)
            remove_forward = cu.is_remove_forward(reaction_info, orig_idx)
            if any(reactant == symbol for reactant, _ in reaction.reactants.items()):
                if remove_forward:
                    consumed += 0
                else:
                    consumed += 1
                if reaction.reversible:
                    produced += 1
            if any(product == symbol for product, _ in reaction.products.items()):
                if remove_forward:
                    produced += 0
                else:
                    produced += 1
                if reaction.reversible:
                    consumed += 1

        if consumed == 0 or produced == 0:
            raise ValueError(
                f"Uh Oh! QSSA species {symbol}"
                " does not have a balanced consumption/production"
                "relationship in mechanism => bad QSSA choice"
            )


def qssa_coupling(mechanism, species_info, reaction_info):
    """Determine from qssa_SSnet which QSSA species depend on each other."""
    is_coupling = False
    coupling_reactions = ""
    list_coupling_reactions = []

    species_info.qssa_info.scnet = species_info.qssa_info.ssnet
    for i in range(species_info.n_qssa_species):
        for j in species_info.qssa_info.ss_sj[species_info.qssa_info.ss_si == i]:
            if j != i:
                count = 0
                for r in species_info.qssa_info.sr_rj[species_info.sr_si == j]:
                    reaction = mechanism.reaction(r)
                    remove_forward = cu.is_remove_forward(reaction_info, r)

                    # put forth any pathological case
                    if any(
                        reactant == species_info.qssa_species_list[j]
                        for reactant, _ in reaction.reactants.items()
                    ):
                        if any(
                            product == species_info.qssa_species_list[j]
                            for product, _ in reaction.products.items()
                        ):
                            raise ValueError(
                                f"Species {species_info.qssa_species_list[j]} appears"
                                " as both prod and reacts. Check reaction"
                                f" {reaction.equation}"
                            )

                    # we know j is in reaction r. Options are
                    # IF r is reversible
                    # j + i <-> prods OR reacts <-> j + i NOT ALLOWED
                    # all other combinations are fine.
                    # IF r is not reversible
                    # j + i -> prods NOT ALLOWED
                    # all other combinations are fine
                    # note that it is assumed no coupling bet same QSSA -- this is if i == j

                    if reaction.reversible:
                        # QSSA spec j is a reactant
                        if any(
                            reactant == species_info.qssa_species_list[j]
                            for reactant, _ in reaction.reactants.items()
                        ):
                            # Check if QSSA species i is a reactant too
                            if (not remove_forward) and any(
                                reactant == species_info.qssa_species_list[i]
                                for reactant, _ in reaction.reactants.items()
                            ):
                                is_coupling = True
                                if r not in list_coupling_reactions:
                                    coupling_reactions += (
                                        "R" + str(r) + " " + reaction.equation + "\n"
                                    )
                                    list_coupling_reactions.append(r)
                                print(
                                    "Quadratic coupling between "
                                    + species_info.qssa_species_list[j]
                                    + " and "
                                    + species_info.qssa_species_list[i]
                                    + " in reaction "
                                    + reaction.equation
                                    + " not allowed !!!"
                                )
                            # if QSSA specices j is a reactant and QSSA species i is a product,
                            # because react is two way then j depend on i and vice-versa
                            elif any(
                                product == species_info.qssa_species_list[i]
                                for product, _ in reaction.products.items()
                            ):
                                count += 1
                        # if QSSA species j is not a reactant, then it must be a product.
                        else:
                            # Check if QSSA species i is also a product
                            if any(
                                product == species_info.qssa_species_list[i]
                                for product, _ in reaction.products.items()
                            ):
                                is_coupling = True
                                if r not in list_coupling_reactions:
                                    coupling_reactions += (
                                        "R" + str(r) + " " + reaction.equation + "\n"
                                    )
                                    list_coupling_reactions.append(r)
                                print(
                                    "Quadratic coupling between "
                                    + species_info.qssa_species_list[j]
                                    + " and "
                                    + species_info.qssa_species_list[i]
                                    + " in reaction "
                                    + reaction.equation
                                    + " not allowed !!!"
                                )
                            # if QSSA specices j is a product and QSSA species i is a reactant
                            # because react is two way then j depend on i and vice-versa
                            elif any(
                                reactant == species_info.qssa_species_list[i]
                                for reactant, _ in reaction.reactants.items()
                            ):
                                count += 1
                    else:
                        # QSSA spec j is a reactant
                        if any(
                            reactant == species_info.qssa_species_list[j]
                            for reactant, _ in reaction.reactants.items()
                        ):
                            # Check if QSSA species i is a reactant too
                            if any(
                                reactant == species_info.qssa_species_list[i]
                                for reactant, _ in reaction.reactants.items()
                            ):
                                is_coupling = True
                                if r not in list_coupling_reactions:
                                    coupling_reactions += (
                                        "R" + str(r) + " " + reaction.equation + "\n"
                                    )
                                    list_coupling_reactions.append(r)
                                print(
                                    "Quadratic coupling between "
                                    + species_info.qssa_species_list[j]
                                    + " and "
                                    + species_info.qssa_species_list[i]
                                    + " in reaction "
                                    + reaction.equation
                                    + " not allowed !!!"
                                )
                            # if QSSA specices j is a reactant and QSSA species i is a product
                            elif any(
                                product == species_info.qssa_species_list[i]
                                for product, _ in reaction.products.items()
                            ):
                                count += 1

                if count == 0:
                    # i depends on j
                    species_info.qssa_info.scnet[i, j] = 0
            else:
                species_info.qssa_info.scnet[i, j] = 0
                for r in species_info.qssa_info.sr_rj[species_info.sr_si == j]:
                    reaction = mechanism.reaction(r)
                    remove_forward = cu.is_remove_forward(reaction_info, r)
                    species_appearances = 0

                    # put forth any pathological case
                    if any(
                        reactant == species_info.qssa_species_list[j]
                        for reactant, _ in reaction.reactants.items()
                    ):
                        if any(
                            product == species_info.qssa_species_list[j]
                            for product, _ in reaction.products.items()
                        ):
                            raise ValueError(
                                f"Species {species_info.qssa_species_list[j]} appears"
                                " as both prod and reacts. Check reaction"
                                f" {reaction.equation}"
                            )

                    if reaction.reversible:
                        # QSSA j is a reactant
                        if (not remove_forward) and any(
                            reactant == species_info.qssa_species_list[j]
                            for reactant, _ in reaction.reactants.items()
                        ):
                            for spec, coeff in reaction.reactants.items():
                                if spec == species_info.qssa_species_list[j]:
                                    species_appearances += 1
                                    if (coeff > 1.0) or (species_appearances > 1):
                                        is_coupling = True
                                        if r not in list_coupling_reactions:
                                            coupling_reactions += (
                                                "R"
                                                + str(r)
                                                + " "
                                                + reaction.equation
                                                + "\n"
                                            )
                                            list_coupling_reactions.append(r)
                                        print(
                                            "Quadratic coupling of "
                                            + species_info.qssa_species_list[j]
                                            + " with itself in reaction "
                                            + reaction.equation
                                            + " not allowed !!!"
                                        )
                        # if QSSA species j is not a reactant, then it must be a product.
                        else:
                            for spec, coeff in reaction.products.items():
                                if spec == species_info.qssa_species_list[j]:
                                    species_appearances += 1
                                    if (coeff > 1.0) or (species_appearances > 1):
                                        is_coupling = True
                                        if r not in list_coupling_reactions:
                                            coupling_reactions += (
                                                "R"
                                                + str(r)
                                                + " "
                                                + reaction.equation
                                                + "\n"
                                            )
                                            list_coupling_reactions.append(r)
                                        print(
                                            "Quadratic coupling of "
                                            + species_info.qssa_species_list[j]
                                            + " with itself in reaction "
                                            + reaction.equation
                                            + " not allowed !!!"
                                        )
                    else:
                        # QSSA spec j is a reactant
                        if any(
                            reactant == species_info.qssa_species_list[j]
                            for reactant, _ in reaction.reactants.items()
                        ):
                            for spec, coeff in reaction.reactants.items():
                                if spec == species_info.qssa_species_list[j]:
                                    species_appearances += 1
                                    if (coeff > 1.0) or (species_appearances > 1):
                                        is_coupling = True
                                        if r not in list_coupling_reactions:
                                            coupling_reactions += (
                                                "R"
                                                + str(r)
                                                + " "
                                                + reaction.equation
                                                + "\n"
                                            )
                                            list_coupling_reactions.append(r)
                                        print(
                                            "Quadratic coupling of "
                                            + species_info.qssa_species_list[j]
                                            + " with itself in reaction "
                                            + reaction.equation
                                            + " not allowed !!!"
                                        )

    species_info.qssa_info.sc_si, species_info.qssa_info.sc_sj = np.nonzero(
        species_info.qssa_info.scnet
    )
    print("\n\n SC network for QSSA: ")
    print(species_info.qssa_info.scnet)
    if is_coupling:
        raise ValueError(
            "There is some quadratic coupling in mechanism."
            + "Here is the list of reactions to check: \n"
            + coupling_reactions
        )


def set_qssa_needs(mechanism, species_info, reaction_info):
    """QSSA needs."""
    for i in range(species_info.n_qssa_species):
        needs_species = []
        count = 0
        for j in species_info.qssa_info.sc_sj[species_info.qssa_info.sc_si == i]:
            if j != i:
                needs_species.append(species_info.qssa_species_list[j])
                count += 1
        species_info.qssa_info.needs[species_info.qssa_species_list[i]] = needs_species
        species_info.qssa_info.needs_count[species_info.qssa_species_list[i]] = count

    species_info.qssa_info.needs_running = species_info.qssa_info.needs.copy()
    species_info.qssa_info.needs_count_running = (
        species_info.qssa_info.needs_count.copy()
    )

    print("Needs report (one per QSSA spec): ")
    print(species_info.qssa_info.needs)


def set_qssa_isneeded(mechanism, species_info, reaction_info):
    """QSSA is needed."""
    for i in range(species_info.n_qssa_species):
        is_needed_species = []
        count = 0
        for j in species_info.qssa_info.sc_si[species_info.qssa_info.sc_sj == i]:
            if j != i:
                is_needed_species.append(species_info.qssa_species_list[j])
                count += 1
        species_info.qssa_info.is_needed[species_info.qssa_species_list[i]] = (
            is_needed_species
        )
        species_info.qssa_info.is_needed_count[species_info.qssa_species_list[i]] = (
            count
        )

    species_info.qssa_info.is_needed_running = species_info.qssa_info.is_needed.copy()
    species_info.qssa_info.is_needed_count_running = (
        species_info.qssa_info.is_needed_count.copy()
    )

    print("Is needed report (one per QSSA spec): ")
    print(species_info.qssa_info.is_needed)


def get_qssa_groups(mechanism, species_info, reaction_info):
    """Get strongly connected components (groups).

    this includes two-way dependencies: (s1 needs s2) and (s2 needs s1) => group
    and cyclical dependencies: (s1 needs s2) and (s2 needs s3) and (s3 needs s1) => group

    This function along with find_closed_cycle is an implementation of:

    Tarjan's Strongly Connected Components Algorithm (1972)

    where node "index" is stored implicitly via the location of the
    node in the lowest_link OrderedDict and boolean "onStack"
    information is stored implicitly by checking for node existence in
    potential_group

    """
    print("\n\nDetermining groups of coupled species now...")
    print("---------------------------------")

    # Loop through species to tackle the needs group
    for member in species_info.qssa_info.needs_running.keys():
        # Only need to check things that have not already been
        # searched; i.e. they have no id/don't exist in the lowest
        # link value list yet
        if member not in species_info.qssa_info.lowest_link.keys():
            print("- dealing with group: ", member)
            find_closed_cycle(mechanism, species_info, member)

    print("** Groups of coupled species are: ", species_info.qssa_info.all_groups)

    group_count = 0
    # Rename and only store strongly connected components involving 2 or more species as groups
    for group in species_info.qssa_info.all_groups:
        if len(species_info.qssa_info.all_groups[group]) > 1:
            species_info.qssa_info.group["group_" + str(group_count)] = (
                species_info.qssa_info.all_groups[group]
            )
            group_count += 1
    print()
    print("** Final clean self groups are: ", species_info.qssa_info.group)

    update_group_needs(mechanism, species_info, reaction_info)
    update_group_dependencies(mechanism, species_info, reaction_info)


def find_closed_cycle(mechanism, species_info, species):
    """Find closed cycle."""
    # We start the recursion on node "species".
    # This species is considered the "parent" node.
    print(
        "      Searching for potential closed cycle involving parent: ",
        species,
    )

    # We only enter the recursion if the species has not already been
    # discovered so we add this species to the potential group list
    # and give it a lowest link value The species location in the
    # lowest link value dictionary denotes its discovery order, or id
    species_info.qssa_info.potential_group.append(species)
    species_info.qssa_info.lowest_link[species] = species_info.qssa_info.discovery_order
    species_info.qssa_info.discovery_order += 1
    parent = species

    print(
        "      Upon initialization, discovery order is: ",
        list(species_info.qssa_info.lowest_link.keys()).index(species),
        " and lowest link value is: ",
        species_info.qssa_info.discovery_order - 1,
    )
    print()

    # Loop through the needs of the parent
    # Each need is denoted as a "child" of the parent
    for need in species_info.qssa_info.needs_running[parent]:
        child = need
        print("       x Start level of needs loop ")
        print("       x Child is: ", child)

        # If the child has not yet been discovered, we recurse so that
        # the child becomes the parent and the search continues as
        # described above
        if child not in species_info.qssa_info.lowest_link.keys():
            print("         xx Child has never been visited at all...")
            print("         xx Initiate recursion to assign lowlink value ")

            find_closed_cycle(mechanism, species_info, child)

            print(
                "         xx We've finished a recursion! The child that"
                " was passed in was: ",
                child,
                " with parent ",
                parent,
            )
            print(
                "         xx The lowest link value of the parent is: ",
                species_info.qssa_info.lowest_link[parent],
            )
            print(
                "         xx The lowest link value of this child is: ",
                species_info.qssa_info.lowest_link[child],
            )
            print()

            # If the lowest link connection of the child is lower than
            # that of the parent's, then this lowest link value
            # becomes the parent's new lowest link value.  This update
            # comes into effect when the child at the end of the
            # recursion matches the original parent of the search or
            # has a lower link connection from an earlier search than
            # the parent currently does
            if child in species_info.qssa_info.potential_group:
                species_info.qssa_info.lowest_link[parent] = min(
                    species_info.qssa_info.lowest_link[parent],
                    species_info.qssa_info.lowest_link[child],
                )

            print(
                "         ** Therefore, the lowest link value of the"
                " parent then becomes: ",
                species_info.qssa_info.lowest_link[parent],
            )
            print(
                "         ** The lowest link connections are now: ",
                species_info.qssa_info.lowest_link,
            )
            print()
            print()

        # If the child is already listed in the potential_group, then
        # it has been discovered during this search and is already a
        # part of the simply connected component Note: If the child
        # already has a lowest link value but is NOT in the
        # potential_group, that means it is a part of a previously
        # found group
        elif child in species_info.qssa_info.potential_group:
            print("         xx Child has already been visited this recursion ")
            print(
                "         xx The lowest link value of the parent is: ",
                species_info.qssa_info.lowest_link[parent],
            )
            print(
                "         xx The discovery order of this child is: ",
                list(species_info.qssa_info.lowest_link.keys()).index(child),
            )

            # Since the child has been discovered already during this
            # search, that means it is in the group but still in the
            # recursion process, Update the parent's lowest link value
            # with this child's discovery order
            species_info.qssa_info.lowest_link[parent] = min(
                species_info.qssa_info.lowest_link[parent],
                list(species_info.qssa_info.lowest_link.keys()).index(child),
            )

            print()
            print(
                "         **** Therefore, the lowest link value of the"
                " parent then becomes: ",
                species_info.qssa_info.lowest_link[parent],
            )
            print(
                "         **** The lowest link connections are now: ",
                species_info.qssa_info.lowest_link,
            )
            print()
            print()

    # If, after searching all children and updating lowest link
    # connections, you reach a parent whose lowest link still matches
    # its original value (or starting position) Then you have reached
    # the root of a simply connected component (aka you've found a
    # group)
    if (
        list(species_info.qssa_info.lowest_link.keys()).index(parent)
        == species_info.qssa_info.lowest_link[parent]
    ):
        hold = []
        while True:
            # remove group species from the running potential group
            # list until you reach the parent species add these to the
            # official group, leaving other potential group components
            # in the potential_group list for continued recursion
            # completion
            node = species_info.qssa_info.potential_group.pop()
            hold.append(node)
            if node == parent:
                break
        species_info.qssa_info.all_groups[parent] = hold
        print("         Group is: ", species_info.qssa_info.all_groups[parent])
        print()
        print()


def update_group_needs(mechanism, species_info, reaction_info):
    """Update group member needs with group names.

    group member needs species -> group needs species
    group member is needed by species -> group is needed by species
    """
    print("\n\nUpdating group needs...")
    print("---------------------------------")

    for group_key in list(species_info.qssa_info.group.keys()):
        print("-Dealing with group " + group_key)

        update_needs = []
        update_is_needed = []
        update_needs_count = 0
        update_needed_count = 0

        group_needs = {}
        group_needs_count = {}
        group_is_needed = {}
        group_is_needed_count = {}

        other_groups = list(species_info.qssa_info.group.keys())
        other_groups.remove(group_key)
        print("  (other groups are: ", other_groups, ")")

        # for each species in the current group
        for spec in species_info.qssa_info.group[group_key]:
            print("... for group member: " + spec)
            # look at any additional needs that are not already
            # accounted for with the group
            for need in list(
                set(species_info.qssa_info.needs_running[spec])
                - set(species_info.qssa_info.group[group_key])
            ):
                print("        An additional not-in-group need is " + need)
                not_in_group = True
                # check the other groups to see if the need can be found in one of them
                for other_group in other_groups:
                    # if the other group is not already accounted for
                    # and it contains the spec need we're looking for,
                    # update the group needs with that group that
                    # contains the spec need
                    if other_group not in update_needs and any(
                        member == need
                        for member in species_info.qssa_info.group[other_group]
                    ):
                        print("        it is found in a different group. Adding it.")
                        not_in_group = False
                        update_needs.append(other_group)
                        update_needs_count += 1
                    elif other_group in update_needs and any(
                        member == need
                        for member in species_info.qssa_info.group[other_group]
                    ):
                        print(
                            "        it is found in a group that was"
                            + "already put in the list due to the fact"
                            + "that another species in the group is"
                            + "needed by the current species."
                        )
                        not_in_group = False
                # alternatively, if this is just a solo need that's
                # not in another group, update the group needs with
                # just that need.
                if not_in_group and need not in update_needs:
                    print(
                        "        this need was not found in a group ! Adding"
                        " the spec directly"
                    )
                    update_needs.append(need)
                    update_needs_count += 1
            # look at any additional species (outside of the group)
            # that depend on the current group member
            for needed in list(
                set(species_info.qssa_info.is_needed_running[spec])
                - set(species_info.qssa_info.group[group_key])
            ):
                print("        An additional not-in-group is-needed is " + needed)
                not_in_group = True
                # for the other groups
                for other_group in other_groups:
                    # if the other group hasn't already been accounted
                    # for and the species is in that group, then that
                    # other group depends on a species in the current
                    # group
                    if other_group not in update_is_needed and any(
                        member == needed
                        for member in species_info.qssa_info.group[other_group]
                    ):
                        print("        it is found in a different group. Adding it.")
                        not_in_group = False
                        update_is_needed.append(other_group)
                        update_needed_count += 1
                    elif other_group in update_is_needed and any(
                        member == needed
                        for member in species_info.qssa_info.group[other_group]
                    ):
                        print(
                            "        it is found in a group that was"
                            + "already put in the list due to the fact"
                            + "that another species in the group is"
                            + "needed by the current species."
                        )
                        not_in_group = False
                # if the species is not in another group, then that
                # lone species just depends on the current group.
                if not_in_group and needed not in update_is_needed:
                    print(
                        "        this is-needed was not found in a group !"
                        + "Adding the spec directly"
                    )
                    update_is_needed.append(needed)
                    update_needed_count += 1

        group_needs[group_key] = update_needs
        group_needs_count[group_key] = update_needs_count
        group_is_needed[group_key] = update_is_needed
        group_is_needed_count[group_key] = update_needed_count

        species_info.qssa_info.needs_running.update(group_needs)
        species_info.qssa_info.needs_count_running.update(group_needs_count)
        species_info.qssa_info.is_needed_running.update(group_is_needed)
        species_info.qssa_info.is_needed_count_running.update(group_is_needed_count)

        print("So, ", group_key, " needs ", update_needs)
        print("So, ", group_key, " is-needed is ", update_is_needed)

    for group in list(species_info.qssa_info.group.keys()):
        for spec in species_info.qssa_info.group[group]:
            if spec in species_info.qssa_info.needs_running:
                del species_info.qssa_info.needs_running[spec]
                del species_info.qssa_info.needs_count_running[spec]
                del species_info.qssa_info.is_needed_running[spec]

    print()
    print("** This is the final needs running and is_needed running: ")
    print(species_info.qssa_info.needs_running)
    print(species_info.qssa_info.is_needed_running)


def update_group_dependencies(mechanism, species_info, reaction_info):
    """Update solo species dependent on group members with group names.

    species needs member -> species needs group
    species is needed by group member -> species is needed by group
    """
    print("\n\nUpdating group dependencies...")
    print("---------------------------------")

    solo_needs = species_info.qssa_info.needs_running.copy()
    solo_needs_count = species_info.qssa_info.needs_count_running.copy()
    solo_is_needed = species_info.qssa_info.is_needed_running.copy()
    solo_is_needed_count = species_info.qssa_info.is_needed_count_running.copy()

    # remove the groups because we're just dealing with things that aren't in groups now
    for group in list(species_info.qssa_info.group.keys()):
        del solo_needs[group]
        del solo_needs_count[group]
        del solo_is_needed[group]
        del solo_is_needed_count[group]

    for solo in list(solo_needs.keys()):
        print("-Dealing with solo species " + solo)
        update_needs = []
        update_is_needed = []
        update_needs_count = 0
        update_needed_count = 0
        for need in solo_needs[solo]:
            print("... who needs: " + need)
            not_in_group = True
            for group in list(species_info.qssa_info.group.keys()):
                if group not in update_needs and any(
                    member == need for member in species_info.qssa_info.group[group]
                ):
                    print("        this species is in group: ", group)
                    not_in_group = False
                    update_needs.append(group)
                    update_needs_count += 1
                elif group in update_needs and any(
                    member == need for member in species_info.qssa_info.group[group]
                ):
                    print(
                        "        this group was already put in the list"
                        + "due to the fact that another species in the"
                        + "group is needed by the current species."
                    )
                    not_in_group = False
            if not_in_group and need not in update_needs:
                print(
                    "        this need was not found in a group ! Adding the"
                    " spec directly"
                )
                update_needs.append(need)
                update_needs_count += 1

        for needed in solo_is_needed[solo]:
            print("... who is-needed needs are: " + needed)
            not_in_group = True
            for group in list(species_info.qssa_info.group.keys()):
                if group not in update_is_needed and any(
                    member == needed for member in species_info.qssa_info.group[group]
                ):
                    print("        this species is in group: ", group)
                    not_in_group = False
                    update_is_needed.append(group)
                    update_needed_count += 1
                if group in update_is_needed and any(
                    member == needed for member in species_info.qssa_info.group[group]
                ):
                    print(
                        "        this group was already put in the list"
                        + "due to the fact that another species in the"
                        + "group is needed by the current species."
                    )
                    not_in_group = False
            if not_in_group and needed not in update_is_needed:
                print(
                    "        this is-needed need was not found in a group !"
                    + "Adding the spec directly"
                )
                update_is_needed.append(needed)
                update_needed_count += 1

        solo_needs[solo] = update_needs
        solo_needs_count[solo] = update_needs_count
        solo_is_needed[solo] = update_is_needed
        solo_is_needed_count[solo] = update_needed_count

    species_info.qssa_info.needs_running.update(solo_needs)
    species_info.qssa_info.needs_count_running.update(solo_needs_count)
    species_info.qssa_info.is_needed_running.update(solo_is_needed)
    species_info.qssa_info.is_needed_count_running.update(solo_is_needed_count)

    print()
    print("** This is the final needs running and is_needed running: ")
    print(species_info.qssa_info.needs_running)
    print(species_info.qssa_info.is_needed_running)


def sort_qssa_computation(mechanism, species_info, reaction_info):
    """Sort order that QSSA species need to be computed using dependencies."""
    # look at how many dependencies each component has
    needs_count_regress = species_info.qssa_info.needs_count_running.copy()

    # There should always be a component present that loses
    # all dependencies as you update the computation
    while 0 in list(needs_count_regress.values()):
        needs_count_base = needs_count_regress.copy()
        # for each component (species, group, sup group, etc.) that needs things...
        for member in needs_count_base:
            # if that component doesn't need anything
            if needs_count_base[member] == 0:
                print("-Dealing with member ", member)
                # solve that component now
                species_info.qssa_info.decouple_index[
                    species_info.qssa_info.decouple_count
                ] = member
                # then delete it out of the updating needs list
                del needs_count_regress[member]
                # for anything needed by that component
                for needed in species_info.qssa_info.is_needed_running[member]:
                    # decrease that thing's dependency since it has now been taken care of
                    needs_count_regress[needed] -= 1
                species_info.qssa_info.decouple_count += 1

    # If your decouple count doesn't match the number of components with needs,
    # then the system is more complicated than what these functions can handle currently
    if len(species_info.qssa_info.decouple_index) != len(
        species_info.qssa_info.needs_running
    ):
        print("WARNING: Some components may not have been taken into account")
    print()
    print(
        "** order of execution for qssa concentration calculations: ",
        species_info.qssa_info.decouple_index,
    )


def sort_qssa_solution_elements(mechanism, species_info, reaction_info, syms):
    """Components needed to set up QSSA algebraic expressions from AX = B.

    where A contains coefficients from qf's and qr's, X contains QSSA
    species concentrations, and B contains qf's and qr's Info stored
    as: RHS vector (non-QSSA and QSSA qf's and qr's), coefficient of
    species (diagonal elements of A), coefficient of group mates
    (coupled off-diagonal elements of A)

    """
    # Need to get qfqr_coeff reaction map
    ispecial = reaction_info.index[5:7]
    nspecial_qssa = 0
    ispecial_qssa = [0, 0]
    special_first = True

    # Find out bounds for special reacs in smaller qssReactions list
    for reac_id in reaction_info.qssa_reactions:
        if reac_id >= ispecial[0] and reac_id < ispecial[1]:
            nspecial_qssa += 1
            if special_first:
                ispecial_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                special_first = False
            ispecial_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1

    # remove special reacs for some reason ?
    reaction_info.qfqr_co_idx_map = reaction_info.qssa_reactions
    if (ispecial_qssa[1] - ispecial_qssa[0]) > 0:
        for index in range(ispecial_qssa[0], ispecial_qssa[1]):
            del reaction_info.qfqr_co_idx_map[index]

    for i in range(species_info.n_qssa_species):
        symbol = species_info.qssa_species_list[i]
        print()
        print("-<>-Dealing with QSSA species ", i, symbol)
        print("__________________________________________")
        coupled = []
        rhs_hold = []
        coeff_hold = []
        group_coeff_hold = defaultdict(list)
        rhs_hold_smp = []
        coeff_hold_smp = []
        group_coeff_hold_smp = defaultdict(list)

        for r in species_info.qssa_info.sr_rj[species_info.sr_si == i]:
            reaction = mechanism.reaction(r)
            remove_forward = cu.is_remove_forward(reaction_info, r)
            print("... who is involved in reac ", r, reaction.equation)
            print(
                "... reaction ",
                r,
                "is QSSA reaction number ",
                reaction_info.qfqr_co_idx_map.index(r),
            )

            direction = species_info.qssa_info.srnet[i][r]

            # Check if reaction contains other QSSA species
            coupled = [
                species
                for species in list(
                    set(species_info.sr_si[species_info.qssa_info.sr_rj == r])
                )
            ]
            coupled_qssa = [species_info.qssa_species_list[j] for j in coupled]

            for species in coupled_qssa:
                if species != symbol:
                    other_qssa = species

            if len(coupled) >= 2:
                # Check if all QSSA are only reactants
                all_qssa_reactants = True
                if any(
                    product == other_qssa for product, _ in reaction.products.items()
                ):
                    all_qssa_reactants = False
                if any(product == symbol for product, _ in reaction.products.items()):
                    all_qssa_reactants = False

            if len(coupled) < 2:
                print("        this reaction only involves that QSSA ")
                # if QSSA species is a reactant
                if direction == -1:
                    print(
                        "        species ",
                        symbol,
                        " in reaction ",
                        r,
                        " is a reactant",
                    )
                    species_appearances = 0
                    for spec, coeff in reaction.reactants.items():
                        if spec == symbol:
                            species_appearances += coeff

                    coeff_hold.append(
                        "-qf_co[" + str(reaction_info.qfqr_co_idx_map.index(r)) + "]"
                    )
                    coeff_hold_smp.append(
                        -syms.qf_qss_smp[reaction_info.qfqr_co_idx_map.index(r)]
                    )
                    if reaction.reversible:
                        rhs_hold.append(
                            "+"
                            + str(float(species_appearances))
                            + "*qr_co["
                            + str(reaction_info.qfqr_co_idx_map.index(r))
                            + "]"
                        )
                        species_appearances = syms.convert_number_to_int(
                            species_appearances
                        )
                        rhs_hold_smp.append(
                            species_appearances
                            * syms.qr_qss_smp[reaction_info.qfqr_co_idx_map.index(r)]
                        )
                # if QSSA species is a product
                elif direction == 1:
                    print(
                        "        species ",
                        symbol,
                        " in reaction ",
                        r,
                        " is a product",
                    )
                    species_appearances = 0
                    for spec, coeff in reaction.products.items():
                        if spec == symbol:
                            species_appearances += coeff

                    rhs_hold.append(
                        "+"
                        + str(float(species_appearances))
                        + "*qf_co["
                        + str(reaction_info.qfqr_co_idx_map.index(r))
                        + "]"
                    )
                    species_appearances = syms.convert_number_to_int(
                        species_appearances
                    )
                    rhs_hold_smp.append(
                        species_appearances
                        * syms.qf_qss_smp[reaction_info.qfqr_co_idx_map.index(r)]
                    )
                    if reaction.reversible:
                        coeff_hold.append(
                            "-qr_co["
                            + str(reaction_info.qfqr_co_idx_map.index(r))
                            + "]"
                        )
                        coeff_hold_smp.append(
                            -syms.qr_qss_smp[reaction_info.qfqr_co_idx_map.index(r)]
                        )

            elif len(coupled) >= 2 and all_qssa_reactants and remove_forward:
                # coupling is actually not a coupling because forward reaction is removed
                if reaction.reversible:
                    # Should always be true
                    # Check how many times species appear in reactants
                    species_appearances = 0
                    for spec, coeff in reaction.reactants.items():
                        if spec == symbol:
                            species_appearances += coeff

                    rhs_hold.append(
                        "+"
                        + str(float(species_appearances))
                        + "*qr_co["
                        + str(reaction_info.qfqr_co_idx_map.index(r))
                        + "]"
                    )
                    species_appearances = syms.convert_number_to_int(
                        species_appearances
                    )
                    rhs_hold_smp.append(
                        species_appearances
                        * syms.qr_qss_smp[reaction_info.qfqr_co_idx_map.index(r)]
                    )

            else:
                # note in this case there can only be 2 QSSA in one reac
                coupled_qssa = [species_info.qssa_species_list[j] for j in coupled]
                print(
                    "        this reaction couples the following QSSA: ",
                    coupled_qssa,
                )

                # assumes only 2 QSSA can appear in a reac now
                # other_qssa_list = []
                # for species in coupled_qssa:
                #    other_qssa_list.append(species)
                # other_qssa_list.remove(symbol)
                for species in coupled_qssa:
                    if species != symbol:
                        other_qssa = species
                # for group in species_info.qssa_info.group:
                #    if set(coupled_qssa).issubset(set(species_info.qssa_info.group[group])):
                #        "        (they are both on the same group)"
                #        group_flag = True

                # THIS is the right groupCoeff list
                # if group_flag:

                # if QSSA species is a reactant (other QSSA must be a
                # product to be coupled to be coupled here, or
                # quadratic coupling would have been triggered
                # earlier)
                if direction == -1:
                    all_qssa_reactants = True
                    if any(
                        product == other_qssa
                        for product, _ in reaction.products.items()
                    ):
                        all_qssa_reactants = False
                    if any(
                        product == symbol for product, _ in reaction.products.items()
                    ):
                        all_qssa_reactants = False

                    remove_forward = cu.is_remove_forward(reaction_info, r)
                    if all_qssa_reactants and remove_forward:
                        print(
                            "        species ",
                            symbol,
                            " and species ",
                            other_qssa,
                            " in purely irreversible reaction ",
                            r,
                            " are both reactants",
                        )
                        print(
                            "        this reaction does not contribute to any QSSA"
                            + "coefficients and is thus ignored"
                        )

                    else:
                        print(
                            "        species ",
                            symbol,
                            " in reaction ",
                            r,
                            " is a reactant",
                        )

                        species_appearances = 0
                        for spec, coeff in reaction.reactants.items():
                            if spec == symbol:
                                species_appearances += coeff

                        coeff_hold.append(
                            "-qf_co["
                            + str(reaction_info.qfqr_co_idx_map.index(r))
                            + "]"
                        )
                        coeff_hold_smp.append(
                            -syms.qf_qss_smp[reaction_info.qfqr_co_idx_map.index(r)]
                        )
                        if reaction.reversible:
                            group_coeff_hold[other_qssa].append(
                                "+"
                                + str(float(species_appearances))
                                + "*qr_co["
                                + str(reaction_info.qfqr_co_idx_map.index(r))
                                + "]"
                            )
                            if len(group_coeff_hold_smp[other_qssa]) == 0:
                                group_coeff_hold_smp[other_qssa] = [0]
                            species_appearances = syms.convert_number_to_int(
                                species_appearances
                            )
                            group_coeff_hold_smp[other_qssa][0] += (
                                species_appearances
                                * syms.qr_qss_smp[
                                    reaction_info.qfqr_co_idx_map.index(r)
                                ]
                            )
                # if QSSA species is a product AND other QSSA species
                # is a reactant (not guaranteed; must check that QSSA
                # are on opposite sides of equation)
                elif direction == 1 and any(
                    reactant == other_qssa for reactant, _ in reaction.reactants.items()
                ):
                    print(
                        "        species ",
                        symbol,
                        " in reaction ",
                        r,
                        " is a product",
                    )
                    print("        other qssa species is ", other_qssa)

                    species_appearances = 0
                    for spec, coeff in reaction.products.items():
                        if spec == symbol:
                            species_appearances += coeff

                    if len(group_coeff_hold_smp[other_qssa]) == 0:
                        group_coeff_hold_smp[other_qssa] = [0]
                    group_coeff_hold[other_qssa].append(
                        "+"
                        + str(float(species_appearances))
                        + "*qf_co["
                        + str(reaction_info.qfqr_co_idx_map.index(r))
                        + "]"
                    )
                    species_appearances = syms.convert_number_to_int(
                        species_appearances
                    )
                    group_coeff_hold_smp[other_qssa][0] += (
                        species_appearances
                        * syms.qf_qss_smp[reaction_info.qfqr_co_idx_map.index(r)]
                    )
                    if reaction.reversible:
                        coeff_hold.append(
                            "-qr_co["
                            + str(reaction_info.qfqr_co_idx_map.index(r))
                            + "]"
                        )
                        coeff_hold_smp.append(
                            -syms.qr_qss_smp[reaction_info.qfqr_co_idx_map.index(r)]
                        )
                # last option is that BOTH QSSA are products, but the
                # reaction is only one way, so it doesn't matter. This
                # is ignored in the quadratic coupling check as the
                # reverse rate would be zero and thus would not affect
                # anything anyway.
                else:
                    print(
                        "        species ",
                        symbol,
                        " and species ",
                        other_qssa,
                        " in irreversible reaction ",
                        r,
                        " are both products",
                    )
                    print(
                        "        this reaction does not contribute to any QSSA"
                        + "coefficients and is thus ignored"
                    )

            print()
            print(
                "After dealing with QSSA species "
                + symbol
                + " in  reaction "
                + str(r)
                + " we have the following: "
            )
            print("rhs_hold is ", rhs_hold)
            print("coeff_hold is ", coeff_hold)
            print("group_coeff_hold is ", group_coeff_hold)
            print()

        species_info.qssa_info.rhs[symbol] = " ".join(rhs_hold)
        species_info.qssa_info.coeff[symbol] = " ".join(coeff_hold)

        species_info.qssa_info.rhs_smp[symbol] = 0
        for rhs_smp in rhs_hold_smp:
            species_info.qssa_info.rhs_smp[symbol] += rhs_smp

        species_info.qssa_info.coeff_smp[symbol] = 0
        for coeff_smp in coeff_hold_smp:
            species_info.qssa_info.coeff_smp[symbol] += coeff_smp

        species_info.qssa_info.qssa_coeff[symbol] = OrderedDict()
        species_info.qssa_info.qssa_coeff_smp[symbol] = OrderedDict()
        for j in range(species_info.n_qssa_species):
            if j != i:
                other_qssa = species_info.qssa_species_list[j]
                if other_qssa in group_coeff_hold:
                    species_info.qssa_info.qssa_coeff[symbol][other_qssa] = " ".join(
                        group_coeff_hold[other_qssa]
                    )
                    species_info.qssa_info.qssa_coeff_smp[symbol][other_qssa] = (
                        group_coeff_hold_smp[other_qssa]
                    )
                    print(species_info.qssa_info.qssa_coeff[symbol])
                    print(species_info.qssa_info.qssa_coeff_smp[symbol])
                else:
                    species_info.qssa_info.qssa_coeff[symbol][other_qssa] = "0.0"
                    species_info.qssa_info.qssa_coeff_smp[symbol][other_qssa] = 0.0

        print("Here is everything: ")
        print()
        print("rhs: ", species_info.qssa_info.rhs[symbol])
        print("self: ", species_info.qssa_info.coeff[symbol])
        print("coupling: ", species_info.qssa_info.qssa_coeff[symbol])
        print()

    print()
    print()
    print("Final lists: ")
    print("-------------")
    print("rhs: ", species_info.qssa_info.rhs)
    print("self: ", species_info.qssa_info.coeff)
    print("coupling: ", species_info.qssa_info.qssa_coeff)
    print()
    print()


def qssa_coeff_functions(fstream, mechanism, species_info, reaction_info, syms):
    """QSSA coeff functions."""
    itroe = reaction_info.index[0:2]
    isri = reaction_info.index[1:3]
    ilindemann = reaction_info.index[2:4]
    i3body = reaction_info.index[3:5]
    isimple = reaction_info.index[4:6]
    ispecial = reaction_info.index[5:7]

    print("troe index range is: ", itroe)
    print("sri index range is: ", isri)
    print("lindemann index range is: ", ilindemann)
    print("3body index range is: ", i3body)
    print("simple index range is: ", isimple)
    print("special index range is: ", ispecial)

    ntroe_qssa = 0
    nsri_qssa = 0
    nlindemann_qssa = 0
    n3body_qssa = 0
    nsimple_qssa = 0
    nspecial_qssa = 0

    itroe_qssa = [0, 0]
    isri_qssa = [0, 0]
    ilindemann_qssa = [0, 0]
    i3body_qssa = [0, 0]
    isimple_qssa = [0, 0]
    ispecial_qssa = [0, 0]

    troe_first = True
    sri_first = True
    lindemann_first = True
    threebody_first = True
    simple_first = True
    special_first = True

    for reac_id in reaction_info.qssa_reactions:
        if reac_id >= itroe[0] and reac_id < itroe[1]:
            print(
                "reaction ",
                reac_id,
                mechanism.reaction(reac_id).equation,
                " goes in troe",
            )
            ntroe_qssa += 1
            if troe_first:
                itroe_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                troe_first = False
            itroe_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1
        if reac_id >= isri[0] and reac_id < isri[1]:
            print(
                "reaction ",
                reac_id,
                mechanism.reaction(reac_id).equation,
                " goes in sri",
            )
            nsri_qssa += 1
            if sri_first:
                isri_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                sri_first = False
            isri_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1
        if reac_id >= ilindemann[0] and reac_id < ilindemann[1]:
            print(
                "reaction ",
                reac_id,
                mechanism.reaction(reac_id).equation,
                " goes in lindemann",
            )
            nlindemann_qssa += 1
            if lindemann_first:
                ilindemann_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                lindemann_first = False
            ilindemann_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1
        if reac_id >= i3body[0] and reac_id < i3body[1]:
            print(
                "reaction ",
                reac_id,
                mechanism.reaction(reac_id).equation,
                " goes in 3body",
            )
            n3body_qssa += 1
            if threebody_first:
                i3body_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                threebody_first = False
            i3body_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1
        if reac_id >= isimple[0] and reac_id < isimple[1]:
            nsimple_qssa += 1
            if simple_first:
                isimple_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                simple_first = False
            isimple_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1
        if reac_id >= ispecial[0] and reac_id < ispecial[1]:
            print(
                "reaction ",
                reac_id,
                mechanism.reaction(reac_id).equation,
                " goes in special",
            )
            nspecial_qssa += 1
            if special_first:
                ispecial_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                special_first = False
            ispecial_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1

    assert len(reaction_info.index) == 7

    # qssa coefficients
    cw.writer(fstream)
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void comp_qss_coeff"
        + "(amrex::Real * k_f, amrex::Real * qf, amrex::Real * qr, const"
        " amrex::Real * sc,"
        + "const amrex::Real T, amrex::Real * g_RT, amrex::Real * g_RT_qss)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "const amrex::Real invT = 1.0 / T;")
    cw.writer(fstream, "const amrex::Real logT = log(T);")
    cw.writer(fstream)

    if mechanism.n_reactions == 0:
        cw.writer(fstream)
    else:
        cw.writer(
            fstream,
            cw.comment("reference concentration: P_atm / (RT) in inverse mol/m^3"),
        )
        cw.writer(
            fstream,
            f"const amrex::Real refC = {cc.Patm_pa:g} /"
            f" {cc.R.to(cc.ureg.joule / (cc.ureg.mole / cc.ureg.kelvin)).m:g} *"
            " invT;",
        )
        # coeff1 = cc.Patm_pa
        # coeff2 = cc.R.to(cc.ureg.joule / (cc.ureg.mole / cc.ureg.kelvin)).m
        # syms.refC_smp = coeff1 / coeff2 * syms.invT_smp
        cw.writer(fstream, "const amrex::Real refCinv = 1. / refC;")
        # syms.refCinv_smp = 1.0 / syms.refC_smp

    cw.writer(fstream, cw.comment("compute the mixture concentration"))
    cw.writer(fstream, "amrex::Real mixture = 0.0;")
    cw.writer(fstream, f"for (int i = 0; i < {species_info.n_species}; ++i)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "mixture += sc[i];")
    cw.writer(fstream, "}")
    nclassd_qssa = reaction_info.n_qssa_reactions - nspecial_qssa
    # nCorr_qssa = n3body_qssa + ntroe_qssa + nsri_qssa + nlindemann_qssa

    for i in range(nclassd_qssa):
        cw.writer(fstream)
        orig_idx = reaction_info.qssa_reactions[i]
        reaction = mechanism.reaction(orig_idx)
        remove_forward = cu.is_remove_forward(reaction_info, orig_idx)
        idx = i
        if bool(reaction.orders):
            dim = cu.phase_space_units(reaction.orders)
        else:
            dim = cu.phase_space_units(reaction.reactants)
        third_body = reaction.third_body is not None
        falloff = reaction.rate.type == "falloff"
        is_troe = reaction.rate.sub_type == "Troe"
        is_sri = reaction.rate.sub_type == "Sri"
        is_lindemann = reaction.rate.sub_type == "Lindemann"
        aeuc = cu.activation_energy_units()
        if not third_body and not falloff:
            # Case 3 !PD, !TB
            ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
            # pef = (reaction.rate.pre_exponential_factor * ctuc).to_base_units()
            beta = reaction.rate.temperature_exponent
            # ae = (
            # reaction.rate.activation_energy * cc.ureg.joule / cc.ureg.kmol
            # ).to(aeuc)
            beta = syms.convert_number_to_int(beta)
        elif not falloff:
            # Case 2 !PD, TB
            ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), -dim)
            # pef = (reaction.rate.pre_exponential_factor * ctuc).to_base_units()
            beta = reaction.rate.temperature_exponent
            # ae = (
            # reaction.rate.activation_energy * cc.ureg.joule / cc.ureg.kmol
            # ).to(aeuc)
            beta = syms.convert_number_to_int(beta)
        else:
            # Case 2 !PD, TB
            ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
            # pef = (
            # reaction.rate.high_rate.pre_exponential_factor * ctuc
            # ).to_base_units()
            beta = reaction.rate.high_rate.temperature_exponent
            # ae = (
            # reaction.rate.high_rate.activation_energy
            # * cc.ureg.joule
            # / cc.ureg.kmol
            # ).to(aeuc)

            low_pef = (
                reaction.rate.low_rate.pre_exponential_factor * ctuc
            ).to_base_units()
            low_beta = reaction.rate.low_rate.temperature_exponent
            low_ae = (
                reaction.rate.low_rate.activation_energy * cc.ureg.joule / cc.ureg.kmol
            ).to(aeuc)
            if is_troe:
                troe = reaction.rate.falloff_coeffs
                ntroe = len(troe)
            elif is_sri:
                pass
                # sri = reaction.rate.falloff_coeffs
                # nsri = len(sri)
            elif is_lindemann:
                pass
            else:
                raise ValueError(
                    f"Unrecognized reaction rate type {reaction.rate.type},"
                    f" {reaction.rate.sub_type} for reaction: {reaction.equation}"
                )

            beta = syms.convert_number_to_int(beta)
            low_beta = syms.convert_number_to_int(low_beta)

        cw.writer(fstream, "{")
        cw.writer(
            fstream,
            cw.comment(f"reaction {orig_idx}: {reaction.equation}"),
        )
        if bool(reaction.orders):
            forward_sc, forward_sc_smp = qssa_return_coeff(
                mechanism, species_info, reaction, reaction.orders, syms
            )
        else:
            forward_sc, forward_sc_smp = qssa_return_coeff(
                mechanism, species_info, reaction, reaction.reactants, syms
            )
        if reaction.reversible:
            reverse_sc, reverse_sc_smp = qssa_return_coeff(
                mechanism, species_info, reaction, reaction.products, syms
            )
        else:
            reverse_sc = "0.0"
            reverse_sc_smp = 0.0

        kc_exp_arg, kc_exp_arg_smp = cu.sorted_kc_exp_arg(
            mechanism, species_info, reaction, syms
        )
        kc_conv_inv, kc_conv_inv_smp = cu.fkc_conv_inv(
            mechanism, species_info, reaction, syms
        )

        alpha = 1.0
        if not third_body and not falloff:
            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(
                    fstream,
                    cw.comment(f"qf[{idx}] = k_f[{idx}] * ({forward_sc});"),
                )
                cw.writer(fstream, f"qf[{idx}] = 0.0;")
                syms.qf_qss_smp[idx] = 0.0
            else:
                cw.writer(
                    fstream,
                    f"qf[{idx}] = k_f[{idx}] * ({forward_sc});",
                )
                syms.qf_qss_smp[idx] = syms.kf_qss_smp[idx] * forward_sc_smp

        elif (
            not falloff
            and len(reaction.third_body.efficiencies) == 1
            and isclose(reaction.third_body.default_efficiency, 0.0)
        ):
            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(
                    fstream,
                    cw.comment(f"qf[{idx}] = k_f[{idx}] * ({forward_sc});"),
                )
                cw.writer(fstream, f"qf[{idx}] = 0.0;")
                syms.qf_qss_smp[idx] = 0.0
            else:
                cw.writer(
                    fstream,
                    f"qf[{idx}] = k_f[{idx}] * ({forward_sc});",
                )
                syms.qf_qss_smp[idx] = syms.kf_qss_smp[idx] * forward_sc_smp

        elif not falloff:
            alpha, alpha_smp = cu.enhancement_d(mechanism, species_info, reaction, syms)
            cw.writer(fstream, f"const amrex::Real Corr = {alpha};")
            corr_smp = alpha_smp
            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(
                    fstream,
                    cw.comment(f"qf[{idx}] = Corr * k_f[{idx}] * ({forward_sc});"),
                )
                cw.writer(fstream, f"qf[{idx}] = 0.0;")
                syms.qf_qss_smp[idx] = 0.0
            else:
                cw.writer(
                    fstream,
                    f"qf[{idx}] = Corr * k_f[{idx}] * ({forward_sc});",
                )
                syms.qf_qss_smp[idx] = corr_smp * syms.kf_qss_smp[idx] * forward_sc_smp
        else:
            alpha, alpha_smp = cu.enhancement_d(mechanism, species_info, reaction, syms)
            cw.writer(fstream, f"amrex::Real Corr = {alpha};")
            corr_smp = alpha_smp
            cw.writer(
                fstream,
                f"const amrex::Real redP = Corr / k_f[{idx}] *"
                f" {10 ** (-dim * 6) * low_pef.m * 10 ** 3 ** dim:.15g} ",
            )
            coeff = 10 ** (-dim * 6) * low_pef.m * 10 ** (3**dim)
            redp_smp = corr_smp / syms.kf_qss_smp[idx] * coeff
            cw.writer(
                fstream,
                f"           * exp({low_beta:.15g}  * logT -"
                f" {(1.0 / cc.Rc / cc.ureg.kelvin).m * low_ae.m:.15g} *"
                " invT);",
            )
            coeff = (1.0 / cc.Rc / cc.ureg.kelvin).m * low_ae.m
            redp_smp *= sme.exp(low_beta * syms.logT_smp - coeff * syms.invT_smp)
            if is_troe:
                cw.writer(fstream, "const amrex::Real F = redP / (1.0 + redP);")
                f_smp = redp_smp / (1.0 + redp_smp)
                cw.writer(fstream, "const amrex::Real logPred = log10(redP);")
                log_pred_smp = sme.log(redp_smp, 10)
                cw.writer(fstream, "const amrex::Real logFcent = log10(")
                int_smp = 0
                if abs(troe[1]) > 1.0e-100:
                    if 1.0 - troe[0] != 0:
                        cw.writer(
                            fstream,
                            f"    {1.0 - troe[0]:.15g} * exp(-T * {1 / troe[1]:.15g})",
                        )
                        first_factor = syms.convert_number_to_int(1.0 - troe[0])
                        second_factor = syms.convert_number_to_int(-1 / troe[1])
                        int_smp += first_factor * sme.exp(syms.T_smp * second_factor)
                else:
                    cw.writer(fstream, "     0.0 ")
                    int_smp += 0.0
                if abs(troe[2]) > 1.0e-100:
                    if troe[0] != 0:
                        cw.writer(
                            fstream,
                            f"    + {troe[0]:.15g} * exp(-T * {1 / troe[2]:.15g})",
                        )
                        first_factor = syms.convert_number_to_int(troe[0])
                        second_factor = syms.convert_number_to_int(-1 / troe[2])
                        int_smp += first_factor * sme.exp(syms.T_smp * second_factor)
                else:
                    cw.writer(fstream, "     0.0 ")
                    int_smp += 0.0
                if ntroe == 4:
                    if troe[3] < 0:
                        cw.writer(
                            fstream,
                            f"    + exp({-troe[3]:.15g} * invT));",
                        )
                        first_factor = syms.convert_number_to_int(-troe[3])
                        int_smp += sme.exp(first_factor * syms.invT_smp)
                    else:
                        cw.writer(
                            fstream,
                            f"    + exp(-{troe[3]:.15g} * invT));",
                        )
                        first_factor = syms.convert_number_to_int(-troe[3])
                        int_smp += sme.exp(first_factor * syms.invT_smp)
                else:
                    cw.writer(fstream, "    + 0.0);")
                    int_smp += 0.0
                log_fcent_smp = sme.log(int_smp, 10)

                cw.writer(
                    fstream,
                    "const amrex::Real troe_c = -0.4 - 0.67 * logFcent;",
                )
                troe_c_smp = -0.4 - 0.67 * log_fcent_smp
                cw.writer(
                    fstream,
                    "const amrex::Real troe_n = 0.75 - 1.27 * logFcent;",
                )
                troe_n_smp = 0.75 - 1.27 * log_fcent_smp
                cw.writer(
                    fstream,
                    "const amrex::Real troe = (troe_c + logPred)"
                    + " / (troe_n - 0.14 * (troe_c + logPred));",
                )
                troe_smp = (troe_c_smp + log_pred_smp) / (
                    troe_n_smp - 0.14 * (troe_c_smp + log_pred_smp)
                )
                cw.writer(
                    fstream,
                    "const amrex::Real F_troe = exp(M_LN10 * logFcent /"
                    " (1.0 + troe * troe));",
                )
                f_troe_smp = 10 ** (log_fcent_smp / (1.0 + troe_smp * troe_smp))
                cw.writer(fstream, "Corr = F * F_troe;")
                corr_smp = f_smp * f_troe_smp
                if remove_forward:
                    cw.writer(fstream, cw.comment("Remove forward reaction"))
                    cw.writer(
                        fstream,
                        cw.comment(f"qf[{idx}]  = Corr * k_f[{idx}] * ({forward_sc});"),
                    )
                    cw.writer(fstream, f"qf[{idx}]  = 0.0;")
                    syms.qf_qss_smp[idx] = 0.0
                else:
                    cw.writer(
                        fstream,
                        f"qf[{idx}]  = Corr * k_f[{idx}] * ({forward_sc});",
                    )
                    syms.qf_qss_smp[idx] = (
                        corr_smp * syms.kf_qss_smp[idx] * forward_sc_smp
                    )
            elif nlindemann_qssa > 0:
                cw.writer(fstream, "Corr = redP / (1.0 + redP);")
                corr_smp = redp_smp / (1.0 + redp_smp)
                if remove_forward:
                    cw.writer(fstream, cw.comment("Remove forward reaction"))
                    cw.writer(
                        fstream,
                        cw.comment(f"qf[{idx}] = Corr * k_f[{idx}] * ({forward_sc});"),
                    )
                    cw.writer(fstream, f"qf[{idx}] = 0.0;")
                    syms.qf_qss_smp[idx] = 0.0
                else:
                    cw.writer(
                        fstream,
                        f"qf[{idx}] = Corr * k_f[{idx}] * ({forward_sc});",
                    )
                    syms.qf_qss_smp[idx] = (
                        corr_smp * syms.kf_qss_smp[idx] * forward_sc_smp
                    )

        if kc_conv_inv:
            if alpha == 1.0:
                if reverse_sc_smp == 0:
                    cw.writer(
                        fstream,
                        f"qr[{idx}] = 0.0;",
                    )
                else:
                    cw.writer(
                        fstream,
                        f"qr[{idx}] = k_f[{idx}] * exp(-({kc_exp_arg})) *"
                        f" ({kc_conv_inv}) * ({reverse_sc});",
                    )
                syms.qr_qss_smp[idx] = (
                    syms.kf_qss_smp[idx]
                    * sme.exp(-kc_exp_arg_smp)
                    * kc_conv_inv_smp
                    * reverse_sc_smp
                )
            else:
                if reverse_sc_smp == 0:
                    cw.writer(
                        fstream,
                        f"qr[{idx}] = 0.0;",
                    )
                else:
                    cw.writer(
                        fstream,
                        f"qr[{idx}] = Corr * k_f[{idx}] *"
                        f" exp(-({kc_exp_arg})) * ({kc_conv_inv}) *"
                        f" ({reverse_sc});",
                    )
                syms.qr_qss_smp[idx] = (
                    corr_smp
                    * syms.kf_qss_smp[idx]
                    * sme.exp(-kc_exp_arg_smp)
                    * kc_conv_inv_smp
                    * reverse_sc_smp
                )
        else:
            if alpha == 1.0:
                if reverse_sc_smp == 0:
                    cw.writer(
                        fstream,
                        f"qr[{idx}] = 0.0;",
                    )
                else:
                    cw.writer(
                        fstream,
                        f"qr[{idx}] = k_f[{idx}] * exp(-({kc_exp_arg})) *"
                        f" ({reverse_sc});",
                    )
                syms.qr_qss_smp[idx] = (
                    syms.kf_qss_smp[idx] * sme.exp(-kc_exp_arg_smp) * reverse_sc_smp
                )
            else:
                if reverse_sc_smp == 0.0:
                    cw.writer(
                        fstream,
                        f"qr[{idx}] = 0.0;",
                    )
                else:
                    cw.writer(
                        fstream,
                        f"qr[{idx}] = Corr * k_f[{idx}] *"
                        f" exp(-({kc_exp_arg})) * ({reverse_sc});",
                    )
                syms.qr_qss_smp[idx] = (
                    corr_smp
                    * syms.kf_qss_smp[idx]
                    * sme.exp(-kc_exp_arg_smp)
                    * reverse_sc_smp
                )

        cw.writer(fstream, "}")

    cw.writer(fstream)
    cw.writer(fstream, "return;")
    cw.writer(fstream, "}")

    return


def qssa_component_functions(
    fstream,
    mechanism,
    species_info,
    reaction_info,
    syms,
    helper_names_to_print,
    intermediate_names_to_print,
):
    """QSSA component functions."""
    itroe = reaction_info.index[0:2]
    isri = reaction_info.index[1:3]
    ilindemann = reaction_info.index[2:4]
    i3body = reaction_info.index[3:5]
    isimple = reaction_info.index[4:6]
    ispecial = reaction_info.index[5:7]

    print("troe index range is: ", itroe)
    print("sri index range is: ", isri)
    print("lindemann index range is: ", ilindemann)
    print("3body index range is: ", i3body)
    print("simple index range is: ", isimple)
    print("special index range is: ", ispecial)

    ntroe_qssa = 0
    nsri_qssa = 0
    nlindemann_qssa = 0
    n3body_qssa = 0
    nsimple_qssa = 0
    nspecial_qssa = 0

    itroe_qssa = [0, 0]
    isri_qssa = [0, 0]
    ilindemann_qssa = [0, 0]
    i3body_qssa = [0, 0]
    isimple_qssa = [0, 0]
    ispecial_qssa = [0, 0]

    troe_first = True
    sri_first = True
    lindemann_first = True
    threebody_first = True
    simple_first = True
    special_first = True

    for reac_id in reaction_info.qssa_reactions:
        if reac_id >= itroe[0] and reac_id < itroe[1]:
            print(
                "reaction ",
                reac_id,
                mechanism.reaction(reac_id).equation,
                " goes in troe",
            )
            ntroe_qssa += 1
            if troe_first:
                itroe_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                troe_first = False
            itroe_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1
        if reac_id >= isri[0] and reac_id < isri[1]:
            print(
                "reaction ",
                reac_id,
                mechanism.reaction(reac_id).equation,
                " goes in sri",
            )
            nsri_qssa += 1
            if sri_first:
                isri_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                sri_first = False
            isri_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1
        if reac_id >= ilindemann[0] and reac_id < ilindemann[1]:
            print(
                "reaction ",
                reac_id,
                mechanism.reaction(reac_id).equation,
                " goes in lindemann",
            )
            nlindemann_qssa += 1
            if lindemann_first:
                ilindemann_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                lindemann_first = False
            ilindemann_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1
        if reac_id >= i3body[0] and reac_id < i3body[1]:
            print(
                "reaction ",
                reac_id,
                mechanism.reaction(reac_id).equation,
                " goes in 3body",
            )
            n3body_qssa += 1
            if threebody_first:
                i3body_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                threebody_first = False
            i3body_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1
        if reac_id >= isimple[0] and reac_id < isimple[1]:
            nsimple_qssa += 1
            if simple_first:
                isimple_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                simple_first = False
            isimple_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1
        if reac_id >= ispecial[0] and reac_id < ispecial[1]:
            print(
                "reaction ",
                reac_id,
                mechanism.reaction(reac_id).equation,
                " goes in special",
            )
            nspecial_qssa += 1
            if special_first:
                ispecial_qssa[0] = reaction_info.qssa_reactions.index(reac_id)
                special_first = False
            ispecial_qssa[1] = reaction_info.qssa_reactions.index(reac_id) + 1

    assert len(reaction_info.index) == 7

    # k_f_qssa function
    cw.writer(fstream)
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void comp_k_f_qss"
        + "(const amrex::Real T, const amrex::Real invT, const amrex::Real logT,"
        " amrex::Real * k_f)",
    )
    cw.writer(fstream, "{")
    for index, qssa_reac in enumerate(reaction_info.qssa_reactions):
        reaction = mechanism.reaction(qssa_reac)
        cw.writer(
            fstream,
            cw.comment(f"reaction {qssa_reac}: {reaction.equation}"),
        )

        if bool(reaction.orders):
            dim = cu.phase_space_units(reaction.orders)
        else:
            dim = cu.phase_space_units(reaction.reactants)
        third_body = reaction.third_body is not None
        falloff = reaction.rate.type == "falloff"
        is_troe = reaction.rate.sub_type == "Troe"
        is_sri = reaction.rate.sub_type == "Sri"
        is_lindemann = reaction.rate.sub_type == "Lindemann"
        aeuc = cu.activation_energy_units()
        if not third_body and not falloff:
            # Case 3 !PD, !TB
            ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
            pef = (reaction.rate.pre_exponential_factor * ctuc).to_base_units()
            beta = reaction.rate.temperature_exponent
            ae = (reaction.rate.activation_energy * cc.ureg.joule / cc.ureg.kmol).to(
                aeuc
            )
        elif not falloff:
            # Case 2 !PD, TB
            ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), -dim)
            pef = (reaction.rate.pre_exponential_factor * ctuc).to_base_units()
            beta = reaction.rate.temperature_exponent
            ae = (reaction.rate.activation_energy * cc.ureg.joule / cc.ureg.kmol).to(
                aeuc
            )
        else:
            # Case 2 !PD, TB
            ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
            pef = (
                reaction.rate.high_rate.pre_exponential_factor * ctuc
            ).to_base_units()
            beta = reaction.rate.high_rate.temperature_exponent
            ae = (
                reaction.rate.high_rate.activation_energy * cc.ureg.joule / cc.ureg.kmol
            ).to(aeuc)

            # low_pef = (
            # reaction.rate.low_rate.pre_exponential_factor * ctuc
            # ).to_base_units()
            # low_beta = reaction.rate.low_rate.temperature_exponent
            # low_ae = (
            # reaction.rate.low_rate.activation_energy
            # * cc.ureg.joule
            # / cc.ureg.kmol
            # ).to(aeuc)
            if is_troe:
                pass
                # troe = reaction.rate.falloff_coeffs
                # ntroe = len(troe)
            elif is_sri:
                pass
                # sri = reaction.rate.falloff_coeffs
                # nsri = len(sri)
            elif is_lindemann:
                pass
            else:
                raise ValueError(
                    f"Unrecognized reaction rate type {reaction.rate.type},"
                    f" {reaction.rate.sub_type} for reaction: {reaction.equation}"
                )

        cw.writer(fstream, f"k_f[{index}] = {pef.m:.15g}")
        syms.kf_qss_smp_tmp[index] = pef.m

        if (beta == 0) and (ae == 0):
            cw.writer(fstream, "           ;")
        else:
            if ae == 0:
                cw.writer(fstream, f"           * exp(({beta:.15g}) * logT);")
                syms.kf_qss_smp_tmp[index] *= sme.exp(beta * syms.logT_smp)
            elif beta == 0:
                cw.writer(
                    fstream,
                    "           *"
                    f" exp(-({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g}) *"
                    " invT);",
                )
                coeff = (((1.0 / cc.Rc / cc.ureg.kelvin)) * ae).magnitude
                syms.kf_qss_smp_tmp[index] *= sme.exp(-coeff * syms.invT_smp)

            else:
                cw.writer(
                    fstream,
                    f"           * exp(({beta:.15g}) * logT -"
                    f" ({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g}) * invT);",
                )
                coeff = (((1.0 / cc.Rc / cc.ureg.kelvin)) * ae).magnitude
                syms.kf_qss_smp_tmp[index] *= sme.exp(
                    beta * syms.logT_smp - coeff * syms.invT_smp
                )

    cw.writer(fstream)
    cw.writer(fstream, "return;")
    cw.writer(fstream, "}")

    # qssa concentrations
    cw.writer(fstream)
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void comp_sc_qss"
        + "(amrex::Real * sc_qss, amrex::Real * qf_co, amrex::Real * qr_co)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real epsilon = 1e-12;")
    cw.writer(fstream)

    # print()
    print("** species_info.qssa_info.decouple_index:")
    print(species_info.qssa_info.decouple_index)
    print(list(species_info.qssa_info.needs.keys()))
    print(list(species_info.qssa_info.group.keys()))

    coeff_submatrix_smp = []
    rhs_submatrix_smp = []

    for i in species_info.qssa_info.decouple_index:
        symbol = species_info.qssa_info.decouple_index[i]
        print("... Dealing with Spec/Group ", symbol)

        # This case does not happen for dodecane_lu_qss, do later
        if symbol in list(species_info.qssa_info.needs.keys()):
            print("    Simple case, single group")
            cpp_var_str_symbol = cf.format_species(symbol)
            denominator = cpp_var_str_symbol + "_denom"
            numerator = cpp_var_str_symbol + "_num"
            cw.writer(
                fstream,
                cw.comment(
                    "QSSA species "
                    + str(species_info.qssa_species_list.index(symbol))
                    + ": "
                    + symbol
                ),
            )
            cw.writer(fstream)

            # RHS
            # cut line if too big !
            long_line_elements = (species_info.qssa_info.rhs[symbol]).split()
            numerator_smp = 1e-12 + species_info.qssa_info.rhs_smp[symbol]
            syms.intermediate_terms_smp[numerator] = species_info.qssa_info.rhs_smp[
                symbol
            ]
            len_long_line = len(long_line_elements)
            # if we have more than 7 elements
            if len_long_line > 7:
                # treat first line separately with the epsilon
                cw.writer(
                    fstream,
                    f"amrex::Real {numerator} = epsilon"
                    f" {' '.join(long_line_elements[0:7])}",
                )
                # proceed by strides of 7
                for kk in range(7, len_long_line, 7):
                    # if there are less than 7 elems left then we are at the end of the list
                    if len(long_line_elements[kk : kk + 7]) < 7:
                        cw.writer(
                            fstream,
                            "                   "
                            f" {' '.join(long_line_elements[kk:kk + 7])};",
                        )
                    # if there are 7 elems we are ...
                    else:
                        # either are in the middle of the list
                        if len(long_line_elements[kk:]) > 7:
                            cw.writer(
                                fstream,
                                "                   "
                                f" {' '.join(long_line_elements[kk:kk + 7])}",
                            )
                        # or at the end but list number was a multiple of 7
                        else:
                            cw.writer(
                                fstream,
                                "                   "
                                f" {' '.join(long_line_elements[kk:kk + 7])};",
                            )
            # if we have less than 7 elements just write them
            else:
                cw.writer(
                    fstream,
                    f"amrex::Real {numerator} = epsilon"
                    f" {species_info.qssa_info.rhs[symbol]};",
                )
            # COEFF
            cw.writer(
                fstream,
                f"amrex::Real {denominator} = epsilon"
                f" {species_info.qssa_info.coeff[symbol]};",
            )
            denominator_smp = 1e-12 + species_info.qssa_info.coeff_smp[symbol]
            syms.intermediate_terms_smp[denominator] = species_info.qssa_info.coeff_smp[
                symbol
            ]
            cw.writer(fstream)
            cw.writer(
                fstream,
                f"sc_qss[{species_info.qssa_species_list.index(symbol)}] ="
                f" - {numerator}/{denominator};",
            )
            syms.sc_qss_smp[species_info.qssa_species_list.index(symbol)] = (
                -numerator_smp / denominator_smp
            )
            cw.writer(fstream)
        # This case happens for dodecane_lu_qss
        if symbol in list(species_info.qssa_info.group.keys()):
            print(
                "    Though case. Submatrix has size ",
                len(species_info.qssa_info.group[symbol]),
                "x",
                len(species_info.qssa_info.group[symbol]),
            )
            coeff_submatrix = [
                ["0"] * len(species_info.qssa_info.group[symbol])
                for i in range(len(species_info.qssa_info.group[symbol]))
            ]
            coeff_submatrix_smp = [
                [0] * len(species_info.qssa_info.group[symbol])
                for i in range(len(species_info.qssa_info.group[symbol]))
            ]
            rhs_submatrix = ["0"] * len(species_info.qssa_info.group[symbol])
            rhs_submatrix_smp = [0] * len(species_info.qssa_info.group[symbol])
            gr_species = species_info.qssa_info.group[symbol]
            print("    Species involved :", gr_species)
            cw.writer(
                fstream,
                cw.comment("QSSA coupling between " + ("  ").join(gr_species)),
            )
            for index, species in enumerate(gr_species):
                print("      x Dealing with spec", species, index)
                cw.writer(
                    fstream,
                    cw.comment(
                        "QSSA species "
                        + str(species_info.qssa_species_list.index(species))
                        + ": "
                        + species
                    ),
                )
                cw.writer(fstream)

                cpp_var_str_species = cf.format_species(species)
                denominator = cpp_var_str_species + "_denom"
                numerator = cpp_var_str_species + "_num"

                # RHS
                # cut line if too big !
                long_line_elements = (species_info.qssa_info.rhs[species]).split()

                len_long_line = len(long_line_elements)
                numerator_smp = 1e-12 + species_info.qssa_info.rhs_smp[species]
                syms.intermediate_terms_smp[numerator] = species_info.qssa_info.rhs_smp[
                    species
                ]
                # if we have more than 7 elements
                if len_long_line > 7:
                    # treat first line separately with the epsilon
                    cw.writer(
                        fstream,
                        f"amrex::Real {numerator} = epsilon"
                        f" {' '.join(long_line_elements[0:7])}",
                    )
                    # proceed by strides of 7
                    for kk in range(7, len_long_line, 7):
                        # if there are less than 7 elems left then we are at the end of the list
                        if len(long_line_elements[kk : kk + 7]) < 7:
                            cw.writer(
                                fstream,
                                "                   "
                                f" {' '.join(long_line_elements[kk:kk + 7])};",
                            )
                        # if there are 7 elems we are ...
                        else:
                            # either are in the middle of the list
                            if len(long_line_elements[kk:]) > 7:
                                cw.writer(
                                    fstream,
                                    "                   "
                                    f" {' '.join(long_line_elements[kk:kk + 7])}",
                                )
                            # or at the end but list number was a multiple of 7
                            else:
                                cw.writer(
                                    fstream,
                                    "                   "
                                    f" {' '.join(long_line_elements[kk:kk + 7])};",
                                )
                # if we have less than 7 elements just write them
                else:
                    cw.writer(
                        fstream,
                        f"amrex::Real {numerator} = epsilon"
                        f" {species_info.qssa_info.rhs[species]};",
                    )
                # COEFF
                # cut line if too big !
                long_line_elements = (species_info.qssa_info.coeff[species]).split()
                len_long_line = len(long_line_elements)
                denominator_smp = 1e-12 + species_info.qssa_info.coeff_smp[species]
                syms.intermediate_terms_smp[denominator] = (
                    species_info.qssa_info.coeff_smp[species]
                )
                # if we have more than 7 elements
                if len_long_line > 7:
                    # treat first line separately with the epsilon
                    cw.writer(
                        fstream,
                        f"amrex::Real {denominator} = epsilon"
                        f" {' '.join(long_line_elements[0:7])}",
                    )
                    # proceed by strides of 7
                    for kk in range(7, len_long_line, 7):
                        # if there are less than 7 elems left then we are at the end of the list
                        if len(long_line_elements[kk : kk + 7]) < 7:
                            cw.writer(
                                fstream,
                                "                   "
                                f" {' '.join(long_line_elements[kk:kk + 7])};",
                            )
                        # if there are 7 elems we are ...
                        else:
                            # either are in the middle of the list
                            if len(long_line_elements[kk:]) > 7:
                                cw.writer(
                                    fstream,
                                    "                   "
                                    f" {' '.join(long_line_elements[kk:kk + 7])}",
                                )
                            # or at the end but list number was a multiple of 7
                            else:
                                cw.writer(
                                    fstream,
                                    "                   "
                                    f" {' '.join(long_line_elements[kk:kk + 7])};",
                                )
                # if we have less than 7 elements just write them
                else:
                    cw.writer(
                        fstream,
                        f"amrex::Real {denominator} = epsilon"
                        f" {species_info.qssa_info.coeff[species]};",
                    )
                # RHS
                cw.writer(
                    fstream,
                    "amrex::Real "
                    + cf.format_species(species)
                    + "_rhs = -"
                    + numerator
                    + "/"
                    + denominator
                    + ";",
                )
                # species_rhs_smp = -numerator_smp / denominator_smp

                syms.intermediate_terms_smp[cf.format_species(species) + "_rhs"] = (
                    -numerator_smp / denominator_smp
                )
                cw.writer(fstream)

                for j in range(len(gr_species)):
                    if j == index:
                        coeff_submatrix[index][j] = "1"
                        coeff_submatrix_smp[index][j] = 1
                    else:
                        if (
                            species_info.qssa_info.qssa_coeff[species][gr_species[j]]
                            != "0.0"
                        ):
                            coeff_submatrix[index][j] = (
                                cf.format_species(species)
                                + "_"
                                + cf.format_species(gr_species[j])
                            )
                            # let us assume for now these lines are not too big
                            cw.writer(
                                fstream,
                                "amrex::Real "
                                + cf.format_species(species)
                                + "_"
                                + cf.format_species(gr_species[j])
                                + " = (epsilon "
                                + species_info.qssa_info.qssa_coeff[species][
                                    gr_species[j]
                                ]
                                + ")/"
                                + denominator
                                + ";",
                            )

                            syms.intermediate_terms_smp[
                                cf.format_species(species)
                                + "_"
                                + cf.format_species(gr_species[j])
                            ] = (
                                1e-12
                                + species_info.qssa_info.qssa_coeff_smp[species][
                                    gr_species[j]
                                ][0]
                            ) / denominator_smp
                            coeff_submatrix_smp[index][j] = syms.intermediate_terms_smp[
                                cf.format_species(species)
                                + "_"
                                + cf.format_species(gr_species[j])
                            ]
                            # REMOVE IF NO DEBUG
                            # coeff_submatrix_smp[index][
                            #    j
                            # ] = sme.symbols(
                            #    cf.format_species(species)
                            #    + "_"
                            #    + cf.format_species(gr_species[j])
                            # )

                # REMOVE IF NO DEBUG
                # for key in list(syms.intermediate_terms_smp.keys()):
                #    syms.intermediate_terms_smp[key] = sme.symbols(key)

                cw.writer(fstream)
                rhs_submatrix[index] = cf.format_species(species) + "_rhs"
                rhs_submatrix_smp[index] = syms.intermediate_terms_smp[
                    cf.format_species(species) + "_rhs"
                ]
                # REMOVE IF NO DEBUG
                # rhs_submatrix_smp[index] = sme.symbols(
                #    cf.format_species(species) + "_rhs"
                # )

            a, x, b, intermediate_helpers, x_smp = gauss_pivoting(
                species_info,
                coeff_submatrix,
                rhs_submatrix,
                coeff_submatrix_smp,
                rhs_submatrix_smp,
                syms,
            )

            print("X is ", x)
            # print("X_smp is ", x_smp)

            cw.writer(fstream, cw.comment("Putting it all together"))
            for helper in intermediate_helpers:
                if helper in species_info.qssa_info.list_of_intermediate_helpers:
                    cw.writer(
                        fstream,
                        f"{helper} = {intermediate_helpers[helper]};",
                    )
                    # print(helper  + "  ->  " +intermediate_helpers[helper])
                else:
                    cw.writer(
                        fstream,
                        f"amrex::Real {helper} = {intermediate_helpers[helper]};",
                    )
                    # print(helper  + "  ->  " +intermediate_helpers[helper])
                    # print(intermediate_helpers[helper])
                    species_info.qssa_info.list_of_intermediate_helpers.append(helper)
            # stop
            for count in range(len(gr_species)):
                max_index = len(gr_species) - 1
                species = gr_species[max_index - count]

                # cut line if too big !
                long_line_elements = x[max_index - count].split()
                len_long_line = len(long_line_elements)
                # print("SPECIES "+ str(count) + " = ")
                syms.sc_qss_smp[species_info.qssa_species_list.index(species)] = x_smp[
                    max_index - count
                ]
                # print(long_line_elements)
                # if we have more than 4 elements
                if len_long_line > 4:
                    # treat first line separately
                    cw.writer(
                        fstream,
                        "sc_qss["
                        + str(species_info.qssa_species_list.index(species))
                        + "] = "
                        + " ".join(long_line_elements[0:4]),
                    )
                    # proceed by strides of 4
                    for kk in range(4, len_long_line, 4):
                        # if there are less than 4 elems left then we are at the end of the list
                        if len(long_line_elements[kk : kk + 4]) < 4:
                            cw.writer(
                                fstream,
                                "                   "
                                f" {' '.join(long_line_elements[kk:kk + 4])};",
                            )
                        # if there are 4 elems we are ...
                        else:
                            # either are in the middle of the list
                            if len(long_line_elements[kk:]) > 4:
                                cw.writer(
                                    fstream,
                                    "                   "
                                    f" {' '.join(long_line_elements[kk:kk + 4])}",
                                )
                            # or at the end but list number was a multiple of 4
                            else:
                                cw.writer(
                                    fstream,
                                    "                   "
                                    f" {' '.join(long_line_elements[kk:kk + 4])};",
                                )
                # if we have less than 4 elements just write them
                else:
                    cw.writer(
                        fstream,
                        "sc_qss["
                        + str(species_info.qssa_species_list.index(species))
                        + "] = "
                        + x[max_index - count]
                        + ";",
                    )
                cw.writer(fstream)

    cw.writer(fstream)
    cw.writer(fstream, "return;")
    cw.writer(fstream, "}")

    return


def gauss_pivoting(species_info, a, b=None, a_smp=None, b_smp=None, syms=None):
    """Gauss pivoting."""
    print()
    print("In Gauss pivot")

    pivots = []
    pivots_smp = []
    intermediate_helpers = OrderedDict()
    helper_counters = 0

    x = [""] * len(a[0])
    x_smp = [0] * len(a[0])

    for i in range(len(a[0])):
        x[i] = "X" + str(i)

    if b is None:
        for i in range(len(a[0])):
            b[i] = "B" + str(i)

    # Get species names:
    species = ["0" for i in range(len(b))]
    for member in range(len(b)):
        hold = str(b[member])
        hold = hold[:-4]
        species[member] = cf.format_species(hold)

    print("Species involved are: ", species)
    print()

    a_num = np.zeros([len(a[0]), len(a[0])])
    for i in range(len(a[0])):
        for j in range(len(a[0])):
            if a[i][j] != "0":
                a_num[i, j] = 1

    print("--A", a)
    print("--B", b)

    indi, indj = np.nonzero(a_num)

    n = len(b)

    for k in range(n - 1):
        pivot = a[k][k]
        pivot_smp = a_smp[k][k]

        # swap lines if needed
        if pivot == 0:
            temp = np.array(a[k + 1][:])
            a[k + 1][:] = a[k][:]
            a[k][:] = temp
            temp_smp = np.array(a_smp[k + 1][:])
            a_smp[k + 1][:] = a_smp[k][:]
            a_smp[k][:] = temp_smp

            temp = str(b[k + 1])
            b[k + 1] = b[k]
            b[k] = temp
            temp_smp = str(b_smp[k + 1])
            b_smp[k + 1] = b_smp[k]
            b_smp[k] = temp_smp

            pivot = a[k][k]
            pivot_smp = a_smp[k][k]

        print()
        print("   **ROW of pivot ", k, " and pivot is ", pivot)
        pivots.append(pivot)
        pivots_smp.append(pivot_smp)
        # print("pivot = ", pivot)
        # print("pivot_smp = ", pivot_smp)

        for i in range(k, len(b) - 1):
            num = a[i + 1][k]
            num_smp = a_smp[i + 1][k]
            print(
                "       xx Treating ROW ",
                i + 1,
                "with numerator ",
                num,
                "subst fact will be",
                num,
                "/",
                pivot,
            )
            # print("num = ", num)
            # print("num_smp = ", num_smp)

            if num == "0":
                print("        !! No need to do anything, already zeroed ... skip")
                continue
            b = list(b)
            print("          - B starts with: ")
            print("           ", b)
            if num != "0":
                if pivot != "1":
                    if num != "1":
                        helper = num + "/" + pivot
                        helper_smp = num_smp / pivot_smp
                        helper_name = "H_" + str(helper_counters)
                        # REMOVE IF NO DEBUG
                        # helper_smp = sme.symbols(helper_name)
                        intermediate_helpers[helper_name] = helper
                        syms.intermediate_helpers_smp[helper_name] = helper_smp
                        b[i + 1] = b[i + 1] + " -" + b[int(k)] + "*" + helper_name

                        b_smp[i + 1] = b_smp[i + 1] - b_smp[int(k)] * helper_smp
                        b[i + 1] = "(" + b[i + 1] + ")"
                        helper_counters += 1
                    else:
                        helper = 1 + "/" + pivot
                        helper_smp = 1 / pivot_smp
                        helper_name = "H_" + str(helper_counters)
                        # REMOVE IF NO DEBUG
                        # helper_smp = sme.symbols(helper_name)
                        intermediate_helpers[helper_name] = helper
                        syms.intermediate_helpers_smp[helper_name] = helper_smp
                        print(" IN THIS CASE !! CHECK THAT ITS OK !! ")
                        b[i + 1] = b[i + 1] + " -" + b[int(k)] + "*" + helper_name
                        b_smp[i + 1] = b_smp[i + 1] - b_smp[int(k)] * helper_smp
                        b[i + 1] = "(" + b[i + 1] + ")"
                        helper_counters += 1
                else:
                    if num != "1":
                        helper = num
                        helper_smp = num_smp
                        helper_name = "H_" + str(helper_counters)
                        # REMOVE IF NO DEBUG
                        # helper_smp = sme.symbols(helper_name)
                        intermediate_helpers[helper_name] = helper
                        syms.intermediate_helpers_smp[helper_name] = helper_smp
                        b[i + 1] = b[i + 1] + " -" + b[int(k)] + "*" + helper_name
                        b_smp[i + 1] = b_smp[i + 1] - b_smp[int(k)] * helper_smp
                        b[i + 1] = "(" + b[i + 1] + ")"
                        helper_counters += 1
                    else:
                        b[i + 1] = b[i + 1] + " -" + b[int(k)]
                        b_smp[i + 1] = b_smp[i + 1] - b_smp[int(k)]
                        b[i + 1] = "(" + b[i + 1] + ")"

            print("          ... and B ends with: ")
            print("            ", b)

            indi, indj = np.nonzero(a_num)

            for j in indj[indi == (i + 1)]:
                print(
                    "          - Dealing with row elem on column ",
                    j,
                    " : ",
                    a[i + 1][j],
                )
                if j == k:
                    print("            !! 0 ELEM !")
                    a[i + 1][j] = "0"
                    a_smp[i + 1][j] = 0
                else:
                    if a[i + 1][j] != "0":
                        if num != "0":
                            if pivot != "1":
                                if num != "1":
                                    if a[k][j] != "0":
                                        a[i + 1][j] = (
                                            a[i + 1][j]
                                            + " -"
                                            + a[k][j]
                                            + "*"
                                            + helper_name
                                        )
                                        a_smp[i + 1][j] = (
                                            a_smp[i + 1][j] - a_smp[k][j] * helper_smp
                                        )
                                        a[i + 1][j] = "(" + a[i + 1][j] + ")"
                                else:
                                    if a[k][j] != "0":
                                        print(" IN THIS CASE !! CHECK THAT ITS OK !! ")
                                        a[i + 1][j] = (
                                            a[i + 1][j]
                                            + " -"
                                            + a[k][j]
                                            + "*"
                                            + helper_name
                                        )
                                        a_smp[i + 1][j] = (
                                            a_smp[i + 1][j] - a_smp[k][j] * helper_smp
                                        )
                                        a[i + 1][j] = "(" + a[i + 1][j] + ")"
                            else:
                                if num != "1":
                                    if a[k][j] != "0":
                                        a[i + 1][j] = (
                                            a[i + 1][j]
                                            + " -"
                                            + a[k][j]
                                            + "*"
                                            + helper_name
                                        )
                                        a_smp[i + 1][j] = (
                                            a_smp[i + 1][j] - a_smp[k][j] * helper_smp
                                        )
                                        a[i + 1][j] = "(" + a[i + 1][j] + ")"
                                else:
                                    if a[k][j] != "0":
                                        a[i + 1][j] = a[i + 1][j] + " -" + a[k][j]
                                        a_smp[i + 1][j] = a_smp[i + 1][j] - a_smp[k][j]
                                        a[i + 1][j] = "(" + a[i + 1][j] + ")"
                    else:
                        if num != "0":
                            if pivot != "1":
                                if num != "1":
                                    if a[k][j] != "0":
                                        a[i + 1][j] = " -" + a[k][j] + "*" + helper_name
                                        a_smp[i + 1][j] = -a_smp[k][j] * helper_smp
                                        a[i + 1][j] = "(" + a[i + 1][j] + ")"
                                else:
                                    if a[k][j] != "0":
                                        print(" IN THIS CASE !! CHECK THAT ITS OK !! ")
                                        a[i + 1][j] = " -" + a[k][j] + "*" + helper_name
                                        a_smp[i + 1][j] = -a_smp[k][j] * helper_smp
                                        a[i + 1][j] = "(" + a[i + 1][j] + ")"
                            else:
                                if num != "1":
                                    if a[k][j] != "0":
                                        a[i + 1][j] = " -" + a[k][j] + "*" + helper_name
                                        a_smp[i + 1][j] = -a_smp[k][j] * helper_smp
                                        a[i + 1][j] = "(" + a[i + 1][j] + ")"
                                else:
                                    if a[k][j] != "0":
                                        a[i + 1][j] = " -" + a[k][j]
                                        a_smp[i + 1][j] = -a_smp[k][j]
                                        a[i + 1][j] = "(" + a[i + 1][j] + ")"
            print("          ... and updated A is: ")
            print("             ", a)

    for i in range(len(b)):
        x = list(x)
        b[i] = str(b[i])

    # start with last elem
    n = n - 1
    if a[n][n] != "1":
        x[n] = b[n] + "/" + a[n][n]
        x_smp[n] = b_smp[n] / a_smp[n][n]
    else:
        x[n] = b[n]
        x_smp[n] = b_smp[n]

    for i in range(1, n + 1):
        sumprod = ""
        sumprod_smp = 0
        for j in range(i):
            flag = False
            if a[n - i][n - j] != "0":
                if flag:
                    sumprod += " + "
                flag = True
                if a[n - i][n - j] == "1":
                    sumprod += " (" + str(x[n - j]) + ")"
                    sumprod_smp += x_smp[n - j]
                elif j != 0:
                    sumprod += (
                        " +"
                        + a[n - i][n - j]
                        + "*"
                        + "sc_qss["
                        + str(
                            species_info.qssa_species_formatted_list.index(
                                species[n - j]
                            )
                        )
                        + "]"
                    )
                    sumprod_smp += (
                        a_smp[n - i][n - j]
                        * syms.sc_qss_smp[
                            species_info.qssa_species_formatted_list.index(
                                species[n - j]
                            )
                        ]
                    )
                else:
                    sumprod += (
                        a[n - i][n - j]
                        + "*"
                        + "sc_qss["
                        + str(
                            species_info.qssa_species_formatted_list.index(
                                species[n - j]
                            )
                        )
                        + "]"
                    )
                    sumprod_smp += (
                        a_smp[n - i][n - j]
                        * syms.sc_qss_smp[
                            species_info.qssa_species_formatted_list.index(
                                species[n - j]
                            )
                        ]
                    )

        if sumprod == "":
            if a[n - i][n - i] != "1":
                x[n - i] = "(" + b[n - i] + ")/" + a[n - i][n - i]
                x_smp[n - i] = b_smp[n - i] / a_smp[n - i][n - i]
            else:
                x[n - i] = b[n - i]
                x_smp[n - i] = b_smp[n - i]
        else:
            if a[n - i][n - i] == "1":
                x[n - i] = b[n - i] + " -(" + sumprod + ")"
                x_smp[n - i] = b_smp[n - i] - (sumprod_smp)
            else:
                x[n - i] = "(" + b[n - i] + " -(" + sumprod + "))/" + a[n - i][n - i]
                x_smp[n - i] = (b_smp[n - i] - (sumprod_smp)) / a_smp[n - i][n - i]
    print()
    print()

    return a, x, b, intermediate_helpers, x_smp


def qssa_return_coeff(mechanism, species_info, reaction, reagents, syms):
    """QSSA coefficient."""
    if reaction.third_body:
        if len(reaction.third_body.efficiencies) == 1:
            if isclose(reaction.third_body.default_efficiency, 0.0):
                if not (reaction.rate.type == "falloff"):
                    # Skip for third-body falloff reactions with single M
                    reagents = copy.deepcopy(
                        dict(
                            sum(
                                (
                                    Counter(x)
                                    for x in [
                                        reagents,
                                        reaction.third_body.efficiencies,
                                    ]
                                ),
                                Counter(),
                            )
                        )
                    )

    phi = []
    phi_smp = []
    dict_species = {v: i for i, v in enumerate(species_info.all_species_list)}
    sorted_reagents = sorted(reagents.keys(), key=lambda v: dict_species[v])
    for symbol in sorted_reagents:
        coefficient = reagents[symbol]
        if symbol not in species_info.qssa_species_list:
            if float(coefficient) == 1.0:
                conc = f"sc[{species_info.ordered_idx_map[symbol]}]"
                conc_smp = syms.sc_smp[species_info.ordered_idx_map[symbol]]
            else:
                if coefficient.is_integer():
                    conc = "*".join(
                        [f"sc[{species_info.ordered_idx_map[symbol]}]"]
                        * int(coefficient)
                    )
                else:
                    conc = (
                        f"pow(sc[{species_info.ordered_idx_map[symbol]}],"
                        f" {float(coefficient):f})"
                    )
                conc_smp = syms.sc_smp[species_info.ordered_idx_map[symbol]] ** float(
                    coefficient
                )
            phi += [conc]
            phi_smp += [conc_smp]
        if len(phi) < 1:
            phi = ["1.0"]
            if syms.remove_1:
                phi_smp = [1]
            else:
                phi_smp = [1.0]

    if syms.remove_1:
        qssa_coeff_smp = 1
    else:
        qssa_coeff_smp = 1.0
    for phival in phi_smp:
        qssa_coeff_smp *= phival

    qssa_coeff_smp = syms.convert_symb_to_int(qssa_coeff_smp)
    return "*".join(phi), qssa_coeff_smp
