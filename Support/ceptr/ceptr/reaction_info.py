"""Reaction information."""

from collections import OrderedDict

import ceptr.constants as cc
import ceptr.writer as cw


def has_plog_reactions(reactions):
    """Return whether any of the reactions has a PLOG rate, with or without M."""
    return any(r.rate.type == "pressure-dependent-Arrhenius" for r in reactions)


class ReactionInfo:
    """Information on reactions."""

    def __init__(self, mechanism, interface):
        """Initialize the reaction information."""
        self.n_reactions = mechanism.n_reactions
        self.rs = []
        self.rs_unsorted = mechanism.reactions()
        self.is_sorted = False
        self.index = [0]
        self.idxmap = OrderedDict()
        self.remove_id_list = []
        try:
            self.remove_id_list = mechanism.input_data["forward_to_remove_idx"]
        except KeyError:
            pass

        self.qssa_reactions = []
        self.n_qssa_reactions = 0
        self.qfqr_co_idx_map = []

        if interface is not None:
            self.rs_unsorted += interface.reactions()

        self.has_plog_reactions = has_plog_reactions(self.rs_unsorted)

        self.has_reactions = len(self.rs_unsorted) > 0


def sort_reactions(mechanism, interface):
    """Sort reactions."""
    reaction_info = ReactionInfo(mechanism, interface)
    i = 0

    # troe
    for k, r in enumerate(reaction_info.rs_unsorted):
        if r not in reaction_info.rs:
            if r.reaction_type == "falloff-Troe":
                reaction_info.idxmap[k] = i
                reaction_info.rs.append(r)
                i += 1
    reaction_info.index.append(i)

    # sri
    for k, r in enumerate(reaction_info.rs_unsorted):
        if r not in reaction_info.rs:
            if r.reaction_type == "falloff-Sri":
                reaction_info.idxmap[k] = i
                reaction_info.rs.append(r)
                i += 1
    reaction_info.index.append(i)

    # lindemann
    for k, r in enumerate(reaction_info.rs_unsorted):
        if r not in reaction_info.rs:
            if r.reaction_type == "falloff-Lindemann":
                reaction_info.idxmap[k] = i
                reaction_info.rs.append(r)
                i += 1
    reaction_info.index.append(i)

    # three-body
    for k, r in enumerate(reaction_info.rs_unsorted):
        if r not in reaction_info.rs:
            if r.third_body is not None:
                reaction_info.idxmap[k] = i
                reaction_info.rs.append(r)
                i += 1
    reaction_info.index.append(i)

    # simplest case (Arrhenius reactions)
    for k, r in enumerate(reaction_info.rs_unsorted):
        if r not in reaction_info.rs:
            # skip all interface-Arrhenius and sticking-Arrhenius heterogeneous reactions
            # heterogeneous reactions are added after all the homogeneous reactions
            if (
                r.reaction_type == "interface-Arrhenius"
                or r.reaction_type == "sticking-Arrhenius"
            ):
                continue

            if r.third_body is None:
                reaction_info.idxmap[k] = i
                reaction_info.rs.append(r)
                i += 1
    reaction_info.index.append(i)

    # everything else
    for k, r in enumerate(reaction_info.rs_unsorted):
        if r not in reaction_info.rs:
            # skip all interface-Arrhenius and sticking-Arrhenius heterogeneous reactions
            # heterogeneous reactions are added after all the homogeneous reactions
            if (
                r.reaction_type == "interface-Arrhenius"
                or r.reaction_type == "sticking-Arrhenius"
            ):
                continue

            reaction_info.idxmap[k] = i
            reaction_info.rs.append(r)
            i += 1
    reaction_info.index.append(i)

    # surface reactions
    if interface is not None:
        for k, r in enumerate(reaction_info.rs_unsorted):
            if r not in reaction_info.rs and r.reaction_type == "interface-Arrhenius":
                reaction_info.idxmap[k] = i
                reaction_info.rs.append(r)
                i += 1
        reaction_info.index.append(i)

        for k, r in enumerate(reaction_info.rs_unsorted):
            if r not in reaction_info.rs and r.reaction_type == "sticking-Arrhenius":
                reaction_info.idxmap[k] = i
                reaction_info.rs.append(r)
                i += 1
        reaction_info.index.append(i)

    reaction_info.is_sorted = True

    return reaction_info


def rmap(fstream, reaction_info):
    """Write reverse reaction map."""
    rmap = reaction_info.idxmap.keys()
    str_rmap = ",".join(str(x) for x in rmap)
    if reaction_info.has_reactions:
        cw.writer(fstream, f"const int rmap[NUM_REACTIONS] = {{{str_rmap}}};")


def get_rmap(fstream, reaction_info):
    """Write function for reverse reaction map."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns 0-based map of reaction order"))
    cw.writer(fstream, "void GET_RMAP" + cc.sym)

    if reaction_info.has_reactions:
        cw.writer(fstream, "(int * _rmap)")
    else:
        cw.writer(fstream, "(int * /*_rmap*/)")

    cw.writer(fstream, "{")

    if reaction_info.has_reactions:
        cw.writer(fstream, "for (int j=0; j<NUM_REACTIONS; ++j)")
        cw.writer(fstream, "{")
        cw.writer(fstream, "_rmap[j] = rmap[j];")
        cw.writer(fstream, "}")

    cw.writer(fstream, "}")
