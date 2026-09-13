"""Utility functions used across ceptr."""

import copy
from collections import Counter
from math import exp, isclose, log

import ceptr.constants as cc


def intersection(lst1, lst2):
    """Return intersection of two lists."""
    return list(set(lst1).intersection(lst2))


def sc_cutoff(exponent):
    """Return cutoff for sc when using a fractional exponent."""
    if exponent < 0:
        return "1e-16"
    else:
        return "0.0"


def qss_sorted_phase_space(mechanism, species_info, reaction, reagents, syms=None):
    """Get string of phase space."""
    record_symbolic_operations = True
    if syms is None:
        record_symbolic_operations = False

    if bool(reaction.orders):
        list_ord = list(reaction.orders.keys())
        for spec in list_ord:
            if spec not in reagents:
                reagents[spec] = 0.0

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
    if record_symbolic_operations:
        phi_smp = []
    dict_species = {v: i for i, v in enumerate(species_info.all_species_list)}
    sorted_reagents = sorted(reagents.keys(), key=lambda v: dict_species[v])
    n_species = species_info.n_species

    for symbol in sorted_reagents:
        coefficient = reagents[symbol]
        if bool(reaction.orders):
            order = reaction.orders[symbol]
        else:
            order = coefficient
        if symbol in species_info.qssa_species_list:
            if float(order) == 1.0:
                conc = f"sc_qss[{species_info.ordered_idx_map[symbol] - n_species}]"
                if record_symbolic_operations:
                    conc_smp = syms.sc_qss_smp[
                        species_info.ordered_idx_map[symbol] - n_species
                    ]
            else:
                if order.is_integer():
                    conc = "*".join(
                        [f"sc_qss[{species_info.ordered_idx_map[symbol] - n_species}]"]
                        * int(order)
                    )
                elif float(order) == 0.5:
                    conc = (
                        "std::sqrt(std::max(sc_qss"
                        f"[{species_info.ordered_idx_map[symbol] - n_species}],"
                        f" {sc_cutoff(0.5)}))"
                    )
                else:
                    conc = (
                        "pow(sc_qss[std::max("
                        f"{species_info.ordered_idx_map[symbol] - n_species}],"
                        f" {sc_cutoff(order)}), {float(order):f})"
                    )
                if record_symbolic_operations:
                    conc_smp = pow(
                        syms.sc_qss_smp[
                            species_info.ordered_idx_map[symbol] - n_species
                        ],
                        float(order),
                    )
            phi += [conc]
            if record_symbolic_operations:
                phi_smp += [conc_smp]
        else:
            if float(order) == 1.0:
                conc = f"sc[{species_info.ordered_idx_map[symbol]}]"
                if record_symbolic_operations:
                    conc_smp = syms.sc_smp[species_info.ordered_idx_map[symbol]]
            else:
                if float(order) == 2.0:
                    conc = (
                        f"(sc[{species_info.ordered_idx_map[symbol]}] *"
                        f" sc[{species_info.ordered_idx_map[symbol]}])"
                    )
                    if record_symbolic_operations:
                        conc_smp = (
                            syms.sc_smp[species_info.ordered_idx_map[symbol]]
                            * syms.sc_smp[species_info.ordered_idx_map[symbol]]
                        )
                elif order.is_integer():
                    conc = "*".join(
                        [f"sc[{species_info.ordered_idx_map[symbol]}]"] * int(order)
                    )
                    if record_symbolic_operations:
                        conc_smp = syms.sc_smp[
                            species_info.ordered_idx_map[symbol]
                        ] ** int(order)
                elif float(order) == 0.5:
                    conc = (
                        f"std::sqrt(std::max(sc[{species_info.ordered_idx_map[symbol]}],"
                        f" {sc_cutoff(0.5)}))"
                    )
                    if record_symbolic_operations:
                        conc_smp = pow(
                            syms.sc_smp[species_info.ordered_idx_map[symbol]],
                            float(order),
                        )
                else:
                    conc = (
                        f"pow(std::max(sc[{species_info.ordered_idx_map[symbol]}],"
                        f" {sc_cutoff(order)}), {float(order):f})"
                    )
                    if record_symbolic_operations:
                        conc_smp = pow(
                            syms.sc_smp[species_info.ordered_idx_map[symbol]],
                            float(order),
                        )
            phi += [conc]
            if record_symbolic_operations:
                phi_smp += [conc_smp]
    if not record_symbolic_operations:
        return "*".join(phi)
    else:
        if syms.remove_1:
            out_smp = 1
        else:
            out_smp = 1.0
        for phi_val_smp in phi_smp:
            out_smp *= phi_val_smp
        out_smp = syms.convert_symb_to_int(out_smp)
        return "*".join(phi), out_smp


def phase_space_units(reagents):
    """Return dimension for phase space."""
    dim = 0.0
    for _, coefficient in reagents.items():
        dim += float(coefficient)
    return dim


def prefactor_units(units, exponent):
    """Return prefactor units."""
    return units**exponent / cc.ureg.second


def activation_energy_units():
    """Return activation energy units."""
    return cc.ureg.cal / cc.ureg.mole


def is_remove_forward(reaction_info, idx):
    """Return if the reaction is in the removed list."""
    return idx in reaction_info.remove_id_list


def fkc_conv_inv(self, mechanism, reaction, syms=None):
    """Return fkc_conv_inv."""
    record_symbolic_operations = True
    if syms is None:
        record_symbolic_operations = False
    dim = 0
    if record_symbolic_operations:
        dim_smp = 0
    for _, coefficient in reaction.reactants.items():
        dim -= coefficient
        if record_symbolic_operations:
            dim_smp -= coefficient
    # flip the signs
    for _, coefficient in reaction.products.items():
        dim += coefficient
        if record_symbolic_operations:
            dim_smp += coefficient

    conversion_smp = 1.0
    if record_symbolic_operations and syms.remove_1:
        conversion_smp = 1
    if dim == 0:
        conversion = ""
    elif dim > 0:
        if dim == 1.0:
            conversion = "*".join(["refCinv"])
            if record_symbolic_operations:
                conversion_smp *= syms.refCinv_smp
        else:
            if dim.is_integer():
                mult_str = "*".join(["refCinv"] * int(dim))
                conversion = "*".join([f"({mult_str})"])
                if record_symbolic_operations:
                    conversion_smp *= syms.refCinv_smp * syms.refCinv_smp
            else:
                conversion = "*".join(
                    [f"pow(std::max(refCinv, {sc_cutoff(dim)}), {dim:f})"]
                )
                if record_symbolic_operations:
                    conversion_smp *= syms.refCinv_smp**dim
    else:
        if dim == -1.0:
            conversion = "*".join(["refC"])
            if record_symbolic_operations:
                conversion_smp *= syms.refC_smp
        else:
            if dim.is_integer():
                conversion = "*".join(["refC"] * int(dim))
            else:
                conversion = "*".join(
                    [f"pow(std::max(refC, {sc_cutoff(abs(dim))}), {abs(dim):f})"]
                )
            if record_symbolic_operations:
                conversion_smp *= syms.refC_smp**dim

    if record_symbolic_operations:
        conversion_smp = syms.convert_symb_to_int(conversion_smp)
        return conversion, conversion_smp
    else:
        return conversion


def kc_conv(mechanism, reaction):
    """Return kc_conv."""
    dim = 0
    for _, coefficient in reaction.reactants.items():
        dim -= coefficient
    # flip the signs
    for _, coefficient in reaction.products.items():
        dim += coefficient

    if dim == 0:
        conversion = ""
    elif dim > 0:
        if dim == 1.0:
            conversion = "*".join(["refC"])
        else:
            if dim == 2.0:
                conversion = "*".join(["(refC * refC)"])
            else:
                if dim.is_integer():
                    conversion = "*".join(["refC"] * int(dim))
                else:
                    conversion = "*".join(
                        [f"pow(std::max(refC, {sc_cutoff(dim)}),{dim:f})"]
                    )
    else:
        if dim == -1.0:
            conversion = "*".join(["refCinv"])
        else:
            if dim.is_integer():
                conversion = "*".join(["refCinv"] * int(dim))
            else:
                conversion = "*".join(
                    [f"pow(std::max(refCinv, {sc_cutoff(abs(dim))}),{abs(dim):f})"]
                )

    return conversion


def sorted_kc(mechanism, species_info, reaction):
    """Return sorted kc."""
    conv = kc_conv(mechanism, reaction)
    exparg = sorted_kc_exp_arg(mechanism, species_info, reaction)
    if conv:
        return conv + " * exp(" + exparg + ")"
    else:
        return "exp(" + exparg + ")"


def sorted_kc_exp_arg(mechanism, species_info, reaction, syms=None):
    """Return sorted kc exponent argument."""
    record_symbolic_operations = True
    if syms is None:
        record_symbolic_operations = False
    terms = []
    if record_symbolic_operations:
        terms_smp = []
    for _ in range(species_info.n_species):
        terms.append("")
        if record_symbolic_operations:
            terms_smp.append(0.0)
    terms_qss = []
    if record_symbolic_operations:
        terms_qss_smp = []
    for _ in range(species_info.n_qssa_species):
        terms_qss.append("")
        if record_symbolic_operations:
            terms_qss_smp.append(0.0)

    for symbol, coefficient in reaction.reactants.items():
        if coefficient == 1.0:
            factor = " + "
            if record_symbolic_operations:
                factor_smp = 1.0
        else:
            factor = f" + {coefficient:f}*"
            if record_symbolic_operations:
                factor_smp = coefficient

        if symbol in species_info.qssa_species_list:
            i = species_info.ordered_idx_map[symbol] - species_info.n_species
            terms_qss[i] += f"{factor}g_RT_qss[{i}]"
            if record_symbolic_operations:
                factor_smp = syms.convert_symb_to_int(factor_smp)
                terms_qss_smp[i] += factor_smp * syms.g_RT_qss_smp[i]
        else:
            i = species_info.ordered_idx_map[symbol]
            terms[i] += f"{factor}g_RT[{i}]"
            if record_symbolic_operations:
                factor_smp = syms.convert_symb_to_int(factor_smp)
                terms_smp[i] += factor_smp * syms.g_RT_smp[i]

    for symbol, coefficient in reaction.products.items():
        if coefficient == 1.0:
            factor = " - "  # flip the signs
            if record_symbolic_operations:
                factor_smp = -1.0
        else:
            factor = f" - {coefficient:f}*"
            if record_symbolic_operations:
                factor_smp = -coefficient

        if symbol in species_info.qssa_species_list:
            i = species_info.ordered_idx_map[symbol] - species_info.n_species
            terms_qss[i] += f"{factor}g_RT_qss[{i}]"
            if record_symbolic_operations:
                factor_smp = syms.convert_symb_to_int(factor_smp)
                terms_qss_smp[i] += factor_smp * syms.g_RT_qss_smp[i]
        else:
            i = species_info.ordered_idx_map[symbol]
            terms[i] += f"{factor}g_RT[{i}]"
            if record_symbolic_operations:
                factor_smp = syms.convert_symb_to_int(factor_smp)
                terms_smp[i] += factor_smp * syms.g_RT_smp[i]

    dg = ""
    if record_symbolic_operations:
        dg_smp = 0
    for i in range(species_info.n_species):
        if terms[i]:
            dg += terms[i]
            if record_symbolic_operations:
                dg_smp += terms_smp[i]
    for i in range(species_info.n_qssa_species):
        if terms_qss[i]:
            dg += terms_qss[i]
            if record_symbolic_operations:
                dg_smp += terms_qss_smp[i]

    if dg[0:3] == " + ":
        # print("p dg = ", dg)
        # print("p dg_smp = ", dg_smp)
        if record_symbolic_operations:
            dg_smp = syms.convert_symb_to_int(dg_smp)
            return dg[3:], dg_smp
        else:
            return dg[3:]
    else:
        # print("m dg = ", dg)
        # print("m dg_smp = ", dg_smp)
        if record_symbolic_operations:
            dg_smp = syms.convert_symb_to_int(dg_smp)
            return "-" + dg[3:], dg_smp
        else:
            return "-" + dg[3:]


def enhancement_d(mechanism, species_info, reaction, syms=None):
    """Write get enhancement."""
    record_symbolic_operations = True
    if syms is None:
        record_symbolic_operations = False

    third_body = reaction.third_body is not None
    falloff = reaction.rate.type == "falloff"
    if not third_body and not falloff:
        raise ValueError("enhancement_d called for a reaction without a third body")

    if not reaction.third_body:
        raise NotImplementedError("FIXME EFFICIENCIES")
        species, coefficient = third_body
        if species == "<mixture>":
            if record_symbolic_operations:
                return "mixture", syms.mixture_smp
            else:
                return "mixture"
        if record_symbolic_operations:
            return (
                f"sc[{species_info.ordered_idx_map[species]}]",
                syms.sc_smp[species_info.ordered_idx_map[species]],
            )
        else:
            return f"sc[{species_info.ordered_idx_map[species]}]"

    efficiencies = reaction.third_body.efficiencies

    if (
        falloff
        and len(reaction.third_body.efficiencies) == 1
        and isclose(reaction.third_body.default_efficiency, 0.0)
    ):
        alpha = []
        if record_symbolic_operations:
            alpha_smp = []
        symbol = list(efficiencies.keys())[0]
        efficiency = efficiencies[symbol]
        if symbol not in species_info.qssa_species_list:
            factor = f"( {efficiency:.15g})"
            if record_symbolic_operations:
                factor_smp = efficiency
            if (efficiency) != 0:
                conc = f"sc[{species_info.ordered_idx_map[symbol]}]"
                if record_symbolic_operations:
                    conc_smp = syms.sc_smp[species_info.ordered_idx_map[symbol]]
                if (efficiency) == 1:
                    alpha.append(f"{conc}")
                    if record_symbolic_operations:
                        alpha_smp.append(conc_smp)
                else:
                    alpha.append(f"{factor}*{conc}")
                    if record_symbolic_operations:
                        factor_smp = syms.convert_symb_to_int(factor_smp)
                        alpha_smp.append(factor_smp * conc_smp)

        if record_symbolic_operations:
            enhancement_smp = 0.0 + alpha_smp[0]
            enhancement_smp = syms.convert_symb_to_int(enhancement_smp)
            return " + ".join(alpha).replace("+ -", "- "), enhancement_smp
        else:
            return " + ".join(alpha).replace("+ -", "- ")

    alpha = ["mixture"]
    if record_symbolic_operations:
        alpha_smp = [syms.mixture_smp]
    dict_species = {v: i for i, v in enumerate(species_info.all_species_list)}
    sorted_efficiencies = sorted(efficiencies.keys(), key=lambda v: dict_species[v])
    for symbol in sorted_efficiencies:
        efficiency = efficiencies[symbol]
        if symbol not in species_info.qssa_species_list:
            factor = f"( {efficiency:.15g} - 1)"
            if record_symbolic_operations:
                factor_smp = efficiency - 1
            if (efficiency - 1) != 0:
                conc = f"sc[{species_info.ordered_idx_map[symbol]}]"
                if record_symbolic_operations:
                    conc_smp = syms.sc_smp[species_info.ordered_idx_map[symbol]]
                if (efficiency - 1) == 1:
                    alpha.append(f"{conc}")
                    if record_symbolic_operations:
                        alpha_smp.append(conc_smp)
                else:
                    alpha.append(f"{factor}*{conc}")
                    if record_symbolic_operations:
                        factor_smp = syms.convert_symb_to_int(factor_smp)
                        alpha_smp.append(factor_smp * conc_smp)

    if record_symbolic_operations:
        enhancement_smp = 0.0
        for alpha_val in alpha_smp:
            enhancement_smp += alpha_val

    if record_symbolic_operations:
        enhancement_smp = syms.convert_symb_to_int(enhancement_smp)
        return " + ".join(alpha).replace("+ -", "- "), enhancement_smp
    else:
        return " + ".join(alpha).replace("+ -", "- ")


def evaluate_plog(rates, plog_pressure):
    """Evaluate rate constants for a PLOG reaction."""
    if plog_pressure <= rates[0][0]:
        # Case 1: plog_pressure <= lower bound -> take lower bound:
        plog_reaction = rates[0][1]
        pef = plog_reaction.pre_exponential_factor
        beta = plog_reaction.temperature_exponent
        ae = plog_reaction.activation_energy
        return pef, beta, ae
    elif plog_pressure >= rates[-1][0]:
        # Case 2: plog_pressure >= upper bound -> take upper bound:
        plog_reaction = rates[-1][1]
        pef = plog_reaction.pre_exponential_factor
        beta = plog_reaction.temperature_exponent
        ae = plog_reaction.activation_energy
        return pef, beta, ae
    else:
        # Case 3: lower bound < plog_pressure < upper bound -> logarithmic interpolation:
        for plog_p_i in range(len(rates) - 1):
            if rates[plog_p_i][0] <= plog_pressure < rates[plog_p_i + 1][0]:
                rate_low = rates[plog_p_i][1]
                rate_high = rates[plog_p_i + 1][1]
                interp_fac = (log(plog_pressure) - log(rates[plog_p_i][0])) / (
                    log(rates[plog_p_i + 1][0]) - log(rates[plog_p_i][0])
                )
                pef = exp(
                    log(rate_low.pre_exponential_factor) * (1 - interp_fac)
                    + log(rate_high.pre_exponential_factor) * interp_fac
                )
                beta = (
                    rate_low.temperature_exponent * (1 - interp_fac)
                    + rate_high.temperature_exponent * interp_fac
                )
                ae = (
                    rate_low.activation_energy * (1 - interp_fac)
                    + rate_high.activation_energy * interp_fac
                )
                return pef, beta, ae
