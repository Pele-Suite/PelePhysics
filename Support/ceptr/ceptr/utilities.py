"""Utility functions used across ceptr."""

import copy
from collections import Counter
from math import exp, isclose, log

import numpy as np

import ceptr.constants as cc
import ceptr.writer as cw

# Largest |exponent| allowed for an exp(-g/RT) factor or a partial product of
# such factors in the Kc product form (double overflows at about 709), and
# temperature range on which it is checked (see kc_product_plan). Below
# KC_T_LOW, which is under the validity range of the thermo polynomials, the
# exp of the summed exponent is always used.
KC_PRODUCT_LIMIT = 600.0
KC_T_LOW = 200.0
KC_T_HIGH = 10000.0


def intersection(lst1, lst2):
    """Return intersection of two lists."""
    return sorted(set(lst1).intersection(lst2))


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
    expterm = kc_exp(mechanism, species_info, reaction, 1)
    if conv:
        return conv + " * " + expterm
    else:
        return expterm


def kc_net_coefficients(reaction):
    """Return the net stoichiometric coefficient (reactants - products)."""
    net = {}
    for symbol, coefficient in reaction.reactants.items():
        net[symbol] = net.get(symbol, 0.0) + coefficient
    for symbol, coefficient in reaction.products.items():
        net[symbol] = net.get(symbol, 0.0) - coefficient
    return {s: nu for s, nu in net.items() if nu != 0.0}


def gibbs_rt(mechanism, symbol, temperatures):
    """Return g/(RT) of a species at the given temperatures.

    Uses the species thermo polynomials, extrapolated outside their range
    exactly as the generated gibbs() does.
    """
    import cantera as ct

    thermo = mechanism.species(symbol).thermo
    return np.array(
        [
            thermo.h(t) / (ct.gas_constant * t) - thermo.s(t) / ct.gas_constant
            for t in temperatures
        ]
    )


def kc_product_plan(mechanism, species_info):
    """Decide where Kc can be written as a product of exp(-g/RT) factors.

    A product of per-species factors equals exp of the summed exponent, but
    the factors and the partial products can overflow where the sum does
    not (|g/RT| grows like 1/T at low temperature, with either sign). The
    product form is therefore used only on a temperature interval
    [t_low, t_high] on which every factor, and every partial product in the
    chosen factor order, stays below exp(KC_PRODUCT_LIMIT). The interval and
    the orders are checked on a temperature grid here, when the code is
    generated. Outside the interval, and in single precision, the generated
    code falls back to the exp of the summed exponent.

    Returns (t_low, t_high, orders), where orders maps a reaction's net
    coefficients to its ordered factors [(symbol, +-1), ...] for sign = -1
    (+1: inv_ge, -1: ge), or to None if no safe order was found.
    """
    if getattr(species_info, "kc_plan", None) is not None:
        return species_info.kc_plan

    grid = np.concatenate(
        [np.arange(KC_T_LOW, 1000.0, 1.0), np.arange(1000.0, KC_T_HIGH + 1.0, 10.0)]
    )
    species = species_info.nonqssa_species_list + species_info.qssa_species_list
    g = {s: gibbs_rt(mechanism, s, grid) for s in species}

    # Contiguous temperature interval around 1000 K where all |g/RT| are small
    ok = np.max(np.abs(np.array(list(g.values()))), axis=0) <= KC_PRODUCT_LIMIT
    mid = int(np.searchsorted(grid, 1000.0))
    lo = mid
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = mid
    while hi < len(grid) - 1 and ok[hi + 1]:
        hi += 1
    if not ok[mid]:
        species_info.kc_plan = (None, None, {})
        return species_info.kc_plan
    window = slice(lo, hi + 1)

    orders = {}
    for reaction in mechanism.reactions():
        net = kc_net_coefficients(reaction)
        key = tuple(sorted(net.items()))
        if key in orders or not all(float(nu).is_integer() for nu in net.values()):
            continue
        # exp(-sum(nu * g)): nu > 0 -> ge (exponent -g), nu < 0 -> inv_ge (+g)
        factors = []
        for symbol in sorted(net, key=lambda s: species_info.ordered_idx_map[s]):
            nu = int(net[symbol])
            factors += [(symbol, -1 if nu > 0 else 1)] * abs(nu)
        partial = np.zeros(hi + 1 - lo)
        order, worst = [], 0.0
        while factors:
            # greedy: next factor keeping the partial exponent smallest
            best = min(
                range(len(factors)),
                key=lambda k: np.max(
                    np.abs(partial + factors[k][1] * g[factors[k][0]][window])
                ),
            )
            symbol, direction = factors.pop(best)
            partial = partial + direction * g[symbol][window]
            worst = max(worst, float(np.max(np.abs(partial))))
            order.append((symbol, direction))
        orders[key] = order if worst <= KC_PRODUCT_LIMIT else None

    species_info.kc_plan = (float(grid[lo]), float(grid[hi]), orders)
    return species_info.kc_plan


def kc_exp_parts(mechanism, species_info, reaction, sign):
    """Return (product form or None, exp form) of exp(sign * kc exponent).

    The exponent is sorted_kc_exp_arg: sum(nu * g_RT) over reactants minus
    the same over products. The product form uses ge[i] = exp(-g_RT[i]) and
    inv_ge[i] = 1 / ge[i] (ge_qss / inv_ge_qss for QSS species); it is None
    for reactions with non-integer net coefficients or without a safe factor
    order (see kc_product_plan).
    """
    exparg = sorted_kc_exp_arg(mechanism, species_info, reaction)
    exp_form = f"exp({exparg})" if sign > 0 else f"exp(-({exparg}))"
    _, _, orders = kc_product_plan(mechanism, species_info)
    order = orders.get(tuple(sorted(kc_net_coefficients(reaction).items())))
    if not order:
        return None, exp_form

    def ref(symbol, direction):
        array = "inv_ge" if direction * -sign > 0 else "ge"
        if symbol in species_info.qssa_species_list:
            i = species_info.ordered_idx_map[symbol] - species_info.n_species
            return f"{array}_qss[{i}]"
        return f"{array}[{species_info.ordered_idx_map[symbol]}]"

    return " * ".join(ref(s, d) for s, d in order), exp_form


def kc_exp(mechanism, species_info, reaction, sign):
    """Return exp(sign * kc exponent) as C++ code.

    Uses the product of shared exp(-g/RT) factors when kc_fast (see
    write_gibbs_exp) holds, and the exp of the summed exponent otherwise.
    """
    product, exp_form = kc_exp_parts(mechanism, species_info, reaction, sign)
    if product is None:
        return exp_form
    return f"(kc_fast ? ({product}) : {exp_form})"


def write_gibbs_exp(fstream, mechanism, species_info, g_rt="g_RT", g_rt_qss="g_RT_qss"):
    """Write kc_fast and the per-species exp(-g/RT) factors used by kc_exp."""
    t_low, t_high, _ = kc_product_plan(mechanism, species_info)
    if t_low is None:
        cw.writer(fstream, "constexpr bool kc_fast = false;")
    else:
        cw.writer(
            fstream,
            cw.comment(
                "Kc from shared exp(-g/RT) factors where no factor or partial"
                " product can overflow (checked by CEPTR on this T range);"
                " exp of the sum otherwise and in single precision"
            ),
        )
        cw.writer(
            fstream,
            "const bool kc_fast = sizeof(amrex::Real) == sizeof(double) &&"
            f" T >= {t_low:.1f} && T <= {t_high:.1f};",
        )
    for n, src, sfx in (
        (species_info.n_species, g_rt, ""),
        (species_info.n_qssa_species, g_rt_qss, "_qss"),
    ):
        if n == 0:
            continue
        cw.writer(fstream, f"amrex::Real ge{sfx}[{n}] = {{}}, inv_ge{sfx}[{n}] = {{}};")
        cw.writer(fstream, "if (kc_fast) {")
        cw.writer(fstream, f"for (int i = 0; i < {n}; ++i) {{")
        cw.writer(fstream, f"ge{sfx}[i] = exp(-{src}[i]);")
        cw.writer(fstream, f"inv_ge{sfx}[i] = 1.0 / ge{sfx}[i];")
        cw.writer(fstream, "}")
        cw.writer(fstream, "}")


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
