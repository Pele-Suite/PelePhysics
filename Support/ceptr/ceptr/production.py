"""Production functions."""

from math import isclose

import symengine as sme

import ceptr.constants as cc
import ceptr.utilities as cu
import ceptr.writer as cw


def production_rate(
    fstream,
    mechanism,
    species_info,
    reaction_info,
    syms=None,
):
    """Write production rate."""
    n_species = species_info.n_species
    n_qss_species = species_info.n_qssa_species
    n_reactions = mechanism.n_reactions

    assert len(reaction_info.index) == 7

    itroe = reaction_info.index[0:2]
    isri = reaction_info.index[1:3]
    ilindemann = reaction_info.index[2:4]
    # i3body = reaction_info.index[3:5]
    # isimple = reaction_info.index[4:6]
    # ispecial = reaction_info.index[5:7]

    ntroe = itroe[1] - itroe[0]
    nsri = isri[1] - isri[0]
    nlindemann = ilindemann[1] - ilindemann[0]
    # n3body = i3body[1] - i3body[0]
    # nsimple = isimple[1] - isimple[0]
    # nspecial = ispecial[1] - ispecial[0]

    # qdot
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment(
            "GPU version of productionRate: no more use of thermo namespace vectors "
        ),
    )
    if n_reactions > 0:
        if n_qss_species > 0:
            cw.writer(
                fstream,
                "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void"
                " comp_qfqr(amrex::Real *  qf, amrex::Real * qr, const"
                " amrex::Real * sc, const amrex::Real * sc_qss, const"
                " amrex::Real T, const amrex::Real invT, const amrex::Real logT)",
            )
        else:
            cw.writer(
                fstream,
                "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void"
                " comp_qfqr(amrex::Real *  qf, amrex::Real * qr, const"
                " amrex::Real * sc, const amrex::Real * /*sc_qss*/,const"
                " amrex::Real T, const amrex::Real invT, const amrex::Real logT)",
            )
    else:
        cw.writer(
            fstream,
            "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void comp_qfqr(amrex::Real *"
            " /*qf*/, amrex::Real * /*qr*/, const amrex::Real * /*sc*/, const"
            " amrex::Real * /*sc_qss*/, const amrex::Real /*T*/, const amrex::Real"
            " /*invT*/, const amrex::Real /*logT*/)",
        )
    cw.writer(fstream, "{")

    if n_reactions > 0:
        # nclassd = n_reactions - nspecial
        # nCorr   = n3body + ntroe + nsri + nlindemann

        # reacs are sorted here
        for orig_idx, idx in reaction_info.idxmap.items():
            cw.writer(fstream)
            reaction = mechanism.reaction(orig_idx)
            cw.writer(
                fstream,
                cw.comment(f"reaction {orig_idx}: {reaction.equation}"),
            )
            if bool(reaction.orders):
                string = cu.qss_sorted_phase_space(
                    mechanism, species_info, reaction, reaction.orders
                )
                cw.writer(
                    fstream,
                    f"qf[{idx}] = {string};",
                )
            else:
                string = cu.qss_sorted_phase_space(
                    mechanism,
                    species_info,
                    reaction,
                    reaction.reactants,
                )
                cw.writer(
                    fstream,
                    f"qf[{idx}] = {string};",
                )

            if reaction.reversible:
                string = cu.qss_sorted_phase_space(
                    mechanism,
                    species_info,
                    reaction,
                    reaction.products,
                )
                cw.writer(
                    fstream,
                    f"qr[{idx}] = {string};",
                )
            else:
                cw.writer(fstream, f"qr[{idx}] = 0.0;")

        cw.writer(fstream)

        # Mixt concentration for PD & TB
        cw.writer(fstream, cw.comment("compute the mixture concentration"))
        cw.writer(fstream, "amrex::Real mixture = 0.0;")
        cw.writer(fstream, f"for (int i = 0; i < {n_species}; ++i) {{")
        cw.writer(fstream, "mixture += sc[i];")
        cw.writer(fstream, "}")
        cw.writer(fstream)
        if species_info.n_qssa_species > 0:
            cw.writer(
                fstream,
                f"for (int i = 0; i < {species_info.n_qssa_species}; ++i)",
            )
            cw.writer(fstream, "{")
            cw.writer(fstream, "mixture += sc_qss[i];")
            cw.writer(fstream, "}")
            cw.writer(fstream)

        # Kc stuff
        cw.writer(fstream, cw.comment("compute the Gibbs free energy"))
        cw.writer(fstream, f"amrex::Real g_RT[{species_info.n_species}];")
        cw.writer(fstream, "gibbs(g_RT, T);")
        if species_info.n_qssa_species > 0:
            cw.writer(
                fstream,
                f"amrex::Real g_RT_qss[{species_info.n_qssa_species}];",
            )
            cw.writer(fstream, "gibbs_qss(g_RT_qss, T);")

        cw.writer(fstream)

        cw.writer(
            fstream,
            cw.comment("reference concentration: P_atm / (RT) in inverse mol/m^3"),
        )
        cw.writer(
            fstream,
            f"amrex::Real refC = {cc.Patm_pa:g} /"
            f" {cc.R.to(cc.ureg.joule / (cc.ureg.mole / cc.ureg.kelvin)).m:g} *"
            " invT;",
        )
        cw.writer(fstream, "amrex::Real refCinv = 1 / refC;")

        cw.writer(fstream)

        # kfs
        cw.writer(fstream, cw.comment("Evaluate the kfs"))
        cw.writer(fstream, "amrex::Real k_f, Corr;")
        if ntroe > 0:
            cw.writer(
                fstream,
                "amrex::Real redP, F, logPred, logFcent, troe_c, troe_n, troe, F_troe;",
            )
        if nsri > 0:
            cw.writer(fstream, "amrex::Real redP, F, X, F_sri;")
        cw.writer(fstream)

        # Loop like you're going through them in the mech.Linp order
        for orig_idx, idx in sorted(reaction_info.idxmap.items()):
            reaction = mechanism.reaction(orig_idx)

            kc_exp_arg = cu.sorted_kc_exp_arg(mechanism, species_info, reaction)
            kc_conv_inv = cu.fkc_conv_inv(mechanism, species_info, reaction)

            if bool(reaction.orders):
                dim = cu.phase_space_units(reaction.orders)
            else:
                dim = cu.phase_space_units(reaction.reactants)
            third_body = reaction.third_body is not None
            if reaction.reaction_type == "pressure-dependent-Arrhenius":
                plog = True
            elif reaction.reaction_type == "three-body-pressure-dependent-Arrhenius":
                if not third_body:
                    raise ValueError(
                        f"Inconsistent reaction type in reaction {orig_idx}."
                    )
                plog = True
            else:
                plog = False
            falloff = reaction.rate.type == "falloff"
            is_troe = reaction.rate.sub_type == "Troe"
            is_sri = reaction.rate.sub_type == "Sri"
            is_lindemann = reaction.rate.sub_type == "Lindemann"
            aeuc = cu.activation_energy_units()
            if not third_body and not falloff and not plog:
                # Case 3 !PD, !TB, !PLOG
                ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
                pef = (reaction.rate.pre_exponential_factor * ctuc).to_base_units()
                beta = reaction.rate.temperature_exponent
                ae = (
                    reaction.rate.activation_energy * cc.ureg.joule / cc.ureg.kmol
                ).to(aeuc)
            elif not falloff and not plog:
                # Case 2 !PD, TB, !PLOG
                ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), -dim)
                pef = (reaction.rate.pre_exponential_factor * ctuc).to_base_units()
                beta = reaction.rate.temperature_exponent
                ae = (
                    reaction.rate.activation_energy * cc.ureg.joule / cc.ureg.kmol
                ).to(aeuc)
            elif not third_body and not falloff and plog:
                # Case 4 PLOG
                ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
                plog_pef, plog_beta, plog_ae = cu.evaluate_plog(
                    reaction.rate.rates, mechanism.P
                )
                pef = (plog_pef * ctuc).to_base_units()
                beta = plog_beta
                ae = (plog_ae * cc.ureg.joule / cc.ureg.kmol).to(aeuc)
            elif third_body and not falloff and plog:
                # Case 5 TB, PLOG
                ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), -dim)
                plog_pef, plog_beta, plog_ae = cu.evaluate_plog(
                    reaction.rate.rates, mechanism.P
                )
                pef = (plog_pef * ctuc).to_base_units()
                beta = plog_beta
                ae = (plog_ae * cc.ureg.joule / cc.ureg.kmol).to(aeuc)
            else:
                # Case 1 PD, TB
                ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
                pef = (
                    reaction.rate.high_rate.pre_exponential_factor * ctuc
                ).to_base_units()
                beta = reaction.rate.high_rate.temperature_exponent
                ae = (
                    reaction.rate.high_rate.activation_energy
                    * cc.ureg.joule
                    / cc.ureg.kmol
                ).to(aeuc)

                low_pef = (
                    reaction.rate.low_rate.pre_exponential_factor * ctuc
                ).to_base_units()
                low_beta = reaction.rate.low_rate.temperature_exponent
                low_ae = (
                    reaction.rate.low_rate.activation_energy
                    * cc.ureg.joule
                    / cc.ureg.kmol
                ).to(aeuc)
                if is_troe:
                    troe = reaction.rate.falloff_coeffs
                    ntroe = len(troe)
                elif is_sri:
                    sri = reaction.rate.falloff_coeffs
                    nsri = len(sri)
                elif is_lindemann:
                    pass
                else:
                    raise ValueError(
                        f"Unrecognized reaction rate type {reaction.rate.type},"
                        f" {reaction.rate.sub_type} for reaction: {reaction.equation}"
                    )

            cw.writer(
                fstream,
                cw.comment(f"reaction {orig_idx}:  {reaction.equation}"),
            )
            cw.writer(fstream, f"k_f = {pef.m:.15g}")
            if (beta == 0) and (ae == 0):
                cw.writer(fstream, "           ;")
            else:
                if ae == 0:
                    cw.writer(fstream, f"           * exp(({beta:.15g}) * logT);")
                elif beta == 0:
                    cw.writer(
                        fstream,
                        "           *"
                        f" exp(-({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g})"
                        " * invT);",
                    )
                else:
                    cw.writer(
                        fstream,
                        f"           * exp(({beta:.15g}) * logT -"
                        f" ({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g})"
                        " * invT);",
                    )

            alpha = None
            if not third_body and not falloff:
                cw.writer(fstream, f"qf[{idx}] *= k_f;")
            elif (
                not falloff
                and len(reaction.third_body.efficiencies) == 1
                and isclose(reaction.third_body.default_efficiency, 0.0)
            ):
                # TB of A + B + C = AB + C
                cw.writer(fstream, f"qf[{idx}] *= k_f;")
            elif not falloff:
                alpha = enhancement_d_with_qss(mechanism, species_info, reaction)
                cw.writer(fstream, f"Corr  = {alpha};")
                cw.writer(fstream, f"qf[{idx}] *= Corr * k_f;")
            else:
                # Falloff, TB
                alpha = enhancement_d_with_qss(mechanism, species_info, reaction)
                cw.writer(fstream, f"Corr  = {alpha};")
                if (
                    falloff
                    and len(reaction.third_body.efficiencies) == 1
                    and isclose(reaction.third_body.default_efficiency, 0.0)
                ):
                    # Avoid float point exception?
                    cw.writer(fstream, f"Corr = std::max(Corr, 1e-20);")
                cw.writer(
                    fstream,
                    "redP = Corr / k_f *"
                    f" {10 ** (-dim * 6) * low_pef.m * 10 ** 3 ** dim:.15g} ",
                )
                if (low_beta == 0) and (low_ae.m == 0):
                    cw.writer(
                        fstream,
                        "           ;",
                    )
                elif low_ae.m == 0:
                    cw.writer(
                        fstream,
                        f"           * exp({low_beta:.15g}  * logT);",
                    )
                elif low_beta == 0:
                    cw.writer(
                        fstream,
                        "           * exp(-"
                        f" ({(1.0 / cc.Rc / cc.ureg.kelvin * low_ae).m:.15g})"
                        " *invT);",
                    )
                else:
                    cw.writer(
                        fstream,
                        f"           * exp({low_beta:.15g}  * logT -"
                        f" ({(1.0 / cc.Rc / cc.ureg.kelvin * low_ae).m:.15g})"
                        " *invT);",
                    )
                if is_troe:
                    cw.writer(fstream, "F = redP / (1.0 + redP);")
                    cw.writer(fstream, "logPred = log10(redP);")
                    cw.writer(fstream, "logFcent = log10(")
                    if abs(troe[1]) > 1.0e-100:
                        if 1.0 - troe[0] != 0:
                            cw.writer(
                                fstream,
                                f"    ({1.0 - troe[0]:.15g})*exp(-T *"
                                f" {1 / troe[1]:.15g})",
                            )
                    else:
                        cw.writer(fstream, "     0.0 ")
                    if abs(troe[2]) > 1.0e-100:
                        if troe[0] == 1:
                            cw.writer(
                                fstream,
                                f"    + exp(-T * {1 / troe[2]:.15g})",
                            )
                        else:
                            cw.writer(
                                fstream,
                                f"    + {troe[0]:.15g} * exp(-T * {1 / troe[2]:.15g})",
                            )
                    else:
                        cw.writer(fstream, "    + 0.0 ")
                    if ntroe == 4:
                        if troe[3] < 0:
                            cw.writer(fstream, f"    + exp({-troe[3]:.15g} * invT));")
                        else:
                            cw.writer(fstream, f"    + exp(-{troe[3]:.15g} * invT));")
                    else:
                        cw.writer(fstream, "    + 0.0);")
                    cw.writer(fstream, "troe_c = -0.4 - 0.67 * logFcent;")
                    cw.writer(fstream, "troe_n = 0.75 - 1.27 * logFcent;")
                    cw.writer(
                        fstream,
                        "troe = (troe_c + logPred) / (troe_n -"
                        " 0.14*(troe_c + logPred));",
                    )
                    cw.writer(
                        fstream,
                        "F_troe = exp(M_LN10 * logFcent / (1.0 + troe*troe));",
                    )
                    cw.writer(fstream, "Corr = F * F_troe;")
                    cw.writer(fstream, f"qf[{idx}] *= Corr * k_f;")
                elif is_sri:
                    cw.writer(fstream, "F = redP / (1.0 + redP);")
                    cw.writer(fstream, "logPred = log10(redP);")
                    cw.writer(fstream, "X = 1.0 / (1.0 + logPred*logPred);")
                    if sri[1] < 0:
                        cw.writer(
                            fstream,
                            f"F_sri = exp(X * log({sri[0]:.15g} *"
                            f" exp({-sri[1]:.15g}*invT)",
                        )
                    else:
                        cw.writer(
                            fstream,
                            f"F_sri = exp(X * log({sri[0]:.15g} *"
                            f" exp(-{sri[1]:.15g}*invT)",
                        )
                    if sri[2] > 1.0e-100:
                        cw.writer(fstream, f"   +  exp(logT/{sri[2]:.15g}) ")
                    else:
                        cw.writer(fstream, "   +  0. ")
                    cw.writer(
                        fstream,
                        f"   *  ({nsri} > 3 ?"
                        f" {sri[3]:.15g}*exp({sri[4]:.15g}*logT : 1.0);",
                    )
                    cw.writer(fstream, "Corr = F * F_sri;")
                    cw.writer(fstream, f"qf[{idx}] *= Corr * k_f;")
                elif nlindemann > 0:
                    cw.writer(fstream, "Corr = redP / (1. + redP);")
                    cw.writer(fstream, f"qf[{idx}] *= Corr * k_f;")

            if kc_conv_inv:
                if alpha is None:
                    cw.writer(
                        fstream,
                        f"qr[{idx}] *= k_f * exp(-({kc_exp_arg})) * ({kc_conv_inv});",
                    )
                else:
                    cw.writer(
                        fstream,
                        f"qr[{idx}] *= Corr * k_f * exp(-({kc_exp_arg})) *"
                        f" ({kc_conv_inv});",
                    )
            else:
                if alpha is None:
                    cw.writer(
                        fstream,
                        f"qr[{idx}] *= k_f * exp(-({kc_exp_arg}));",
                    )
                else:
                    cw.writer(
                        fstream,
                        f"qr[{idx}] *= Corr * k_f * exp(-({kc_exp_arg}));",
                    )

        cw.writer(fstream)

    cw.writer(fstream)
    cw.writer(fstream, "}")

    cw.writer(fstream)

    # main function
    if n_reactions > 0:
        cw.writer(
            fstream,
            "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void "
            " productionRate(amrex::Real * wdot, const amrex::Real * sc,"
            " const amrex::Real T)",
        )
    else:
        cw.writer(
            fstream,
            "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void "
            " productionRate(amrex::Real * wdot, const amrex::Real *"
            " /*sc*/, const amrex::Real /*T*/)",
        )
    cw.writer(fstream, "{")

    if n_reactions == 0:
        cw.writer(fstream)
    else:
        cw.writer(fstream, "const amrex::Real invT = 1.0 / T;")
        cw.writer(fstream, "const amrex::Real logT = log(T);")
        cw.writer(fstream)
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
        cw.writer(fstream, "const amrex::Real refCinv = 1 / refC;")

        if nsri > 0:
            cw.writer(fstream, "amrex::Real X, F_sri;")

    cw.writer(fstream)
    cw.writer(fstream, f"for (int i = 0; i < {n_species}; ++i) {{")
    cw.writer(fstream, "wdot[i] = 0.0;")
    cw.writer(fstream, "}")
    cw.writer(fstream)

    # initialize the symbolic wdot array
    for i in range(n_species):
        syms.wdot_smp[i] = 0.0

    # initialize the symbolic jacobian array
    for i in range(n_species * (n_species + 1)):
        syms.jac_smp[i] = 0.0

    if n_reactions > 0:
        # nclassd = n_reactions - nspecial
        # nCorr   = n3body + ntroe + nsri + nlindemann

        # Mixt concentration for PD & TB
        cw.writer(fstream, cw.comment("compute the mixture concentration"))
        cw.writer(fstream, "amrex::Real mixture = 0.0;")
        cw.writer(fstream, f"for (int i = 0; i < {n_species}; ++i) {{")
        cw.writer(fstream, "mixture += sc[i];")
        cw.writer(fstream, "}")
        cw.writer(fstream)

        # Kc stuff
        cw.writer(fstream, cw.comment("compute the Gibbs free energy"))
        cw.writer(fstream, f"amrex::Real g_RT[{species_info.n_species}];")
        cw.writer(fstream, "gibbs(g_RT, T);")
        if species_info.n_qssa_species > 0:
            cw.writer(
                fstream,
                f"amrex::Real g_RT_qss[{species_info.n_qssa_species}];",
            )
            cw.writer(fstream, "gibbs_qss(g_RT_qss, T);")
        cw.writer(fstream)

        if species_info.n_qssa_species > 0:
            cw.writer(
                fstream,
                f"amrex::Real sc_qss[{max(1, species_info.n_qssa_species)}];",
            )
            cw.writer(
                fstream,
                f"amrex::Real kf_qss[{reaction_info.n_qssa_reactions}],"
                f" qf_qss[{reaction_info.n_qssa_reactions}],"
                f" qr_qss[{reaction_info.n_qssa_reactions}];",
            )
            cw.writer(fstream, cw.comment("Fill sc_qss here"))
            cw.writer(fstream, "comp_k_f_qss(T, invT, logT, kf_qss);")
            # cw.writer(fstream,"comp_Kc_qss(invT, g_RT, g_RT_qss, Kc_qss);")
            cw.writer(
                fstream,
                "comp_qss_coeff(kf_qss, qf_qss, qr_qss, sc, T, g_RT, g_RT_qss);",
            )
            cw.writer(fstream, "comp_sc_qss(sc_qss, qf_qss, qr_qss);")
            cw.writer(fstream)

        # Loop like you're going through them in the mech.Linp order
        for orig_idx, _ in reaction_info.idxmap.items():
            reaction = mechanism.reaction(orig_idx)
            cw.writer(fstream, "{")
            if bool(reaction.orders):
                forward_sc, forward_sc_smp = cu.qss_sorted_phase_space(
                    mechanism, species_info, reaction, reaction.orders, syms
                )
            else:
                forward_sc, forward_sc_smp = cu.qss_sorted_phase_space(
                    mechanism, species_info, reaction, reaction.reactants, syms
                )
            if reaction.reversible:
                reverse_sc, reverse_sc_smp = cu.qss_sorted_phase_space(
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

            if bool(reaction.orders):
                dim = cu.phase_space_units(reaction.orders)
            else:
                dim = cu.phase_space_units(reaction.reactants)

            third_body = reaction.third_body is not None
            if reaction.reaction_type == "pressure-dependent-Arrhenius":
                plog = True
            elif reaction.reaction_type == "three-body-pressure-dependent-Arrhenius":
                if not third_body:
                    raise ValueError(
                        f"Inconsistent reaction type in reaction {orig_idx}."
                    )
                plog = True
            else:
                plog = False
            falloff = reaction.rate.type == "falloff"
            is_troe = reaction.rate.sub_type == "Troe"
            is_sri = reaction.rate.sub_type == "Sri"
            is_lindemann = reaction.rate.sub_type == "Lindemann"
            aeuc = cu.activation_energy_units()
            if not third_body and not falloff and not plog:
                # Case 3 !PD, !TB
                ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
                pef = (reaction.rate.pre_exponential_factor * ctuc).to_base_units()
                beta = reaction.rate.temperature_exponent
                ae = (
                    reaction.rate.activation_energy * cc.ureg.joule / cc.ureg.kmol
                ).to(aeuc)
            elif not falloff and not plog:
                # Case 2 !PD, TB, !PLOG
                ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), -dim)
                pef = (reaction.rate.pre_exponential_factor * ctuc).to_base_units()
                beta = reaction.rate.temperature_exponent
                ae = (
                    reaction.rate.activation_energy * cc.ureg.joule / cc.ureg.kmol
                ).to(aeuc)
            elif not third_body and not falloff and plog:
                # Case 4 PLOG
                ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
                plog_pef, plog_beta, plog_ae = cu.evaluate_plog(
                    reaction.rate.rates, mechanism.P
                )
                pef = (plog_pef * ctuc).to_base_units()
                beta = plog_beta
                ae = (plog_ae * cc.ureg.joule / cc.ureg.kmol).to(aeuc)
            elif third_body and not falloff and plog:
                # Case 5 TB, PLOG
                ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), -dim)
                plog_pef, plog_beta, plog_ae = cu.evaluate_plog(
                    reaction.rate.rates, mechanism.P
                )
                pef = (plog_pef * ctuc).to_base_units()
                beta = plog_beta
                ae = (plog_ae * cc.ureg.joule / cc.ureg.kmol).to(aeuc)
            else:
                # Case 1 PD, TB, !PLOG
                ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
                pef = (
                    reaction.rate.high_rate.pre_exponential_factor * ctuc
                ).to_base_units()
                beta = reaction.rate.high_rate.temperature_exponent
                ae = (
                    reaction.rate.high_rate.activation_energy
                    * cc.ureg.joule
                    / cc.ureg.kmol
                ).to(aeuc)

                low_pef = (
                    reaction.rate.low_rate.pre_exponential_factor * ctuc
                ).to_base_units()
                low_beta = reaction.rate.low_rate.temperature_exponent
                low_ae = (
                    reaction.rate.low_rate.activation_energy
                    * cc.ureg.joule
                    / cc.ureg.kmol
                ).to(aeuc)
                if is_troe:
                    troe = reaction.rate.falloff_coeffs
                    ntroe = len(troe)
                elif is_sri:
                    sri = reaction.rate.falloff_coeffs
                    nsri = len(sri)
                elif is_lindemann:
                    pass
                else:
                    raise ValueError(
                        f"Unrecognized reaction rate type {reaction.rate.type},"
                        f" {reaction.rate.sub_type} for reaction: {reaction.equation}"
                    )

                low_beta = syms.convert_number_to_int(low_beta)

            beta = syms.convert_number_to_int(beta)

            cw.writer(
                fstream,
                cw.comment(f"reaction {orig_idx}:  {reaction.equation}"),
            )
            cw.writer(fstream, f"const amrex::Real k_f = {pef.m:.15g}")

            k_f_smp = pef.m

            if (beta == 0) and (ae == 0):
                cw.writer(fstream, "           ;")
            else:
                if ae == 0:
                    cw.writer(fstream, f"           * exp(({beta:.15g}) * logT);")
                    k_f_smp *= sme.exp(beta * syms.logT_smp)
                elif beta == 0:
                    cw.writer(
                        fstream,
                        "           *"
                        f" exp(-({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g})"
                        " * invT);",
                    )
                    coeff = (((1.0 / cc.Rc / cc.ureg.kelvin)) * ae).magnitude
                    k_f_smp *= sme.exp(-coeff * syms.invT_smp)
                else:
                    cw.writer(
                        fstream,
                        f"           * exp(({beta:.15g}) * logT -"
                        f" ({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g})"
                        " * invT);",
                    )
                    coeff = ((1.0 / cc.Rc / cc.ureg.kelvin)) * ae
                    k_f_smp *= sme.exp(beta * syms.logT_smp - coeff * syms.invT_smp)

            alpha = None
            if not third_body and not falloff:
                cw.writer(
                    fstream,
                    f"const amrex::Real qf = k_f * ({forward_sc});",
                )
                qf_smp = k_f_smp * forward_sc_smp
            elif (
                not falloff
                and len(reaction.third_body.efficiencies) == 1
                and isclose(reaction.third_body.default_efficiency, 0.0)
            ):
                cw.writer(
                    fstream,
                    f"const amrex::Real qf = k_f * ({forward_sc});",
                )
                qf_smp = k_f_smp * forward_sc_smp
            elif not falloff:
                alpha, alpha_smp = enhancement_d_with_qss(
                    mechanism, species_info, reaction, syms
                )
                cw.writer(fstream, f"const amrex::Real Corr = {alpha};")
                corr_smp = alpha_smp
                cw.writer(
                    fstream,
                    f"const amrex::Real qf = Corr * k_f * ({forward_sc});",
                )
                qf_smp = corr_smp * k_f_smp * forward_sc_smp
            else:
                alpha, alpha_smp = enhancement_d_with_qss(
                    mechanism, species_info, reaction, syms
                )
                cw.writer(fstream, f"amrex::Real Corr = {alpha};")
                corr_smp = alpha_smp
                if (
                    falloff
                    and len(reaction.third_body.efficiencies) == 1
                    and isclose(reaction.third_body.default_efficiency, 0.0)
                ):
                    # Avoid float point exception?
                    cw.writer(fstream, f"Corr = std::max(Corr, 1e-20);")
                cw.writer(
                    fstream,
                    "const amrex::Real redP = Corr / k_f *"
                    f" {10 ** (-dim * 6) * low_pef.m * 10 ** 3 ** dim:.15g} ",
                )
                redp_smp = (
                    corr_smp / k_f_smp * (10 ** (-dim * 6) * low_pef.m * 10 ** (3**dim))
                )
                if (low_beta == 0) and (low_ae.m == 0):
                    cw.writer(
                        fstream,
                        "           ;",
                    )
                elif low_ae.m == 0:
                    cw.writer(
                        fstream,
                        f"           * exp({low_beta:.15g} * logT);",
                    )
                elif low_beta == 0:
                    cw.writer(
                        fstream,
                        "           * exp(-"
                        f" {(1.0 / cc.Rc / cc.ureg.kelvin * low_ae).m:.15g} *"
                        " invT);",
                    )
                else:
                    cw.writer(
                        fstream,
                        f"           * exp({low_beta:.15g} * logT -"
                        f" {(1.0 / cc.Rc / cc.ureg.kelvin * low_ae).m:.15g} *"
                        " invT);",
                    )
                coeff = (1.0 / cc.Rc / cc.ureg.kelvin * low_ae).magnitude
                redp_smp *= sme.exp(low_beta * syms.logT_smp - coeff * syms.invT_smp)
                if is_troe:
                    cw.writer(fstream, "const amrex::Real F = redP / (1.0 + redP);")
                    f_smp = redp_smp / (1.0 + redp_smp)
                    cw.writer(fstream, "const amrex::Real logPred = log10(redP);")
                    logpred_smp = sme.log(redp_smp, 10)
                    cw.writer(fstream, "const amrex::Real logFcent = log10(")
                    int_smp = 0.0
                    if abs(troe[1]) > 1.0e-100:
                        if 1.0 - troe[0] != 0:
                            cw.writer(
                                fstream,
                                f"    {1.0 - troe[0]:.15g} * exp(-T *"
                                f" {1 / troe[1]:.15g})",
                            )
                            first_factor = syms.convert_number_to_int(1.0 - troe[0])
                            second_factor = syms.convert_number_to_int(-1 / troe[1])
                            int_smp += first_factor * sme.exp(
                                syms.T_smp * second_factor
                            )
                    else:
                        cw.writer(fstream, "     0.0 ")
                    if abs(troe[2]) > 1.0e-100:
                        if troe[0] != 0:
                            cw.writer(
                                fstream,
                                f"    + {troe[0]:.15g} * exp(-T * {1 / troe[2]:.15g})",
                            )
                            first_factor = syms.convert_number_to_int(troe[0])
                            second_factor = syms.convert_number_to_int(-1 / troe[2])
                            int_smp += first_factor * sme.exp(
                                syms.T_smp * second_factor
                            )
                    else:
                        cw.writer(fstream, "    + 0.0 ")
                    if ntroe == 4:
                        if troe[3] < 0:
                            cw.writer(fstream, f"    + exp({-troe[3]:.15g} * invT));")
                            first_factor = syms.convert_number_to_int(-troe[3])
                            int_smp += sme.exp(first_factor * syms.invT_smp)
                        else:
                            cw.writer(fstream, f"    + exp(-{troe[3]:.15g} * invT));")
                            first_factor = syms.convert_number_to_int(-troe[3])
                            int_smp += sme.exp(first_factor * syms.invT_smp)
                    else:
                        cw.writer(fstream, "    + 0.0);")
                    logfcent_smp = sme.log(int_smp, 10)
                    cw.writer(
                        fstream,
                        "const amrex::Real troe_c = -0.4 - 0.67 * logFcent;",
                    )
                    troe_c_smp = -0.4 - 0.67 * logfcent_smp
                    cw.writer(
                        fstream,
                        "const amrex::Real troe_n = 0.75 - 1.27 * logFcent;",
                    )
                    troe_n_smp = 0.75 - 1.27 * logfcent_smp
                    cw.writer(
                        fstream,
                        "const amrex::Real troe = (troe_c + logPred) /"
                        " (troe_n - 0.14 * (troe_c + logPred));",
                    )
                    troe_smp = (troe_c_smp + logpred_smp) / (
                        troe_n_smp - 0.14 * (troe_c_smp + logpred_smp)
                    )
                    cw.writer(
                        fstream,
                        "const amrex::Real F_troe = exp(M_LN10 *logFcent /"
                        " (1.0 + troe * troe));",
                    )
                    f_troe_smp = pow(10, logfcent_smp / (1.0 + troe_smp * troe_smp))
                    cw.writer(fstream, "Corr = F * F_troe;")
                    corr_smp = f_smp * f_troe_smp
                    cw.writer(
                        fstream,
                        f"const amrex::Real qf = Corr * k_f * ({forward_sc});",
                    )
                    qf_smp = corr_smp * k_f_smp * forward_sc_smp
                elif is_sri:
                    cw.writer(fstream, "const amrex::Real F = redP / (1.0 + redP);")
                    f_smp = redp_smp / (1.0 + redp_smp)
                    cw.writer(fstream, "const amrex::Real logPred = log10(redP);")
                    logpred_smp = sme.log(redp_smp, 10)
                    cw.writer(fstream, "X = 1.0 / (1.0 + logPred*logPred);")
                    # x_smp = 1.0 / (1.0 + logpred_smp * logpred_smp)
                    if sri[1] < 0:
                        cw.writer(
                            fstream,
                            f"F_sri = exp(X * log({sri[0]:.15g} *"
                            f" exp({-sri[1]:.15g} * invT)",
                        )
                        if syms is not None:
                            raise NotImplementedError("Not done for now")
                    else:
                        cw.writer(
                            fstream,
                            f"F_sri = exp(X * log({sri[0]:.15g} *"
                            f" exp(-{sri[1]:.15g} * invT)",
                        )
                        if syms is not None:
                            raise NotImplementedError("Not done for now")
                    if sri[2] > 1.0e-100:
                        cw.writer(fstream, f"   +  exp(logT / {sri[2]:.15g}) ")
                        if syms is not None:
                            raise NotImplementedError("Not done for now")
                    else:
                        cw.writer(fstream, "   +  0. ")
                        if syms is not None:
                            raise NotImplementedError("Not done for now")
                    cw.writer(
                        fstream,
                        f"   *  ({nsri} > 3 ? {sri[3]:.15g} *"
                        f" exp({sri[4]:.15g} * logT) : 1.0);",
                    )
                    if syms is not None:
                        raise NotImplementedError("Not done for now")
                    cw.writer(fstream, "Corr = F * F_sri;")
                    cw.writer(
                        fstream,
                        f"const amrex::Real qf = Corr * k_f * ({forward_sc});",
                    )
                elif nlindemann > 0:
                    cw.writer(fstream, "Corr = redP / (1.0 + redP);")
                    corr_smp = redp_smp / (1.0 + redp_smp)
                    cw.writer(
                        fstream,
                        f"const amrex::Real qf = Corr * k_f * ({forward_sc});",
                    )
                    qf_smp = corr_smp * k_f_smp * forward_sc_smp
            if kc_conv_inv:
                if alpha is None:
                    if reverse_sc_smp == 0:
                        cw.writer(
                            fstream,
                            "const amrex::Real qr = 0.0;",
                        )
                    else:
                        cw.writer(
                            fstream,
                            "const amrex::Real qr = k_f *"
                            f" exp(-({kc_exp_arg})) * ({kc_conv_inv}) *"
                            f" ({reverse_sc});",
                        )
                    qr_smp = (
                        k_f_smp
                        * sme.exp(-(kc_exp_arg_smp))
                        * (kc_conv_inv_smp)
                        * (reverse_sc_smp)
                    )
                else:
                    cw.writer(
                        fstream,
                        "const amrex::Real qr = Corr * k_f *"
                        f" exp(-({kc_exp_arg})) * ({kc_conv_inv}) *"
                        f" ({reverse_sc});",
                    )
                    qr_smp = (
                        corr_smp
                        * k_f_smp
                        * sme.exp(-kc_exp_arg_smp)
                        * (kc_conv_inv_smp)
                        * (reverse_sc_smp)
                    )
            else:
                if alpha is None:
                    if reverse_sc_smp == 0.0:
                        cw.writer(
                            fstream,
                            "const amrex::Real qr = 0.0;",
                        )
                    else:
                        cw.writer(
                            fstream,
                            "const amrex::Real qr = k_f *"
                            f" exp(-({kc_exp_arg})) * ({reverse_sc});",
                        )
                    qr_smp = k_f_smp * sme.exp(-(kc_exp_arg_smp)) * reverse_sc_smp
                else:
                    cw.writer(
                        fstream,
                        "const amrex::Real qr = Corr * k_f *"
                        f" exp(-({kc_exp_arg})) * ({reverse_sc});",
                    )
                    qr_smp = (
                        corr_smp * k_f_smp * sme.exp(-(kc_exp_arg_smp)) * reverse_sc_smp
                    )

            remove_forward = cu.is_remove_forward(reaction_info, orig_idx)

            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(fstream, cw.comment("const amrex::Real qdot = qf - qr;"))
                cw.writer(fstream, "const amrex::Real qdot = - qr;")
                qdot_smp = -qr_smp
            else:
                cw.writer(fstream, "const amrex::Real qdot = qf - qr;")
                qdot_smp = qf_smp - qr_smp

            reaction = mechanism.reaction(orig_idx)
            lst_reactants = [(k, v) for k, v in reaction.reactants.items()]
            lst_products = [(k, v) for k, v in reaction.products.items()]
            all_agents = list(set(lst_reactants + lst_products))
            agents = []
            # remove QSS species from agents
            for symbol, coefficient in all_agents:
                if symbol not in species_info.qssa_species_list:
                    agents.append((symbol, coefficient))
            dict_species = {v: i for i, v in enumerate(species_info.all_species_list)}
            agents = sorted(
                agents,
                key=lambda v, dict_species=dict_species: dict_species[v[0]],
            )
            # Check for duplicates
            assert len(agents) == len(
                set(agents)
            ), f"Reaction {reaction} contains duplicate agents"
            # note that a species might appear as both reactant and product
            # a species might also appear twice or more on on each side
            # agents is a set that contains unique (symbol, coefficient)
            for a in agents:
                symbol, coefficient = a
                for b in reaction.reactants:
                    if b == a[0] and reaction.reactants[b] == a[1]:
                        if coefficient == 1.0:
                            cw.writer(
                                fstream,
                                f"wdot[{species_info.ordered_idx_map[symbol]}]"
                                " -= qdot;",
                            )
                            syms.wdot_smp[
                                species_info.ordered_idx_map[symbol]
                            ] -= qdot_smp
                        else:
                            cw.writer(
                                fstream,
                                f"wdot[{species_info.ordered_idx_map[symbol]}]"
                                f" -= {coefficient:f} * qdot;",
                            )
                            coefficient = syms.convert_number_to_int(coefficient)
                            syms.wdot_smp[species_info.ordered_idx_map[symbol]] -= (
                                coefficient * qdot_smp
                            )
                for b in reaction.products:
                    if b == a[0] and reaction.products[b] == a[1]:
                        if coefficient == 1.0:
                            cw.writer(
                                fstream,
                                f"wdot[{species_info.ordered_idx_map[symbol]}]"
                                " += qdot;",
                            )
                            syms.wdot_smp[
                                species_info.ordered_idx_map[symbol]
                            ] += qdot_smp
                        else:
                            cw.writer(
                                fstream,
                                f"wdot[{species_info.ordered_idx_map[symbol]}]"
                                f" += {coefficient:f} * qdot;",
                            )
                            coefficient = syms.convert_number_to_int(coefficient)
                            syms.wdot_smp[species_info.ordered_idx_map[symbol]] += (
                                coefficient * qdot_smp
                            )

            cw.writer(fstream, "}")
            cw.writer(fstream)

    cw.writer(fstream)

    cw.writer(fstream, "}")

    cw.writer(fstream)


def production_rate_light(fstream, mechanism, species_info, reaction_info):
    """Write low memory production rate."""
    n_species = species_info.n_species
    n_reactions = mechanism.n_reactions

    assert len(reaction_info.index) == 7

    itroe = reaction_info.index[0:2]
    isri = reaction_info.index[1:3]
    ilindemann = reaction_info.index[2:4]
    # i3body = reaction_info.index[3:5]
    # isimple = reaction_info.index[4:6]
    # ispecial = reaction_info.index[5:7]

    ntroe = itroe[1] - itroe[0]
    nsri = isri[1] - isri[0]
    nlindemann = ilindemann[1] - ilindemann[0]
    # n3body = i3body[1] - i3body[0]
    # nsimple = isimple[1] - isimple[0]
    # nspecial = ispecial[1] - ispecial[0]

    # qdot
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment(
            "GPU version of productionRate: no more use of thermo namespace vectors "
        ),
    )
    # main function
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void "
        "productionRate_light("
        "amrex::Real * wdot,"
        "amrex::Real * sc,"
        "amrex::Real * g_RT,"
        "amrex::Real * g_RT_qss,"
        "amrex::Real * sc_qss,"
        "amrex::Real * kf_qss,"
        "amrex::Real * qf_qss,"
        "amrex::Real * qr_qss,"
        "const amrex::Real T,"
        "const amrex::Real invT,"
        "const amrex::Real logT)",
    )
    cw.writer(fstream, "{")

    if n_reactions == 0:
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
        cw.writer(fstream, "const amrex::Real refCinv = 1 / refC;")

        if nsri > 0:
            cw.writer(fstream, "amrex::Real X, F_sri;")

    cw.writer(fstream)
    cw.writer(fstream, f"for (int i = 0; i < {n_species}; ++i) {{")
    cw.writer(fstream, "wdot[i] = 0.0;")
    cw.writer(fstream, "}")
    cw.writer(fstream)

    if n_reactions > 0:
        # nclassd = n_reactions - nspecial
        # nCorr   = n3body + ntroe + nsri + nlindemann

        # Mixt concentration for PD & TB
        cw.writer(fstream, cw.comment("compute the mixture concentration"))
        cw.writer(fstream, "amrex::Real mixture = 0.0;")
        cw.writer(fstream, f"for (int i = 0; i < {n_species}; ++i) {{")
        cw.writer(fstream, "mixture += sc[i];")
        cw.writer(fstream, "}")
        cw.writer(fstream)

        # Kc stuff
        cw.writer(fstream, cw.comment("compute the Gibbs free energy"))
        cw.writer(fstream, "gibbs(g_RT, T);")
        if species_info.n_qssa_species > 0:
            cw.writer(fstream, "gibbs_qss(g_RT_qss, T);")
        cw.writer(fstream)

        if species_info.n_qssa_species > 0:
            cw.writer(fstream, cw.comment("Fill sc_qss here"))
            cw.writer(fstream, "comp_k_f_qss(T, invT, logT, kf_qss);")
            # cw.writer(fstream,"comp_Kc_qss(invT, g_RT, g_RT_qss, Kc_qss);")
            cw.writer(
                fstream,
                "comp_qss_coeff(kf_qss, qf_qss, qr_qss, sc, T, g_RT, g_RT_qss);",
            )
            cw.writer(fstream, "comp_sc_qss(sc_qss, qf_qss, qr_qss);")
            cw.writer(fstream)

        # Loop like you're going through them in the mech.Linp order
        for orig_idx, _ in reaction_info.idxmap.items():
            reaction = mechanism.reaction(orig_idx)
            cw.writer(fstream, "{")
            if bool(reaction.orders):
                forward_sc = cu.qss_sorted_phase_space(
                    mechanism, species_info, reaction, reaction.orders
                )
            else:
                forward_sc = cu.qss_sorted_phase_space(
                    mechanism, species_info, reaction, reaction.reactants
                )
            if reaction.reversible:
                reverse_sc = cu.qss_sorted_phase_space(
                    mechanism, species_info, reaction, reaction.products
                )
            else:
                reverse_sc = "0.0"

            kc_exp_arg = cu.sorted_kc_exp_arg(mechanism, species_info, reaction)
            kc_conv_inv = cu.fkc_conv_inv(mechanism, species_info, reaction)

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
                ae = (
                    reaction.rate.activation_energy * cc.ureg.joule / cc.ureg.kmol
                ).to(aeuc)
            elif not falloff:
                # Case 2 !PD, TB
                ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), -dim)
                pef = (reaction.rate.pre_exponential_factor * ctuc).to_base_units()
                beta = reaction.rate.temperature_exponent
                ae = (
                    reaction.rate.activation_energy * cc.ureg.joule / cc.ureg.kmol
                ).to(aeuc)
            else:
                # Case 1 PD, TB
                ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
                pef = (
                    reaction.rate.high_rate.pre_exponential_factor * ctuc
                ).to_base_units()
                beta = reaction.rate.high_rate.temperature_exponent
                ae = (
                    reaction.rate.high_rate.activation_energy
                    * cc.ureg.joule
                    / cc.ureg.kmol
                ).to(aeuc)

                low_pef = (
                    reaction.rate.low_rate.pre_exponential_factor * ctuc
                ).to_base_units()
                low_beta = reaction.rate.low_rate.temperature_exponent
                low_ae = (
                    reaction.rate.low_rate.activation_energy
                    * cc.ureg.joule
                    / cc.ureg.kmol
                ).to(aeuc)
                if is_troe:
                    troe = reaction.rate.falloff_coeffs
                    ntroe = len(troe)
                    is_troe = True
                elif is_sri:
                    sri = reaction.rate.falloff_coeffs
                    nsri = len(sri)
                elif is_lindemann:
                    pass
                else:
                    raise ValueError(
                        f"Unrecognized reaction rate type {reaction.rate.type},"
                        f" {reaction.rate.sub_type} for reaction: {reaction.equation}"
                    )

            cw.writer(
                fstream,
                cw.comment(f"reaction {orig_idx}:  {reaction.equation}"),
            )
            cw.writer(fstream, f"const amrex::Real k_f = {pef.m:.15g}")

            if (beta == 0) and (ae == 0):
                cw.writer(fstream, "           ;")
            else:
                if ae == 0:
                    cw.writer(fstream, f"           * exp(({beta:.15g}) * logT);")
                elif beta == 0:
                    cw.writer(
                        fstream,
                        "           *"
                        f" exp(-({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g})"
                        " * invT);",
                    )
                else:
                    cw.writer(
                        fstream,
                        f"           * exp(({beta:.15g}) * logT -"
                        f" ({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g}) *"
                        " invT);",
                    )

            alpha = None
            if not third_body and not falloff:
                cw.writer(
                    fstream,
                    f"const amrex::Real qf = k_f * ({forward_sc});",
                )
            elif (
                not falloff
                and len(reaction.third_body.efficiencies) == 1
                and isclose(reaction.third_body.default_efficiency, 0.0)
            ):
                cw.writer(
                    fstream,
                    f"const amrex::Real qf = k_f * ({forward_sc});",
                )
            elif not falloff:
                alpha = enhancement_d_with_qss(mechanism, species_info, reaction)
                cw.writer(fstream, f"const amrex::Real Corr = {alpha};")
                cw.writer(
                    fstream,
                    f"const amrex::Real qf = Corr * k_f * ({forward_sc});",
                )
            else:
                alpha = enhancement_d_with_qss(mechanism, species_info, reaction)
                cw.writer(fstream, f"amrex::Real Corr = {alpha};")
                cw.writer(
                    fstream,
                    "const amrex::Real redP = Corr / k_f *"
                    f" {10 ** (-dim * 6) * low_pef.m * 10 ** 3 ** dim:.15g} ",
                )
                cw.writer(
                    fstream,
                    f"           * exp({low_beta:.15g} * logT -"
                    f" {(1.0 / cc.Rc / cc.ureg.kelvin * low_ae).m:.15g} *"
                    " invT);",
                )
                if is_troe:
                    cw.writer(fstream, "const amrex::Real F = redP / (1.0 + redP);")
                    cw.writer(fstream, "const amrex::Real logPred = log10(redP);")
                    cw.writer(fstream, "const amrex::Real logFcent = log10(")
                    if abs(troe[1]) > 1.0e-100:
                        if 1.0 - troe[0] != 0:
                            cw.writer(
                                fstream,
                                f"    {1.0 - troe[0]:.15g} * exp(-T *"
                                f" {1 / troe[1]:.15g})",
                            )
                    else:
                        cw.writer(fstream, "     0.0 ")
                    if abs(troe[2]) > 1.0e-100:
                        if troe[0] != 0:
                            cw.writer(
                                fstream,
                                f"    + {troe[0]:.15g} * exp(-T * {1 / troe[2]:.15g})",
                            )
                    else:
                        cw.writer(fstream, "    + 0.0 ")
                    if ntroe == 4:
                        if troe[3] < 0:
                            cw.writer(fstream, f"    + exp({-troe[3]:.15g} * invT));")
                        else:
                            cw.writer(fstream, f"    + exp(-{troe[3]:.15g} * invT));")
                    else:
                        cw.writer(fstream, "    + 0.0);")
                    cw.writer(
                        fstream,
                        "const amrex::Real troe_c = -0.4 - 0.67 * logFcent;",
                    )
                    cw.writer(
                        fstream,
                        "const amrex::Real troe_n = 0.75 - 1.27 * logFcent;",
                    )
                    cw.writer(
                        fstream,
                        "const amrex::Real troe = (troe_c + logPred) /"
                        " (troe_n - 0.14 * (troe_c + logPred));",
                    )
                    cw.writer(
                        fstream,
                        "const amrex::Real F_troe = exp(M_LN10 *logFcent /"
                        " (1.0 + troe * troe));",
                    )
                    cw.writer(fstream, "Corr = F * F_troe;")
                    cw.writer(
                        fstream,
                        f"const amrex::Real qf = Corr * k_f * ({forward_sc});",
                    )
                elif is_sri:
                    cw.writer(fstream, "const amrex::Real F = redP / (1.0 + redP);")
                    cw.writer(fstream, "const amrex::Real logPred = log10(redP);")
                    cw.writer(fstream, "X = 1.0 / (1.0 + logPred*logPred);")
                    if sri[1] < 0:
                        cw.writer(
                            fstream,
                            f"F_sri = exp(X * log({sri[0]:.15g} *"
                            f" exp({-sri[1]:.15g} * invT)",
                        )
                    else:
                        cw.writer(
                            fstream,
                            f"F_sri = exp(X * log({sri[0]:.15g} *"
                            f" exp(-{sri[1]:.15g} * invT)",
                        )
                    if sri[2] > 1.0e-100:
                        cw.writer(fstream, f"   +  exp(logT / {sri[2]:.15g}) ")
                    else:
                        cw.writer(fstream, "   +  0. ")
                    cw.writer(
                        fstream,
                        f"   *  ({nsri} > 3 ? {sri[3]:.15g} *"
                        f" exp({sri[4]:.15g} * logT) : 1.0);",
                    )
                    cw.writer(fstream, "Corr = F * F_sri;")
                    cw.writer(
                        fstream,
                        f"const amrex::Real qf = Corr * k_f * ({forward_sc});",
                    )
                elif nlindemann > 0:
                    cw.writer(fstream, "Corr = redP / (1.0 + redP);")
                    cw.writer(
                        fstream,
                        f"const amrex::Real qf = Corr * k_f * ({forward_sc});",
                    )
            if kc_conv_inv:
                if alpha is None:
                    if reverse_sc == "0.0":
                        cw.writer(
                            fstream,
                            "const amrex::Real qr = 0.0;",
                        )
                    else:
                        cw.writer(
                            fstream,
                            "const amrex::Real qr = k_f *"
                            f" exp(-({kc_exp_arg})) * ({kc_conv_inv}) *"
                            f" ({reverse_sc});",
                        )
                else:
                    cw.writer(
                        fstream,
                        "const amrex::Real qr = Corr * k_f *"
                        f" exp(-({kc_exp_arg})) * ({kc_conv_inv}) *"
                        f" ({reverse_sc});",
                    )
            else:
                if alpha is None:
                    if reverse_sc == "0.0":
                        cw.writer(
                            fstream,
                            "const amrex::Real qr = 0.0;",
                        )
                    else:
                        cw.writer(
                            fstream,
                            "const amrex::Real qr = k_f *"
                            f" exp(-({kc_exp_arg})) * ({reverse_sc});",
                        )
                else:
                    cw.writer(
                        fstream,
                        "const amrex::Real qr = Corr * k_f *"
                        f" exp(-({kc_exp_arg})) * ({reverse_sc});",
                    )
            remove_forward = cu.is_remove_forward(reaction_info, orig_idx)

            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(fstream, cw.comment("const amrex::Real qdot = qf - qr;"))
                cw.writer(fstream, "const amrex::Real qdot = - qr;")
            else:
                cw.writer(fstream, "const amrex::Real qdot = qf - qr;")

            reaction = mechanism.reaction(orig_idx)
            lst_reactants = [(k, v) for k, v in reaction.reactants.items()]
            lst_products = [(k, v) for k, v in reaction.products.items()]
            all_agents = list(set(lst_reactants + lst_products))
            agents = []
            # remove QSS species from agents
            for symbol, coefficient in all_agents:
                if symbol not in species_info.qssa_species_list:
                    agents.append((symbol, coefficient))
            dict_species = {v: i for i, v in enumerate(species_info.all_species_list)}
            agents = sorted(
                agents,
                key=lambda v, dict_species=dict_species: dict_species[v[0]],
            )
            # Check for duplicates
            assert len(agents) == len(
                set(agents)
            ), f"Reaction {reaction} contains duplicate agents"
            # note that a species might appear as both reactant and product
            # a species might also appear twice or more on on each side
            # agents is a set that contains unique (symbol, coefficient)
            for a in agents:
                symbol, coefficient = a
                for b in reaction.reactants:
                    if b == a[0] and reaction.reactants[b] == a[1]:
                        if coefficient == 1.0:
                            cw.writer(
                                fstream,
                                f"wdot[{species_info.ordered_idx_map[symbol]}]"
                                " -= qdot;",
                            )
                        else:
                            cw.writer(
                                fstream,
                                f"wdot[{species_info.ordered_idx_map[symbol]}]"
                                f" -= {coefficient:f} * qdot;",
                            )
                for b in reaction.products:
                    if b == a[0] and reaction.products[b] == a[1]:
                        if coefficient == 1.0:
                            cw.writer(
                                fstream,
                                f"wdot[{species_info.ordered_idx_map[symbol]}]"
                                " += qdot;",
                            )
                        else:
                            cw.writer(
                                fstream,
                                f"wdot[{species_info.ordered_idx_map[symbol]}]"
                                f" += {coefficient:f} * qdot;",
                            )
            cw.writer(fstream, "}")
            cw.writer(fstream)

    cw.writer(fstream)

    cw.writer(fstream, "return;")
    cw.writer(fstream, "}")

    cw.writer(fstream)


def progress_rate_fr(fstream, mechanism, species_info, reaction_info):
    """Write progress rates."""
    n_reactions = mechanism.n_reactions

    assert len(reaction_info.index) == 7

    cw.writer(fstream)
    cw.writer(fstream, cw.comment("compute the progress rate for each reaction"))
    cw.writer(fstream, cw.comment("USES progressRate : todo switch to GPU"))
    if n_reactions > 0:
        cw.writer(
            fstream,
            "void progressRateFR"
            + "(amrex::Real *  q_f, amrex::Real *  q_r, amrex::Real *  sc,"
            " amrex::Real T)",
        )
    else:
        cw.writer(
            fstream,
            "void progressRateFR"
            + "(amrex::Real *  /*q_f*/, amrex::Real *  /*q_r*/, amrex::Real *  /*sc*/,"
            " amrex::Real /*T*/)",
        )

    cw.writer(fstream, "{")

    if n_reactions > 0:
        cw.writer(fstream, "const amrex::Real invT = 1.0 / T;")
        cw.writer(fstream, "const amrex::Real logT = log(T);")

        cw.writer(fstream, cw.comment("compute the Gibbs free energy"))
        cw.writer(fstream, f"amrex::Real g_RT[{species_info.n_species}];")
        cw.writer(fstream, "gibbs(g_RT, T);")
        if species_info.n_qssa_species > 0:
            cw.writer(
                fstream,
                f"amrex::Real g_RT_qss[{species_info.n_qssa_species}];",
            )
            cw.writer(fstream, "gibbs_qss(g_RT_qss, T);")

        cw.writer(fstream)
        cw.writer(
            fstream,
            f"amrex::Real sc_qss[{max(1, species_info.n_qssa_species)}];",
        )
        if species_info.n_qssa_species > 0:
            cw.writer(fstream, cw.comment("Fill sc_qss here"))
            cw.writer(
                fstream,
                f"amrex::Real kf_qss[{reaction_info.n_qssa_reactions}],"
                f" qf_qss[{reaction_info.n_qssa_reactions}],"
                f" qr_qss[{reaction_info.n_qssa_reactions}];",
            )
            cw.writer(fstream, "comp_k_f_qss(T, invT, logT, kf_qss);")
            cw.writer(
                fstream,
                "comp_qss_coeff(kf_qss, qf_qss, qr_qss, sc, T, g_RT, g_RT_qss);",
            )
            cw.writer(fstream, "comp_sc_qss(sc_qss, qf_qss, qr_qss);")

        cw.writer(fstream, "comp_qfqr(q_f, q_r, sc, sc_qss, T, invT, logT);")
        cw.writer(fstream)

    cw.writer(fstream, "}")


def enhancement_d_with_qss(mechanism, species_info, reaction, syms=None):
    """Write enhancement with QSS."""
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
            factor = f"({efficiency:.15g})"
            if record_symbolic_operations:
                factor_smp = efficiency
            conc = f"sc[{species_info.ordered_idx_map[symbol]}]"
            if record_symbolic_operations:
                conc_smp = syms.sc_smp[species_info.ordered_idx_map[symbol]]
            if efficiency == 1:
                alpha.append(f"{conc}")
                if record_symbolic_operations:
                    alpha_smp.append(conc_smp)
            else:
                alpha.append(f"{factor}*{conc}")
                if record_symbolic_operations:
                    alpha_smp.append(factor_smp * conc_smp)
        else:
            factor = f"({efficiency:.15g})"
            if record_symbolic_operations:
                factor_smp = efficiency
            if (efficiency) != 0:
                idx = species_info.ordered_idx_map[symbol] - species_info.n_species
                conc = f"sc_qss[{idx}]"
                if record_symbolic_operations:
                    conc_smp = syms.sc_qss_smp[
                        species_info.ordered_idx_map[symbol] - species_info.n_species
                    ]
                if (efficiency) == 1:
                    alpha.append(f"{conc}")
                    if record_symbolic_operations:
                        alpha_smp.append(conc_smp)
                else:
                    alpha.append(f"{factor}*{conc}")
                    if record_symbolic_operations:
                        alpha_smp.append(factor_smp * conc_smp)
    
        if record_symbolic_operations:
            alpha_val = alpha_smp[0]
            enhancement_smp = 0.0 + alpha_val
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
            factor = f"({efficiency - 1:.15g})"
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
                        alpha_smp.append(factor_smp * conc_smp)
        else:
            factor = f"({efficiency - 1:.15g})"
            if record_symbolic_operations:
                factor_smp = efficiency - 1
            if (efficiency - 1) != 0:
                idx = species_info.ordered_idx_map[symbol] - species_info.n_species
                conc = f"sc_qss[{idx}]"
                if record_symbolic_operations:
                    conc_smp = syms.sc_qss_smp[
                        species_info.ordered_idx_map[symbol] - species_info.n_species
                    ]
                if (efficiency - 1) == 1:
                    alpha.append(f"{conc}")
                    if record_symbolic_operations:
                        alpha_smp.append(conc_smp)
                else:
                    alpha.append(f"{factor}*{conc}")
                    if record_symbolic_operations:
                        alpha_smp.append(factor_smp * conc_smp)

    if record_symbolic_operations:
        enhancement_smp = 0.0
        for alpha_val in alpha_smp:
            enhancement_smp += alpha_val

    if not record_symbolic_operations:
        return " + ".join(alpha).replace("+ -", "- ")
    else:
        return " + ".join(alpha).replace("+ -", "- "), enhancement_smp
