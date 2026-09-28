"""Write jacobian functions."""

import copy
from collections import Counter, OrderedDict
from math import isclose

import ceptr.constants as cc
import ceptr.utilities as cu
import ceptr.writer as cw


def ajac(
    fstream,
    mechanism,
    species_info,
    reaction_info,
    jacobian=True,
    precond=False,
    syms=None,
):
    """Write jacobian for a reaction."""
    n_species = species_info.n_species
    n_reactions = mechanism.n_reactions

    cw.writer(fstream)
    if precond:
        cw.writer(fstream, cw.comment("compute an approx to the reaction Jacobian"))
    else:
        cw.writer(fstream, cw.comment("compute the reaction Jacobian"))
    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE")
    if n_reactions > 0:
        if precond:
            cw.writer(
                fstream,
                "void aJacobian_precond(amrex::Real *  J, const"
                " amrex::Real *  sc, const amrex::Real T, const int HP)",
            )
        else:
            cw.writer(
                fstream,
                "void aJacobian(amrex::Real * J, const amrex::Real * sc,"
                " const amrex::Real T, const int consP)",
            )
    else:
        if precond:
            cw.writer(
                fstream,
                "void aJacobian_precond(amrex::Real *  J, const"
                " amrex::Real *  /*sc*/, const amrex::Real /*T*/, const"
                " int /*HP*/)",
            )
        else:
            cw.writer(
                fstream,
                "void aJacobian(amrex::Real * J, const amrex::Real *"
                " /*sc*/, const amrex::Real /*T*/, const int /*consP*/)",
            )
    cw.writer(fstream, "{")

    cw.writer(fstream)
    # Analytical jacobian not ready with QSS
    if species_info.n_qssa_species > 0:
        cw.writer(
            fstream,
            cw.comment(
                "Analytical Jacobian with QSSA is only supported with symbolic"
                " implementation. Re-build in ceptr with -qsj flag."
            ),
        )
        cw.writer(fstream, "amrex::Abort();")
        cw.writer(fstream)
    elif not jacobian:
        cw.writer(
            fstream,
            cw.comment(
                "Mechanism was generated without a jacobian."
                " Re-build in ceptr without the -nj flag."
            ),
        )
        cw.writer(fstream, "amrex::Abort();")
    else:
        cw.writer(
            fstream,
            "#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)",
        )
        cw.writer(fstream, f"for (int i=0; i<{(n_species + 1) ** 2}; i++) {{")
        cw.writer(fstream, "J[i] = 0.0;")
        cw.writer(fstream, "}")

        if n_reactions > 0:
            cw.writer(fstream)

            cw.writer(fstream, f"amrex::Real wdot[{n_species}];")
            cw.writer(fstream, "for (auto& val : wdot) {")
            cw.writer(fstream, "val = 0.0;")
            cw.writer(fstream, "}")

            cw.writer(fstream)

            cw.writer(fstream, "const amrex::Real invT = 1.0 / T;")
            cw.writer(fstream, "const amrex::Real invT2 = invT * invT;")
            cw.writer(fstream, "const amrex::Real logT = log(T);")

            cw.writer(fstream)

            cw.writer(
                fstream,
                cw.comment("reference concentration: P_atm / (RT) in inverse mol/m^3"),
            )
            cw.writer(
                fstream,
                f"amrex::Real refC = {cc.Patm_pa:g} /"
                f" {cc.R.to(cc.ureg.joule / (cc.ureg.mole / cc.ureg.kelvin)).m:g} / T;",
            )
            cw.writer(fstream, "amrex::Real refCinv = 1.0 / refC;")

            cw.writer(fstream)

            cw.writer(fstream, cw.comment("compute the mixture concentration"))
            cw.writer(fstream, "amrex::Real mixture = 0.0;")
            cw.writer(fstream, f"for (int k = 0; k < {n_species}; ++k) {{")
            cw.writer(fstream, "mixture += sc[k];")
            cw.writer(fstream, "}")

            cw.writer(fstream)

            cw.writer(fstream, cw.comment("compute the Gibbs free energy"))
            cw.writer(fstream, f"amrex::Real g_RT[{n_species}];")
            cw.writer(fstream, "gibbs(g_RT, T);")
            if species_info.n_qssa_species > 0:
                cw.writer(
                    fstream,
                    f"amrex::Real g_RT_qss[{species_info.n_qssa_species}];",
                )
                cw.writer(fstream, "gibbs_qss(g_RT_qss, T);")

            cw.writer(fstream)

            cw.writer(fstream, cw.comment("compute the species enthalpy"))
            cw.writer(fstream, f"amrex::Real h_RT[{n_species}];")
            cw.writer(fstream, "speciesEnthalpy(h_RT, T);")
            if species_info.n_qssa_species > 0:
                cw.writer(
                    fstream,
                    f"amrex::Real h_RT_qss[{species_info.n_qssa_species}];",
                )
                cw.writer(fstream, "speciesEnthalpy_qss(h_RT_qss, T);")

            if species_info.n_qssa_species > 0:
                cw.writer(fstream)
                cw.writer(fstream, cw.comment("Fill sc_qss here"))
                cw.writer(
                    fstream,
                    f"amrex::Real sc_qss[{species_info.n_qssa_species}];",
                )
                cw.writer(
                    fstream,
                    "amrex::Real"
                    f" kf_qss[{reaction_info.n_qssa_reactions}],"
                    f" qf_qss[{reaction_info.n_qssa_reactions}],"
                    f" qr_qss[{reaction_info.n_qssa_reactions}];",
                )
                cw.writer(fstream, "comp_k_f_qss(T, invT, logT, kf_qss);")
                cw.writer(
                    fstream,
                    "comp_qss_coeff(kf_qss, qf_qss, qr_qss, sc, T, g_RT, g_RT_qss);",
                )
                cw.writer(fstream, "comp_sc_qss(sc_qss, qf_qss, qr_qss);")
                cw.writer(fstream)

            cw.writer(fstream)

            cw.writer(
                fstream,
                "amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;",
            )
            cw.writer(fstream, "amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;")
            cw.writer(fstream, f"amrex::Real dqdci, dcdc_fac, dqdc[{n_species}];")
            cw.writer(fstream, "amrex::Real Pr, fPr, F, k_0, logPr;")
            cw.writer(
                fstream,
                "amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;",
            )
            cw.writer(fstream, "amrex::Real Fcent1, Fcent2, Fcent3, Fcent;")
            cw.writer(fstream, "amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;")
            cw.writer(
                fstream,
                "amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT,"
                " dlogFdlogPr, dlnCorrdT;",
            )
            cw.writer(fstream, "const amrex::Real ln10 = log(10.0);")
            cw.writer(fstream, "const amrex::Real log10e = 1.0/log(10.0);")

            for orig_idx, _ in reaction_info.idxmap.items():
                reaction = mechanism.reaction(orig_idx)

                cw.writer(
                    fstream,
                    cw.comment(f"reaction {orig_idx}: {reaction.equation}"),
                )
                ajac_reaction_d(
                    fstream,
                    mechanism,
                    species_info,
                    reaction_info,
                    reaction,
                    orig_idx,
                    precond=precond,
                    syms=syms,
                )
                cw.writer(fstream)

            cw.writer(
                fstream,
                f"amrex::Real c_R[{n_species}], dcRdT[{n_species}], e_RT[{n_species}];",
            )
            cw.writer(fstream, "amrex::Real * eh_RT;")
            if precond:
                cw.writer(fstream, "if (HP == 1) {")
            else:
                cw.writer(fstream, "if (consP == 1) {")

            cw.writer(fstream, "cp_R(c_R, T);")
            cw.writer(fstream, "dcvpRdT(dcRdT, T);")
            cw.writer(fstream, "eh_RT = &h_RT[0];")

            cw.writer(fstream, "}")
            cw.writer(fstream, "else {")

            cw.writer(fstream, "cv_R(c_R, T);")
            cw.writer(fstream, "dcvpRdT(dcRdT, T);")
            cw.writer(fstream, "speciesInternalEnergy(e_RT, T);")
            cw.writer(fstream, "eh_RT = &e_RT[0];")

            cw.writer(fstream, "}")

            cw.writer(fstream)

            cw.writer(
                fstream,
                "amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;",
            )
            cw.writer(fstream, f"for (int k = 0; k < {n_species}; ++k) {{")
            cw.writer(fstream, "cmix += c_R[k]*sc[k];")
            cw.writer(fstream, "dcmixdT += dcRdT[k]*sc[k];")
            cw.writer(fstream, "ehmix += eh_RT[k]*wdot[k];")
            cw.writer(
                fstream,
                "dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] +"
                f" eh_RT[k]*J[{n_species * (n_species + 1)}+k];",
            )
            cw.writer(fstream, "}")

            cw.writer(fstream)
            cw.writer(fstream, "amrex::Real cmixinv = 1.0/cmix;")
            cw.writer(fstream, "amrex::Real tmp1 = ehmix*cmixinv;")
            cw.writer(fstream, "amrex::Real tmp3 = cmixinv*T;")
            cw.writer(fstream, "amrex::Real tmp2 = tmp1*tmp3;")
            cw.writer(fstream, "amrex::Real dehmixdc;")

            cw.writer(fstream, cw.comment("dTdot/d[X]"))
            cw.writer(fstream, f"for (int k = 0; k < {n_species}; ++k) {{")
            cw.writer(fstream, "dehmixdc = 0.0;")
            cw.writer(fstream, f"for (int m = 0; m < {n_species}; ++m) {{")
            cw.writer(fstream, f"dehmixdc += eh_RT[m]*J[k*{n_species + 1}+m];")
            cw.writer(fstream, "}")
            cw.writer(
                fstream,
                f"J[k*{n_species + 1}+{n_species}] = tmp2*c_R[k] - tmp3*dehmixdc;",
            )
            cw.writer(fstream, "}")

            cw.writer(fstream, cw.comment("dTdot/dT"))
            cw.writer(
                fstream,
                f"J[{n_species * (n_species + 1) + n_species}] = -tmp1 +"
                " tmp2*dcmixdT - tmp3*dehmixdT;",
            )
        cw.writer(fstream, "#else")
        cw.writer(fstream, "amrex::Abort();")
        cw.writer(fstream, "#endif")

    cw.writer(fstream, "}")


def ajac_symbolic(
    fstream,
    mechanism,
    species_info,
    reaction_info,
    jacobian=True,
    syms=None,
):
    """Print the Jacobian obtained from symbolic recording."""
    n_species = species_info.n_species
    n_qssa_species = species_info.n_qssa_species
    n_reactions = mechanism.n_reactions

    cw.writer(fstream)

    # main
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void "
        " aJacobian(amrex::Real * J, amrex::Real * sc, amrex::Real T,"
        " const int consP)",
    )
    cw.writer(fstream, "{")

    if not jacobian:
        cw.writer(
            fstream,
            cw.comment(
                "Mechanism was generated without a jacobian."
                " Re-build in ceptr without the -nj flag."
            ),
        )
        cw.writer(fstream, "amrex::Abort();")
        return

    cw.writer(
        fstream,
        "#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)",
    )

    if syms.hformat == "cpu":
        cw.writer(
            fstream,
            f"amrex::Real dscqss_dsc[{n_species * n_qssa_species}];",
        )
        cw.writer(
            fstream,
            f"for (int i=0; i<{(n_species * n_qssa_species)}; i++) {{",
        )
        cw.writer(fstream, "dscqss_dsc[i] = 0.0;")
        cw.writer(fstream, "}")

    cw.writer(fstream, "const amrex::Real invT = 1.0 / T;")
    cw.writer(fstream, "const amrex::Real logT = log(T);")
    cw.writer(fstream)

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

    cw.writer(fstream, f"amrex::Real g_RT[{species_info.n_species}];")
    cw.writer(fstream, f"amrex::Real h_RT[{n_species}];")
    if species_info.n_qssa_species > 0:
        cw.writer(
            fstream,
            f"amrex::Real g_RT_qss[{species_info.n_qssa_species}];",
        )
        cw.writer(
            fstream,
            f"amrex::Real h_RT_qss[{species_info.n_qssa_species}];",
        )
        cw.writer(
            fstream,
            f"amrex::Real sc_qss[{max(1, species_info.n_qssa_species)}];",
        )
        if syms.store_in_jacobian:
            cw.writer(
                fstream,
                f"amrex::Real kf_qss[{reaction_info.n_qssa_reactions}];",
            )
        else:
            cw.writer(
                fstream,
                f"amrex::Real kf_qss[{reaction_info.n_qssa_reactions}],"
                f" qf_qss[{reaction_info.n_qssa_reactions}],"
                f" qr_qss[{reaction_info.n_qssa_reactions}];",
            )

    # prepare dwdotdT
    cw.writer(fstream, "amrex::Real T_pert1, pertT;")
    cw.writer(
        fstream,
        f"amrex::Real wdot_pert1[{n_species}], wdot[{n_species}];",
    )
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("dwdot/dT by finite difference"))
    cw.writer(fstream, "pertT = 1e-2;")
    cw.writer(fstream, "T_pert1 = T + pertT;")
    cw.writer(fstream)
    if syms.store_in_jacobian:
        cw.writer(fstream, "const amrex::Real invT_pert1 = 1.0 / T_pert1;")
        cw.writer(fstream, "const amrex::Real logT_pert1 = log(T_pert1);")
        cw.writer(
            fstream,
            "productionRate_light(wdot_pert1, sc, g_RT, g_RT_qss, sc_qss,"
            f" kf_qss, &J[{0}], &J[{reaction_info.n_qssa_reactions}], T_pert1,"
            " invT_pert1, logT_pert1);",
        )
        cw.writer(
            fstream,
            "productionRate_light(wdot, sc, g_RT, g_RT_qss, sc_qss,"
            f" kf_qss, &J[{0}], &J[{reaction_info.n_qssa_reactions}], T,"
            " invT, logT);",
        )
    else:
        cw.writer(fstream, "productionRate(wdot_pert1, sc, T_pert1);")
        cw.writer(fstream, "productionRate(wdot, sc, T);")

    cw.writer(fstream)
    if n_reactions > 0:
        # nclassd = n_reactions - nspecial
        # nCorr   = n3body + ntroe + nsri + nlindemann

        # Kc stuff
        if not syms.store_in_jacobian:
            cw.writer(fstream, cw.comment("compute the Gibbs free energy"))
            cw.writer(fstream, "gibbs(g_RT, T);")
            if species_info.n_qssa_species > 0:
                cw.writer(fstream, "gibbs_qss(g_RT_qss, T);")
            cw.writer(fstream)
        cw.writer(fstream, cw.comment("compute the species enthalpy"))
        cw.writer(fstream, "speciesEnthalpy(h_RT, T);")
        if not syms.store_in_jacobian:
            if species_info.n_qssa_species > 0:
                cw.writer(fstream, "speciesEnthalpy_qss(h_RT_qss, T);")

        if species_info.n_qssa_species > 0:
            if not syms.store_in_jacobian:
                cw.writer(fstream, cw.comment("Fill sc_qss here"))
                cw.writer(fstream, "comp_k_f_qss(T, invT, logT, kf_qss);")
                cw.writer(
                    fstream,
                    "comp_qss_coeff(kf_qss, qf_qss, qr_qss, sc, T, g_RT, g_RT_qss);",
                )
                cw.writer(fstream, "comp_sc_qss(sc_qss, qf_qss, qr_qss);")
                cw.writer(fstream)
            # cw.writer(fstream,"comp_Kc_qss(invT, g_RT, g_RT_qss, Kc_qss);")

            # Initialize the big Jacobian array
            cw.writer(fstream, f"for (int i=0; i<{(n_species + 1) ** 2}; i++) {{")
            cw.writer(fstream, "J[i] = 0.0;")
            cw.writer(fstream, "}")

            # Now write out the species jacobian terms
            cw.writer(fstream, cw.comment("Species terms"))
            if syms.hformat == "cpu":
                syms.write_symjac_to_cpp_cpu(species_info, cw, fstream)
            else:
                syms.write_symjac_to_cpp_gpu(species_info, cw, fstream)

            cw.writer(fstream)

    # dwdotdT
    cw.writer(fstream)
    cw.writer(fstream, f"for (int k = 0; k < {n_species} ; k++) {{")
    cw.writer(
        fstream,
        f"J[{n_species * (n_species + 1)} + k] = (wdot_pert1[k] - wdot[k])/(pertT);",
    )
    cw.writer(fstream, "}")

    # depends on dwdotdT and dwdotdsc
    cw.writer(
        fstream,
        f"amrex::Real c_R[{n_species}], dcRdT[{n_species}], e_RT[{n_species}];",
    )
    cw.writer(fstream, "amrex::Real * eh_RT;")
    # if precond:
    #    cw.writer(fstream, "if (HP) {")
    # else:
    #    cw.writer(fstream, "if (consP) {")

    cw.writer(fstream, "if (consP == 1) {")

    cw.writer(fstream, "cp_R(c_R, T);")
    cw.writer(fstream, "dcvpRdT(dcRdT, T);")
    cw.writer(fstream, "eh_RT = &h_RT[0];")

    cw.writer(fstream, "}")
    cw.writer(fstream, "else {")

    cw.writer(fstream, "cv_R(c_R, T);")
    cw.writer(fstream, "dcvpRdT(dcRdT, T);")
    cw.writer(fstream, "speciesInternalEnergy(e_RT, T);")
    cw.writer(fstream, "eh_RT = &e_RT[0];")

    cw.writer(fstream, "}")

    cw.writer(fstream)

    cw.writer(
        fstream,
        "amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;",
    )
    cw.writer(fstream, f"for (int k = 0; k < {n_species}; ++k) {{")
    cw.writer(fstream, "cmix += c_R[k]*sc[k];")
    cw.writer(fstream, "dcmixdT += dcRdT[k]*sc[k];")
    cw.writer(fstream, "ehmix += eh_RT[k]*wdot[k];")
    cw.writer(
        fstream,
        "dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] +"
        f" eh_RT[k]*J[{n_species * (n_species + 1)}+k];",
    )
    cw.writer(fstream, "}")

    cw.writer(fstream)
    cw.writer(fstream, "amrex::Real cmixinv = 1.0/cmix;")
    cw.writer(fstream, "amrex::Real tmp1 = ehmix*cmixinv;")
    cw.writer(fstream, "amrex::Real tmp3 = cmixinv*T;")
    cw.writer(fstream, "amrex::Real tmp2 = tmp1*tmp3;")
    cw.writer(fstream, "amrex::Real dehmixdc;")

    cw.writer(fstream, cw.comment("dTdot/d[X]"))
    cw.writer(fstream, f"for (int k = 0; k < {n_species}; ++k) {{")
    cw.writer(fstream, "dehmixdc = 0.0;")
    cw.writer(fstream, f"for (int m = 0; m < {n_species}; ++m) {{")
    cw.writer(fstream, f"dehmixdc += eh_RT[m]*J[k*{n_species + 1}+m];")
    cw.writer(fstream, "}")
    cw.writer(
        fstream,
        f"J[k*{n_species + 1}+{n_species}] = tmp2*c_R[k] - tmp3*dehmixdc;",
    )
    cw.writer(fstream, "}")

    cw.writer(fstream, cw.comment("dTdot/dT"))
    cw.writer(
        fstream,
        f"J[{n_species * (n_species + 1) + n_species}] = -tmp1 +"
        " tmp2*dcmixdT - tmp3*dehmixdT;",
    )

    cw.writer(fstream, "#else")
    cw.writer(fstream, "amrex::Abort();")
    cw.writer(fstream, "#endif")
    cw.writer(fstream, "}")

    cw.writer(fstream)


def ajac_reaction_d(
    fstream,
    mechanism,
    species_info,
    reaction_info,
    reaction,
    orig_idx,
    precond=False,
    syms=None,
):
    """Write jacobian of reaction."""
    n_species = species_info.n_species
    remove_forward = cu.is_remove_forward(reaction_info, orig_idx)

    if bool(reaction.orders):
        dim = cu.phase_space_units(reaction.orders)
    else:
        dim = cu.phase_space_units(reaction.reactants)
    third_body = reaction.third_body is not None
    if reaction.reaction_type == "pressure-dependent-Arrhenius":
        plog = True
    elif reaction.reaction_type == "three-body-pressure-dependent-Arrhenius":
        if not third_body:
            raise ValueError(f"Inconsistent reaction type in reaction {orig_idx}.")
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
        cw.writer(
            fstream,
            cw.comment("a non-third-body and non-pressure-fall-off reaction"),
        )
        ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
        pef = (reaction.rate.pre_exponential_factor * ctuc).to_base_units()
        beta = reaction.rate.temperature_exponent
        ae = (reaction.rate.activation_energy * cc.ureg.joule / cc.ureg.kmol).to(aeuc)
    elif not falloff and not plog:
        # Case 2 !PD, TB
        cw.writer(
            fstream,
            cw.comment("a third-body and non-pressure-fall-off reaction"),
        )
        ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), -dim)
        pef = (reaction.rate.pre_exponential_factor * ctuc).to_base_units()
        beta = reaction.rate.temperature_exponent
        ae = (reaction.rate.activation_energy * cc.ureg.joule / cc.ureg.kmol).to(aeuc)
    elif not third_body and not falloff and plog:
        # Case 4 PLOG
        cw.writer(
            fstream,
            cw.comment("a non-third-body and non-pressure-fall-off reaction (PLOG)"),
        )
        ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
        plog_pef, plog_beta, plog_ae = cu.evaluate_plog(
            reaction.rate.rates, mechanism.P
        )
        pef = (plog_pef * ctuc).to_base_units()
        beta = plog_beta
        ae = (plog_ae * cc.ureg.joule / cc.ureg.kmol).to(aeuc)
    elif third_body and not falloff and plog:
        # Case 5 TB, PLOG
        cw.writer(
            fstream,
            cw.comment("a third-body, non-pressure-fall-off, and PLOG reaction"),
        )
        ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), -dim)
        plog_pef, plog_beta, plog_ae = cu.evaluate_plog(
            reaction.rate.rates, mechanism.P
        )
        pef = (plog_pef * ctuc).to_base_units()
        beta = plog_beta
        ae = (plog_ae * cc.ureg.joule / cc.ureg.kmol).to(aeuc)
    else:
        # Case 1 PD, TB
        cw.writer(
            fstream,
            cw.comment("a third-body and pressure-fall-off reaction"),
        )
        ctuc = cu.prefactor_units(cc.ureg("kmol/m**3"), 1 - dim)
        pef = (reaction.rate.high_rate.pre_exponential_factor * ctuc).to_base_units()
        beta = reaction.rate.high_rate.temperature_exponent
        ae = (
            reaction.rate.high_rate.activation_energy * cc.ureg.joule / cc.ureg.kmol
        ).to(aeuc)

        low_pef = (reaction.rate.low_rate.pre_exponential_factor * ctuc).to_base_units()
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

    has_alpha = False
    corr_s = ""
    if not third_body and not falloff:
        pass
    elif (
        not falloff
        and len(reaction.third_body.efficiencies) == 1
        and isclose(reaction.third_body.default_efficiency, 0.0)
    ):
        pass
    elif not falloff:
        corr_s = "alpha *"
        has_alpha = True
    else:
        corr_s = "Corr *"
        has_alpha = True

    rea_dict = OrderedDict()
    pro_dict = OrderedDict()
    all_dict = OrderedDict()
    all_wqss_dict = OrderedDict()
    sum_nuk = 0
    all_reactants = copy.deepcopy(reaction.reactants)
    all_products = copy.deepcopy(reaction.products)
    if reaction.third_body:
        if len(reaction.third_body.efficiencies) == 1:
            if isclose(reaction.third_body.default_efficiency, 0.0):
                # Do not update third-body falloff reactions
                if not (reaction.rate.type == "falloff"):
                    all_reactants = dict(
                        sum(
                            (
                                Counter(x)
                                for x in [
                                    all_reactants,
                                    reaction.third_body.efficiencies,
                                ]
                            ),
                            Counter(),
                        )
                    )
                    all_products = dict(
                        sum(
                            (
                                Counter(x)
                                for x in [
                                    all_products,
                                    reaction.third_body.efficiencies,
                                ]
                            ),
                            Counter(),
                        )
                    )

    # Build rea_dict containing reaction species
    for symbol, coefficient in all_reactants.items():
        k = species_info.ordered_idx_map[symbol]
        sum_nuk -= coefficient
        if k in rea_dict:
            coe_old = rea_dict[k][1]
            rea_dict[k] = (symbol, coefficient + coe_old)
        else:
            rea_dict[k] = (symbol, coefficient)

    # Build pro_dict containing product species
    for symbol, coefficient in all_products.items():
        k = species_info.ordered_idx_map[symbol]
        sum_nuk += coefficient
        if k in pro_dict:
            coe_old = pro_dict[k][1]
            pro_dict[k] = (symbol, coefficient + coe_old)
        else:
            pro_dict[k] = (symbol, coefficient)

    # Build the dict with species and coefficients
    for k in range(n_species):
        if k in rea_dict and k in pro_dict:
            sr, nur = rea_dict[k]
            sp, nup = pro_dict[k]
            all_dict[k] = (sr, nup - nur)
        elif k in rea_dict:
            sr, nur = rea_dict[k]
            all_dict[k] = (sr, -nur)
        elif k in pro_dict:
            sp, nup = pro_dict[k]
            all_dict[k] = (sp, nup)
        elif species_info.all_species_list[k] in reaction.orders:
            sp = species_info.all_species_list[k]
            all_dict[k] = (sp, 0)

    # Build the dict including qss species
    for k in range(len(species_info.all_species_list)):
        if k in rea_dict and k in pro_dict:
            sr, nur = rea_dict[k]
            sp, nup = pro_dict[k]
            all_wqss_dict[k] = (sr, nup - nur)
        elif k in rea_dict:
            sr, nur = rea_dict[k]
            all_wqss_dict[k] = (sr, -nur)
        elif k in pro_dict:
            sp, nup = pro_dict[k]
            all_wqss_dict[k] = (sp, nup)
        elif species_info.all_species_list[k] in reaction.orders:
            sp = species_info.all_species_list[k]
            all_wqss_dict[k] = (sp, 0)

    sorted_reactants = sorted(rea_dict.values())
    sorted_products = sorted(pro_dict.values())

    if not reaction.reversible:
        if falloff or has_alpha:
            print("FIXME: irreversible reaction in _ajac_reaction may not work")
            cw.writer(
                fstream,
                cw.comment(
                    "Irreversible reaction in _ajac_reaction may not work",
                ),
            )
        for k in range(n_species):
            if k in sorted_reactants and k in sorted_products:
                print("FIXME: irreversible reaction in _ajac_reaction may not work")
                cw.writer(
                    fstream,
                    cw.comment(
                        "Irreversible reaction in _ajac_reaction may not work",
                    ),
                )

    if has_alpha:
        cw.writer(fstream, cw.comment("3-body correction factor"))
        enhancement_d = cu.enhancement_d(
            # mechanism, species_info, reaction, syms
            mechanism,
            species_info,
            reaction,
            syms=None,
        )
        cw.writer(fstream, f"alpha = {enhancement_d};")
        if (
            falloff
            and len(reaction.third_body.efficiencies) == 1
            and isclose(reaction.third_body.default_efficiency, 0.0)
        ):
            # Avoid float point exception
            cw.writer(fstream, "alpha = std::max(alpha, 1e-20);")

    # forward
    qss_ps = cu.qss_sorted_phase_space(
        mechanism, species_info, reaction, reaction.reactants
    )
    cw.writer(fstream, cw.comment("forward"))
    cw.writer(
        fstream,
        f"phi_f = {qss_ps};",
    )
    cw.writer(fstream, f"k_f = {pef.m:.15g}")
    if (ae.m == 0) and (beta == 0):
        cw.writer(fstream, "           ;")
    elif ae.m == 0:
        cw.writer(
            fstream,
            f"            * exp({beta:.15g} * logT);",
        )
    elif beta == 0:
        cw.writer(
            fstream,
            "            * exp(-"
            f" ({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g}) * invT);",
        )
    else:
        cw.writer(
            fstream,
            f"            * exp({beta:.15g} * logT -"
            f" ({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g}) * invT);",
        )
    if remove_forward:
        cw.writer(fstream, cw.comment("Remove forward reaction"))
        cw.writer(
            fstream,
            cw.comment(
                f"dlnkfdT = {beta:.15g} * invT +"
                f" ({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g}) * invT2;"
            ),
        )
        cw.writer(fstream, "dlnkfdT = 0.0;")
    else:
        if (beta == 0) and (ae.m == 0):
            cw.writer(
                fstream,
                "dlnkfdT = 0.0;",
            )
        elif ae.m == 0:
            cw.writer(
                fstream,
                f"dlnkfdT = {beta:.15g} * invT;",
            )
        elif beta == 0:
            cw.writer(
                fstream,
                f"dlnkfdT = ({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g}) * invT2;",
            )
        else:
            cw.writer(
                fstream,
                f"dlnkfdT = {beta:.15g} * invT +"
                f" ({(1.0 / cc.Rc / cc.ureg.kelvin * ae).m:.15g}) * invT2;",
            )

    if falloff:
        cw.writer(fstream, cw.comment("pressure-fall-off"))
        if low_beta == 0 and low_ae.m == 0:
            cw.writer(
                fstream,
                f"k_0 = {low_pef.m * 10 ** 3 ** dim:.15g};",
            )
        elif low_ae.m == 0:
            cw.writer(
                fstream,
                f"k_0 = {low_pef.m * 10 ** 3 ** dim:.15g} *"
                f" exp({low_beta:.15g} * logT);",
            )
        elif low_beta == 0:
            cw.writer(
                fstream,
                f"k_0 = {low_pef.m * 10 ** 3 ** dim:.15g} *"
                f" exp(-({(1.0 / cc.Rc / cc.ureg.kelvin * low_ae).m:.15g})"
                " * invT);",
            )
        else:
            cw.writer(
                fstream,
                f"k_0 = {low_pef.m * 10 ** 3 ** dim:.15g} *"
                f" exp({low_beta:.15g} * logT -"
                f" ({(1.0 / cc.Rc / cc.ureg.kelvin * low_ae).m:.15g}) *"
                " invT);",
            )
        cw.writer(fstream, f"Pr = 1e-{int(dim * 6)} * alpha / k_f * k_0;")
        cw.writer(fstream, "fPr = Pr / (1.0+Pr);")
        if (low_beta == 0) and (low_ae.m == 0):
            cw.writer(
                fstream,
                "dlnk0dT = 0.0;",
            )
        elif low_ae.m == 0:
            cw.writer(
                fstream,
                f"dlnk0dT = {low_beta:.15g} * invT;",
            )
        elif low_beta == 0:
            cw.writer(
                fstream,
                "dlnk0dT ="
                f" ({(1.0 / cc.Rc / cc.ureg.kelvin * low_ae).m:.15g}) *"
                " invT2;",
            )
        else:
            cw.writer(
                fstream,
                f"dlnk0dT = {low_beta:.15g} * invT +"
                f" ({(1.0 / cc.Rc / cc.ureg.kelvin * low_ae).m:.15g}) *"
                " invT2;",
            )
        cw.writer(fstream, "dlogPrdT = log10e*(dlnk0dT - dlnkfdT);")
        cw.writer(fstream, "dlogfPrdT = dlogPrdT / (1.0+Pr);")
        #
        if is_sri:
            cw.writer(fstream, cw.comment("SRI form"))
            raise NotImplementedError("FIXME: sri not supported in _ajac_reaction yet")
        elif is_troe:
            cw.writer(fstream, cw.comment("Troe form"))
            troe = reaction.rate.falloff_coeffs
            ntroe = len(troe)
            cw.writer(fstream, "logPr = log10(Pr);")
            if abs(troe[1]) > 1.0e-100:
                if troe[0] < 0:
                    cw.writer(
                        fstream,
                        f"Fcent1 = (1.+{-troe[0]:.15g})*exp(-T/{troe[1]:.15g});",
                    )
                else:
                    cw.writer(
                        fstream,
                        f"Fcent1 = (1.-{troe[0]:.15g})*exp(-T/{troe[1]:.15g});",
                    )
            else:
                cw.writer(fstream, "Fcent1 = 0.;")
            if abs(troe[2]) > 1.0e-100:
                cw.writer(
                    fstream,
                    f"Fcent2 = {troe[0]:.15g} * exp(-T/{troe[2]:.15g});",
                )
            else:
                cw.writer(fstream, "Fcent2 = 0.;")
            if ntroe == 4:
                if troe[3] < 0:
                    cw.writer(fstream, f"Fcent3 = exp({-troe[3]:.15g} * invT);")
                else:
                    cw.writer(fstream, f"Fcent3 = exp(-{troe[3]:.15g} * invT);")
            else:
                cw.writer(fstream, "Fcent3 = 0.;")
            cw.writer(fstream, "Fcent = Fcent1 + Fcent2 + Fcent3;")
            cw.writer(fstream, "logFcent = log10(Fcent);")
            cw.writer(fstream, "troe_c = -.4 - .67 * logFcent;")
            cw.writer(fstream, "troe_n = .75 - 1.27 * logFcent;")
            cw.writer(fstream, "troePr_den = 1.0 / (troe_n - .14*(troe_c + logPr));")
            cw.writer(fstream, "troePr = (troe_c + logPr) * troePr_den;")
            cw.writer(fstream, "troe = 1.0 / (1.0 + troePr*troePr);")
            cw.writer(fstream, "F = exp(M_LN10 * logFcent * troe);")

            cw.writer(fstream, "dlogFcentdT = log10e/Fcent*( ")
            if abs(troe[1]) > 1.0e-100:
                cw.writer(fstream, f"    -Fcent1/{troe[1]:.15g}")
            if abs(troe[2]) > 1.0e-100:
                cw.writer(fstream, f"    -Fcent2/{troe[2]:.15g}")
            if ntroe == 4:
                if abs(troe[3]) > 1.0e-100:
                    cw.writer(fstream, f"    + Fcent3*{troe[3]:.15g}*invT2")
            cw.writer(fstream, ");")

            cw.writer(
                fstream,
                "dlogFdcn_fac = 2.0 * logFcent * troe*troe * troePr * troePr_den;",
            )
            cw.writer(fstream, "dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;")
            cw.writer(fstream, "dlogFdn = dlogFdcn_fac * troePr;")
            cw.writer(fstream, "dlogFdlogPr = dlogFdc;")
            cw.writer(
                fstream,
                "dlogFdT = dlogFcentdT*(troe - 0.67*dlogFdc -"
                " 1.27*dlogFdn) + dlogFdlogPr * dlogPrdT;",
            )
        else:
            cw.writer(fstream, cw.comment("Lindemann form"))
            cw.writer(fstream, "F = 1.0;")
            if precond:
                cw.writer(fstream, "// dlogFdlogPr is 0.0 and unused")
            else:
                cw.writer(fstream, "dlogFdlogPr = 0.0;")
            cw.writer(fstream, "dlogFdT = 0.0;")

    # reverse
    if not reaction.reversible:
        cw.writer(fstream, cw.comment("rate of progress"))
        if (not has_alpha) and (not falloff):
            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(fstream, cw.comment("q = k_f*phi_f;"))
                cw.writer(fstream, "q = 0.0;")
            else:
                cw.writer(fstream, "q = k_f*phi_f;")
        else:
            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(fstream, cw.comment("q_nocor = k_f*phi_f;"))
                cw.writer(fstream, "q_nocor = 0.0;")
            else:
                cw.writer(fstream, "q_nocor = k_f*phi_f;")

            if falloff:
                cw.writer(fstream, "Corr = fPr * F;")
                cw.writer(fstream, "q = Corr * q_nocor;")
            else:
                cw.writer(fstream, "q = alpha * q_nocor;")

        if falloff:
            cw.writer(fstream, "dlnCorrdT = ln10*(dlogfPrdT + dlogFdT);")
            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(
                    fstream,
                    cw.comment(
                        f"dqdT = {corr_s}dlnkfdT*k_f*phi_f + dlnCorrdT*q;",
                    ),
                )
                cw.writer(fstream, "dqdT = dlnCorrdT*q;")
            else:
                cw.writer(
                    fstream,
                    f"dqdT = {corr_s}dlnkfdT*k_f*phi_f + dlnCorrdT*q;",
                )
        else:
            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(fstream, cw.comment(f"dqdT = {corr_s}dlnkfdT*k_f*phi_f;"))
                cw.writer(fstream, "dqdT = 0;")
            else:
                cw.writer(fstream, f"dqdT = {corr_s}dlnkfdT*k_f*phi_f;")
    else:
        cw.writer(fstream, cw.comment("reverse"))
        qss_ps = cu.qss_sorted_phase_space(
            mechanism, species_info, reaction, reaction.products
        )
        cw.writer(
            fstream,
            f"phi_r = {qss_ps};",
        )
        cw.writer(
            fstream,
            f"Kc = {cu.sorted_kc(mechanism, species_info, reaction)};",
        )
        cw.writer(fstream, "k_r = k_f / Kc;")

        dlnkcdt_s = "invT * ("
        terms = []
        for symbol, coefficient in sorted(
            sorted_reactants, key=lambda v: species_info.dict_species[v[0]]
        ):
            k = species_info.ordered_idx_map[symbol]
            if symbol not in species_info.qssa_species_list:
                if coefficient == 1.0:
                    terms.append(f"h_RT[{k}]")
                else:
                    terms.append(f"{coefficient:f}*h_RT[{k}]")
            else:
                if coefficient == 1.0:
                    terms.append(f"h_RT_qss[{k - n_species}]")
                else:
                    terms.append(f"{coefficient:f}*h_RT_qss[{k - n_species}]")
        dlnkcdt_s += "-(" + " + ".join(terms) + ")"
        terms = []
        for symbol, coefficient in sorted(
            sorted_products, key=lambda v: species_info.dict_species[v[0]]
        ):
            k = species_info.ordered_idx_map[symbol]
            if symbol not in species_info.qssa_species_list:
                if coefficient == 1.0:
                    terms.append(f"h_RT[{k}]")
                else:
                    terms.append(f"{coefficient:f}*h_RT[{k}]")
            else:
                if coefficient == 1.0:
                    terms.append(f"h_RT_qss[{k - n_species}]")
                else:
                    terms.append(f"{coefficient:f}*h_RT_qss[{k - n_species}]")
        dlnkcdt_s += " + (" + " + ".join(terms) + ")"
        if sum_nuk > 0:
            dlnkcdt_s += f" - {sum_nuk:f}"
        elif sum_nuk < 0:
            dlnkcdt_s += f" + {-sum_nuk:f}"
        dlnkcdt_s += ")"
        cw.writer(fstream, f"dlnKcdT = {dlnkcdt_s};")

        cw.writer(fstream, "dkrdT = (dlnkfdT - dlnKcdT)*k_r;")

        cw.writer(fstream, cw.comment("rate of progress"))
        if (not has_alpha) and (not falloff):
            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(fstream, cw.comment("q = k_f*phi_f - k_r*phi_r;"))
                cw.writer(fstream, "q = - k_r*phi_r;")
            else:
                cw.writer(fstream, "q = k_f*phi_f - k_r*phi_r;")
        else:
            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(fstream, cw.comment("q_nocor = k_f*phi_f - k_r*phi_r;"))
                cw.writer(fstream, "q_nocor = - k_r*phi_r;")
            else:
                cw.writer(fstream, "q_nocor = k_f*phi_f - k_r*phi_r;")
            if falloff:
                cw.writer(fstream, "Corr = fPr * F;")
                cw.writer(fstream, "q = Corr * q_nocor;")
            else:
                cw.writer(fstream, "q = alpha * q_nocor;")

        if falloff:
            cw.writer(fstream, "dlnCorrdT = ln10*(dlogfPrdT + dlogFdT);")
            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(
                    fstream,
                    cw.comment(
                        f"dqdT = {corr_s}(dlnkfdT*k_f*phi_f - dkrdT*phi_r)"
                        " + dlnCorrdT*q;",
                    ),
                )
                cw.writer(fstream, f"dqdT = {corr_s}(- dkrdT*phi_r) + dlnCorrdT*q;")
            else:
                cw.writer(
                    fstream,
                    f"dqdT = {corr_s}(dlnkfdT*k_f*phi_f - dkrdT*phi_r) + dlnCorrdT*q;",
                )
        else:
            if remove_forward:
                cw.writer(fstream, cw.comment("Remove forward reaction"))
                cw.writer(
                    fstream,
                    cw.comment(
                        f"dqdT = {corr_s}(dlnkfdT*k_f*phi_f - dkrdT*phi_r);",
                    ),
                )
                cw.writer(fstream, f"dqdT = {corr_s}( - dkrdT*phi_r);")
            else:
                cw.writer(
                    fstream,
                    f"dqdT = {corr_s}(dlnkfdT*k_f*phi_f - dkrdT*phi_r);",
                )

    cw.writer(fstream, cw.comment("update wdot"))
    for k in sorted(all_dict.keys()):
        s, nu = all_dict[k]
        if nu == 1:
            cw.writer(fstream, f"wdot[{k}] += q;" + cw.comment(f"{s}"))
        elif nu == -1:
            cw.writer(fstream, f"wdot[{k}] -= q;" + cw.comment(f"{s}"))
        elif nu > 0:
            cw.writer(
                fstream,
                f"wdot[{k}] += {nu:.15g} * q;" + cw.comment(f"{s}"),
            )
        elif nu < 0:
            cw.writer(
                fstream,
                f"wdot[{k}] -= {-nu:.15g} * q;" + cw.comment(f"{s}"),
            )

    if falloff:
        cw.writer(fstream, cw.comment("for convenience"))
        cw.writer(fstream, "k_f *= Corr;")
        if reaction.reversible:
            cw.writer(fstream, "k_r *= Corr;")
    elif has_alpha:
        cw.writer(fstream, cw.comment("for convenience"))
        cw.writer(fstream, "k_f *= alpha;")
        if reaction.reversible:
            cw.writer(fstream, "k_r *= alpha;")

    if falloff:
        if precond:
            cw.writer(fstream, "dcdc_fac = 0.0;")
        else:
            cw.writer(fstream, "dcdc_fac = q/alpha*(1.0/(Pr+1.0) + dlogFdlogPr);")
    # elif has_alpha:
    #    cw.writer(fstream,'dcdc_fac = q_nocor;')

    if has_alpha or falloff:
        if not precond:
            cw.writer(fstream, "if (consP == 1) {")

            for k in range(n_species):
                dqdc_s = denhancement_d(mechanism, species_info, reaction, k, True)
                if dqdc_s != "0":
                    if falloff:
                        if dqdc_s == "1":
                            dqdc_s = "dcdc_fac"
                        else:
                            dqdc_s += "*dcdc_fac"
                    elif has_alpha:
                        if dqdc_s == "1":
                            dqdc_s = "q_nocor"
                        else:
                            dqdc_s += "*q_nocor"

                dqdc_s = dqdc_d(
                    fstream,
                    mechanism,
                    species_info,
                    reaction,
                    sorted_reactants,
                    sorted_products,
                    rea_dict,
                    pro_dict,
                    dqdc_s,
                    k,
                    remove_forward,
                    reaction.orders,
                    syms,
                )
                if dqdc_s:
                    symb_k = species_info.all_species_list[k]
                    cw.writer(fstream, cw.comment(f"d()/d[{symb_k}]"))
                    cw.writer(fstream, f"dqdci = {dqdc_s};")
                    #
                    for m in sorted(all_dict.keys()):
                        if all_dict[m][1] != 0:
                            s1 = (
                                f"J[{k * (n_species + 1) + m}] +="
                                f" {all_dict[m][1]:.15g} * dqdci;"
                            )
                            s1 = s1.replace("+= 1 *", "+=").replace("+= -1 *", "-=")
                            s2 = cw.comment(f"dwdot[{all_dict[m][0]}]/d[{symb_k}]")
                            cw.writer(fstream, s1.ljust(30) + s2)

            cw.writer(fstream, "}")
            cw.writer(fstream, "else {")

        for k in range(n_species):
            # for k in range(len(species_info.all_species_list)):
            dqdc_s = denhancement_d(mechanism, species_info, reaction, k, False)
            if dqdc_s != "0":
                if falloff:
                    if dqdc_s == "1":
                        dqdc_s = "dcdc_fac"
                    else:
                        dqdc_s += "*dcdc_fac"
                elif has_alpha:
                    if dqdc_s == "1":
                        dqdc_s = "q_nocor"
                    else:
                        dqdc_s += "*q_nocor"

            dqdc_s = dqdc_d(
                fstream,
                mechanism,
                species_info,
                reaction,
                sorted_reactants,
                sorted_products,
                rea_dict,
                pro_dict,
                dqdc_s,
                k,
                remove_forward,
                reaction.orders,
                syms,
            )
            if dqdc_s:
                cw.writer(fstream, f"dqdc[{k}] = {dqdc_s};")
            elif precond:
                cw.writer(fstream, f"dqdc[{k}] = 0.0;")

        cw.writer(fstream, f"for (int k=0; k<{n_species}; k++) {{")
        for m in sorted(all_dict.keys()):
            if all_dict[m][1] != 0:
                s1 = f"J[{n_species + 1}*k+{m}] += {all_dict[m][1]:.15g} * dqdc[k];"
                s1 = s1.replace("+= 1 *", "+=").replace("+= -1 *", "-=")
                cw.writer(fstream, s1)
        cw.writer(fstream, "}")

        if not precond:
            cw.writer(fstream, "}")

        for m in sorted(all_dict.keys()):
            if all_dict[m][1] != 0:
                s1 = (
                    f"J[{n_species * (n_species + 1) + m}] +="
                    f" {all_dict[m][1]:.15g} * dqdT;"
                    + cw.comment(f"dwdot[{all_dict[m][0]}]/dT")
                )
                s1 = s1.replace("+= 1 *", "+=").replace("+= -1 *", "-=")
                cw.writer(fstream, s1)

    else:
        for k in range(n_species):
            dqdc_s = dqdc_d(
                fstream,
                mechanism,
                species_info,
                reaction,
                sorted_reactants,
                sorted_products,
                rea_dict,
                pro_dict,
                "",
                k,
                remove_forward,
                reaction.orders,
                syms,
            )
            if dqdc_s:
                cw.writer(fstream, cw.comment(f"d()/d[{all_wqss_dict[k][0]}]"))
                cw.writer(fstream, f"dqdci = {dqdc_s};")
                if (
                    reaction.reversible
                    or k in rea_dict
                    or species_info.all_species_list[k] in reaction.orders
                ):
                    for m in sorted(all_dict.keys()):
                        if all_dict[m][1] != 0:
                            s1 = (
                                f"J[{k * (n_species + 1) + m}] +="
                                f" {all_wqss_dict[m][1]:.15g} * dqdci;"
                            )
                            s1 = s1.replace("+= 1 *", "+=").replace("+= -1 *", "-=")
                            s2 = cw.comment(
                                f"dwdot[{all_wqss_dict[m][0]}]/d[{all_wqss_dict[k][0]}]"
                            )
                            cw.writer(fstream, s1.ljust(30) + s2)
        cw.writer(fstream, cw.comment("d()/dT"))
        for m in sorted(all_dict.keys()):
            if all_dict[m][1] != 0:
                s1 = (
                    f"J[{n_species * (n_species + 1) + m}] +="
                    f" {all_dict[m][1]:.15g} * dqdT;"
                )
                s1 = (
                    s1.replace("+= 1 *", "+=")
                    .replace("+= -1 *", "-=")
                    .replace("+= -1 *", "-=")
                )
                s2 = cw.comment(f"dwdot[{all_dict[m][0]}]/dT")
                cw.writer(fstream, s1.ljust(30) + s2)


def dqdc_d(
    fstream,
    mechanism,
    species_info,
    reaction,
    sorted_reactants,
    sorted_products,
    rea_dict,
    pro_dict,
    dqdc_s,
    k,
    remove_forward,
    reaction_orders,
    syms,
):
    """Write dqdc."""
    if dqdc_s == "0":
        dqdc_s = ""
    if (
        k in sorted(rea_dict.keys())
        or species_info.all_species_list[k] in reaction.orders
    ):
        dps = dphase_space(
            mechanism,
            species_info,
            sorted_reactants,
            species_info.all_species_list[k],
            reaction_orders,
            syms,
        )
        if dps == "1.0":
            dps_s = ""
        else:
            dps_s = "*" + dps
        if remove_forward:
            cw.writer(fstream, cw.comment("Remove forward reaction"))
            dqdc_s += ""
        else:
            dqdc_s += f" + k_f{dps_s}"
    if reaction.reversible:
        if k in sorted(pro_dict.keys()):
            dps = dphase_space(
                mechanism,
                species_info,
                sorted_products,
                pro_dict[k][0],
                reaction_orders,
                syms,
            )
            if dps == "1.0":
                dps_s = ""
            else:
                dps_s = "*" + dps
            dqdc_s += f" - k_r{dps_s}"
    return dqdc_s


def denhancement_d(mechanism, species_info, reaction, kid, cons_p):
    """Get enhancement gradient."""
    third_body = reaction.third_body is not None
    falloff = reaction.rate.type == "falloff"
    if not third_body and not falloff:
        raise ValueError("denhancement_d called for a reaction without a third body")

    if not reaction.third_body:
        raise NotImplementedError("FIXME")
        species, coefficient = third_body
        if species == "<mixture>":
            if cons_p:
                return "0"
            else:
                return "1"
        elif species_info.ordered_idx_map[species] == kid:
            return "1"
        else:
            return "0"
    else:
        efficiencies = reaction.third_body.efficiencies
        if (
            falloff
            and len(reaction.third_body.efficiencies) == 1
            and isclose(reaction.third_body.default_efficiency, 0.0)
        ):
            # Revise the efficiencies for Jacobian
            for symbol in species_info.all_species_list:
                if symbol not in efficiencies:
                    efficiencies[symbol] = 0.0
        if cons_p:
            for _, (symbol, efficiency) in enumerate(efficiencies.items()):
                if species_info.ordered_idx_map[symbol] == kid:
                    return f"({efficiency:.15g} - 1)"
            return "0"
        else:
            for _, (symbol, efficiency) in enumerate(efficiencies.items()):
                if species_info.ordered_idx_map[symbol] == kid:
                    return f"{efficiency:.15g}"
            return "1"


def dphase_space(mechanism, species_info, reagents, r, reaction_orders, syms):
    """Get string for phase space gradient."""
    reagents = {x[0]: x[1] for x in reagents}
    if bool(reaction_orders):
        list_ord = list(reaction_orders.keys())
        for spec in list_ord:
            if spec not in reagents:
                reagents[spec] = 0.0
    phi = []
    sorted_reagents = sorted(
        reagents.keys(), key=lambda v: species_info.dict_species[v]
    )

    for symbol in sorted_reagents:
        coefficient = reagents[symbol]
        if bool(reaction_orders):
            order = reaction_orders[symbol]
        else:
            order = coefficient
        if symbol not in species_info.qssa_species_list:
            if symbol == r:
                if order > 1:
                    phi += [f"{order:f}"]
                    if (order - 1) == 1.0:
                        conc = f"sc[{species_info.ordered_idx_map[symbol]}]"
                    else:
                        exponent = order - 1
                        if exponent.is_integer():
                            conc = "*".join(
                                [f"sc[{species_info.ordered_idx_map[symbol]}]"]
                                * int(exponent)
                            )
                        elif exponent == 0.5:
                            idx = species_info.ordered_idx_map[symbol]
                            conc = (
                                f"std::sqrt(std::max(sc[{idx}], {cu.sc_cutoff(0.5)}))"
                            )
                        else:
                            idx = species_info.ordered_idx_map[symbol]
                            conc = (
                                f"pow(std::max(sc[{idx}],"
                                f" {cu.sc_cutoff(exponent)}),{exponent:f})"
                            )
                    phi += [conc]
                elif order < 1:
                    phi += [f"{order:f}"]
                    exponent = order - 1.0
                    idx = species_info.ordered_idx_map[symbol]
                    conc = (
                        f"pow(std::max(sc[{idx}],"
                        f" {cu.sc_cutoff(exponent)}),{exponent:f})"
                    )
                    phi += [conc]
            else:
                if order == 1.0:
                    conc = f"sc[{species_info.ordered_idx_map[symbol]}]"
                else:
                    if order.is_integer():
                        conc = "*".join(
                            [f"sc[{species_info.ordered_idx_map[symbol]}]"] * int(order)
                        )
                    elif order == 0.5:
                        conc = (
                            f"std::sqrt(std::max(sc[{species_info.ordered_idx_map[symbol]}],"
                            f" {cu.sc_cutoff(0.5)}))"
                        )
                    else:
                        conc = (
                            f"pow(std::max(sc[{species_info.ordered_idx_map[symbol]}],"
                            f" {cu.sc_cutoff(order)}), {order:f})"
                        )
                phi += [conc]
        # Symbol is in qssa_species_list
        else:
            if symbol == r:
                if order > 1:
                    phi += [f"{order:f}"]
                    if (order - 1) == 1.0:
                        idx = (
                            species_info.ordered_idx_map[symbol]
                            - species_info.n_species
                        )
                        conc = f"sc_qss[{idx}]"
                    else:
                        exponent = order - 1
                        if exponent.is_integer():
                            conc = "*".join(
                                [f"sc[{species_info.ordered_idx_map[symbol]}]"]
                                * int(exponent)
                            )
                        else:
                            idx = (
                                species_info.ordered_idx_map[symbol]
                                - species_info.n_species
                            )
                            conc = f"pow(sc_qss[{idx}],{exponent:f})"
                    phi += [conc]
            else:
                if order == 1.0:
                    idx = species_info.ordered_idx_map[symbol] - species_info.n_species
                    conc = f"sc_qss[{idx}]"
                else:
                    if order.is_integer():
                        conc = "*".join(
                            [f"sc[{species_info.ordered_idx_map[symbol]}]"] * int(order)
                        )
                    else:
                        idx = (
                            species_info.ordered_idx_map[symbol]
                            - species_info.n_species
                        )
                        conc = f"pow(sc_qss[{idx}], {order:f})"
                phi += [conc]

    if phi:
        return "*".join(phi)
    else:
        return "1.0"


def dproduction_rate(fstream, mechanism, species_info, reaction_info, precond=False):
    """Write the reaction jacobian."""
    n_species = species_info.n_species

    cw.writer(fstream)
    if precond:
        cw.writer(
            fstream,
            cw.comment(
                "compute an approx to the reaction Jacobian (for preconditioning)"
            ),
        )
        cw.writer(
            fstream,
            "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void"
            " DWDOT_SIMPLIFIED(amrex::Real *  J, const amrex::Real *  sc,"
            " const amrex::Real *  Tp, const int * HP)",
        )
    else:
        cw.writer(fstream, cw.comment("compute the reaction Jacobian"))
        cw.writer(
            fstream,
            "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void"
            " DWDOT(amrex::Real *  J, const amrex::Real *  sc, const"
            " amrex::Real *  Tp, const int * consP)",
        )

    cw.writer(fstream, "{")
    cw.writer(fstream, f"amrex::Real c[{n_species}];")
    cw.writer(fstream)
    cw.writer(fstream, f"for (int k=0; k<{n_species}; k++) {{")
    cw.writer(fstream, "c[k] = 1.e6 * sc[k];")
    cw.writer(fstream, "}")

    cw.writer(fstream)
    if precond:
        cw.writer(fstream, "aJacobian_precond(J, c, *Tp, *HP);")
    else:
        cw.writer(fstream, "aJacobian(J, c, *Tp, *consP);")

    cw.writer(fstream)
    cw.writer(fstream, cw.comment("dwdot[k]/dT"))
    cw.writer(fstream, cw.comment("dTdot/d[X]"))
    cw.writer(fstream, f"for (int k=0; k<{n_species}; k++) {{")
    cw.writer(fstream, f"J[{n_species * (n_species + 1)}+k] *= 1.e-6;")
    cw.writer(fstream, f"J[k*{n_species + 1}+{n_species}] *= 1.e6;")
    cw.writer(fstream, "}")

    cw.writer(fstream)
    cw.writer(fstream, "}")
    cw.writer(fstream)
