"""Thermodynamics functions."""

import bisect
import itertools

from cantera.speciesthermo import Nasa9PolyMultiTempRegion, NasaPoly2
import numpy as np

import ceptr.writer as cw


def thermo(fstream, mechanism, species_info, syms=None):
    """Write thermodynamics routines."""
    models = analyze_thermodynamics(mechanism, species_info.nonqssa_species_list)
    if species_info.n_qssa_species > 0:
        qss_models = analyze_thermodynamics(mechanism, species_info.qssa_species_list)

    cv(fstream, species_info, models)
    cp(fstream, species_info, models)
    gibbs(fstream, species_info, models, 0, syms)
    if species_info.n_qssa_species > 0:
        gibbs(fstream, species_info, qss_models, 1, syms)
    helmholtz(fstream, species_info, models)
    species_internal_energy(fstream, species_info, models)
    species_enthalpy(fstream, species_info, models, 0, syms)
    if species_info.n_qssa_species > 0:
        species_enthalpy(fstream, species_info, qss_models, 1, syms)
    species_entropy(fstream, species_info, models)
    dcvpdtemp(fstream, species_info, models)


def model_type(model):
    """Return string for the model type."""
    if isinstance(model, NasaPoly2):
        assert model.n_coeffs == 15, "Unexpected coefficient form."
        return "nasa7"
    elif isinstance(model, Nasa9PolyMultiTempRegion):
        return "nasa9"
    else:
        raise TypeError(
            f"Model {model.__class__} is not supported. CEPTR supports NASA7 and NASA9"
            " models."
        )


def expression_map_nasa7(name):
    """Return the function generator for NASA7 polynomials."""
    dct = {
        "cv_R": cv_nasa7,
        "cp_R": cp_nasa7,
        "gibbs": gibbs_nasa7,
        "gibbs_qss": gibbs_nasa7,
        "helmholtz": helmholtz_nasa7,
        "speciesInternalEnergy": internal_energy_nasa7,
        "speciesEnthalpy": enthalpy_nasa7,
        "speciesEnthalpy_qss": enthalpy_nasa7,
        "speciesEntropy": entropy_nasa7,
        "dcvpRdT": dcpdtemp_nasa7,
    }
    return dct[name]


def expression_map_nasa9(name):
    """Return the function generator for NASA7 polynomials."""
    dct = {
        "cv_R": cv_nasa9,
        "cp_R": cp_nasa9,
        "gibbs": gibbs_nasa9,
        "gibbs_qss": gibbs_nasa9,
        "helmholtz": helmholtz_nasa9,
        "speciesInternalEnergy": internal_energy_nasa9,
        "speciesEnthalpy": enthalpy_nasa9,
        "speciesEnthalpy_qss": enthalpy_nasa9,
        "speciesEntropy": entropy_nasa9,
        "dcvpRdT": dcpdtemp_nasa9,
    }
    return dct[name]


def variables_nasa7(name):
    """Return the variables needed for NASA7 polynomials."""
    dct = {
        "cv_R": {
            "T4": True,
            "inv_temp": False,
            "inv_temp2": False,
            "inv_temp3": False,
            "log_temp": False,
        },
        "cp_R": {
            "T4": True,
            "inv_temp": False,
            "inv_temp2": False,
            "inv_temp3": False,
            "log_temp": False,
        },
        "gibbs": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": False,
            "inv_temp3": False,
            "log_temp": True,
        },
        "gibbs_qss": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": False,
            "inv_temp3": False,
            "log_temp": True,
        },
        "helmholtz": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": False,
            "inv_temp3": False,
            "log_temp": True,
        },
        "speciesInternalEnergy": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": False,
            "inv_temp3": False,
            "log_temp": False,
        },
        "speciesEnthalpy": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": False,
            "inv_temp3": False,
            "log_temp": False,
        },
        "speciesEnthalpy_qss": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": False,
            "inv_temp3": False,
            "log_temp": False,
        },
        "speciesEntropy": {
            "T4": True,
            "inv_temp": False,
            "inv_temp2": False,
            "inv_temp3": False,
            "log_temp": True,
        },
        "dcvpRdT": {
            "T4": False,
            "inv_temp": False,
            "inv_temp2": False,
            "inv_temp3": False,
            "log_temp": False,
        },
    }
    return dct[name]


def variables_nasa9(name):
    """Return the variables needed for NASA9 polynomials."""
    dct = {
        "cv_R": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": True,
            "inv_temp3": False,
            "log_temp": False,
        },
        "cp_R": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": True,
            "inv_temp3": False,
            "log_temp": False,
        },
        "gibbs": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": True,
            "inv_temp3": False,
            "log_temp": True,
        },
        "gibbs_qss": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": True,
            "inv_temp3": False,
            "log_temp": True,
        },
        "helmholtz": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": True,
            "inv_temp3": False,
            "log_temp": True,
        },
        "speciesInternalEnergy": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": True,
            "inv_temp3": False,
            "log_temp": True,
        },
        "speciesEnthalpy": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": True,
            "inv_temp3": False,
            "log_temp": True,
        },
        "speciesEnthalpy_qss": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": True,
            "inv_temp3": False,
            "log_temp": True,
        },
        "speciesEntropy": {
            "T4": True,
            "inv_temp": True,
            "inv_temp2": True,
            "inv_temp3": False,
            "log_temp": True,
        },
        "dcvpRdT": {
            "T4": False,
            "inv_temp": True,
            "inv_temp2": True,
            "inv_temp3": True,
            "log_temp": False,
        },
    }
    return dct[name]


def analyze_thermodynamics(mechanism, species_list):
    """Extract information from the thermodynamics model."""
    models = []
    for symbol in species_list:
        species = mechanism.species(symbol)
        model = species.thermo
        mtype = model_type(model)
        dct = {"species": species, "interval": [], "coefficients": [], "type": mtype}

        if mtype == "nasa7":
            dct["coefficients"] = [model.coeffs[1:8]]
            if not np.allclose(model.coeffs[1:8], model.coeffs[8:15], atol=1e-22):
                dct["interval"] = [model.coeffs[0]]
                dct["coefficients"] = [model.coeffs[8:15], model.coeffs[1:8]]
        elif mtype == "nasa9":
            nzones = int(model.coeffs[0])
            dct["coefficients"] = [
                model.coeffs[3 + 11 * x : 12 + 11 * x] for x in range(nzones)
            ]
            for c in dct["coefficients"]:
                assert len(c) == 9, "NASA9 polynomial coefficients must be of length 9."
            min_bounds = [model.coeffs[1 + 11 * x] for x in range(nzones)]
            max_bounds = [model.coeffs[2 + 11 * x] for x in range(nzones)]
            dct["interval"] = sorted(list(set(min_bounds) & set(max_bounds)))

        models.append(dct)

    return models


def generate_thermo_routine(
    fstream,
    species_info,
    name,
    models,
    qss_flag,
    syms=None,
    inline=False,
):
    """Write a thermodynamics routine."""
    if not inline:
        cw.writer(
            fstream,
            f"AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void {name}(amrex::Real"
            " * species, const amrex::Real T)",
        )

    if not inline:
        cw.writer(fstream, "{")

    variables = {
        "T4": False,
        "inv_temp": False,
        "inv_temp2": False,
        "inv_temp3": False,
        "log_temp": False,
    }
    for model in models:
        if model["type"] == "nasa7":
            mvars = variables_nasa7(name)
        elif model["type"] == "nasa9":
            mvars = variables_nasa9(name)

        for k, v in mvars.items():
            variables[k] = v if v else variables[k]

    cw.writer(fstream, "const amrex::Real T2 = T*T;")
    cw.writer(fstream, "const amrex::Real T3 = T*T2;")
    if variables["T4"]:
        cw.writer(fstream, "const amrex::Real T4 = T*T3;")
    if variables["inv_temp"]:
        cw.writer(fstream, "const amrex::Real invT = 1.0 / T;")
    if variables["inv_temp2"]:
        cw.writer(fstream, "const amrex::Real invT2 = invT*invT;")
    if variables["inv_temp3"]:
        cw.writer(fstream, "const amrex::Real invT3 = invT*invT*invT;")
    if variables["log_temp"]:
        cw.writer(fstream, "const amrex::Real logT = log(T);")
    cw.writer(fstream)

    intervals = sorted([x["interval"] for x in models])
    intervals = list(intervals for intervals, _ in itertools.groupby(intervals))

    for interval in intervals:
        cw.writer(fstream)
        if len(interval) == 0:
            cw.writer(
                fstream,
                cw.comment("species with no change across T"),
            )
        elif len(interval) == 1:
            cw.writer(
                fstream,
                cw.comment(f"species with midpoint at T={interval[0]:g} kelvin"),
            )
        elif len(interval) > 1:
            cw.writer(
                fstream,
                cw.comment(
                    f"species with inflection points at T = {(*interval,)} kelvin"
                ),
            )
        for k in range(len(interval) + 1):
            if len(interval) == 1:
                if k == 0:
                    cw.writer(
                        fstream,
                        f"""if (T < {interval[0]:g}) {{""",
                    )
                else:
                    cw.writer(
                        fstream,
                        "else {",
                    )
            elif len(interval) > 1:
                if k == 0:
                    cw.writer(
                        fstream,
                        f"""if (T < {interval[0]:g}) {{""",
                    )
                elif 0 < k and k < len(interval):
                    cw.writer(
                        fstream,
                        f"""else if ( ({interval[k - 1]:g} <= T) && (T < {interval[k]:g})) {{""",
                    )
                else:
                    cw.writer(
                        fstream,
                        "else {",
                    )

            for model in [x for x in models if x["interval"] == interval]:
                species = model["species"]
                if model["type"] == "nasa7":
                    expression_generator = expression_map_nasa7(name)
                elif model["type"] == "nasa9":
                    expression_generator = expression_map_nasa9(name)

                index = (
                    species_info.ordered_idx_map[species.name] - species_info.n_species
                    if qss_flag
                    else species_info.ordered_idx_map[species.name]
                )
                cw.writer(fstream, cw.comment(f"species {index}: {species.name}"))
                cw.writer(
                    fstream,
                    (f"result += y[{index}] * (" if inline else f"species[{index}] ="),
                )
                if syms:
                    models_smp = (
                        syms.models_smp_tmp if not qss_flag else syms.models_qss_smp_tmp
                    )
                    model_smp = [
                        x for x in models_smp if x["species"].name == species.name
                    ][0]
                    model_smp[name][k] = expression_generator(
                        fstream, model["coefficients"][k], syms
                    )
                else:
                    expression_generator(fstream, model["coefficients"][k])
                if inline:
                    spec_idx = species_info.ordered_idx_map[species.name]
                    sp = species_info.nonqssa_species[spec_idx]
                    imw = 1.0 / sp.weight
                    cw.writer(fstream, f")* {imw:.16f}")
                cw.writer(fstream, ";")

            if len(interval) >= 1:
                cw.writer(
                    fstream,
                    "}",
                )

    if not inline:
        cw.writer(fstream, "}")


def cv(fstream, species_info, models):
    """Write cv."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("compute Cv/R at the given temperature"))
    generate_thermo_routine(fstream, species_info, "cv_R", models, 0)


def cp(fstream, species_info, models):
    """Write cp."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("compute Cp/R at the given temperature"))
    generate_thermo_routine(fstream, species_info, "cp_R", models, 0)


def gibbs(fstream, species_info, models, qss_flag, syms=None):
    """Write Gibbs."""
    name = "gibbs_qss" if qss_flag else "gibbs"
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("compute the g/(RT) at the given temperature"))
    generate_thermo_routine(fstream, species_info, name, models, qss_flag, syms)


def helmholtz(fstream, species_info, models):
    """Write Helmholtz."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("compute the a/(RT) at the given temperature"))
    generate_thermo_routine(
        fstream,
        species_info,
        "helmholtz",
        models,
        0,
    )


def species_internal_energy(fstream, species_info, models):
    """Write species internal energy."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("compute the e/(RT) at the given temperature"))
    generate_thermo_routine(
        fstream,
        species_info,
        "speciesInternalEnergy",
        models,
        0,
    )


def species_enthalpy(fstream, species_info, models, qss_flag, syms=None):
    """Write species enthalpy."""
    name = "speciesEnthalpy_qss" if qss_flag else "speciesEnthalpy"
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("compute the h/(RT) at the given temperature (Eq 20)"),
    )

    generate_thermo_routine(
        fstream,
        species_info,
        name,
        models,
        qss_flag,
        syms,
    )


def species_entropy(fstream, species_info, models):
    """Write species entropy."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("compute the S/R at the given temperature (Eq 21)"))
    generate_thermo_routine(
        fstream,
        species_info,
        "speciesEntropy",
        models,
        0,
    )


def dcvpdtemp(fstream, species_info, models):
    """Write gradient of cp/cv wrt temperature."""
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature"),
    )
    generate_thermo_routine(fstream, species_info, "dcvpRdT", models, 0)


def param2str(param, sfx="", fmt="+20.15e"):
    """Convert a parameter to a string."""
    return f"{param:{fmt}} {sfx}" if param != 0.0 else ""


def eval_cv_species(mechanism, species, temp):
    """Evaluate cv."""
    model = mechanism.species(species.name).thermo
    mtype = model_type(model)
    if mtype == "nasa7":
        return eval_cv_species_nasa7(model, temp)
    elif mtype == "nasa9":
        return eval_cv_species_nasa9(model, temp)


def cv_nasa7(fstream, parameters):
    """Write NASA7 polynomial for cv."""
    expression = (
        param2str(parameters[0] - 1.0)
        + param2str(parameters[1], "* T")
        + param2str(parameters[2], " * T2")
        + param2str(parameters[3], "* T3")
        + param2str(parameters[4], "* T4")
    )
    cw.writer(fstream, expression if expression else "0.0")


def eval_cv_species_nasa7(model, temp):
    """Evaluate cv with NASA7 polynomial."""
    assert model.n_coeffs == 15, "Unsupported thermo model."
    mid = model.coeffs[0]
    high_range = model.coeffs[1:8]
    low_range = model.coeffs[8:15]

    if temp < mid:
        parameters = low_range
    else:
        parameters = high_range

    return (
        (parameters[0] - 1.0)
        + parameters[1] * temp
        + parameters[2] * temp * temp
        + parameters[3] * temp * temp * temp
        + parameters[4] * temp * temp * temp * temp
    )


def cp_nasa7(fstream, parameters):
    """Write NASA7 polynomial for cp."""
    expression = (
        param2str(parameters[0])
        + param2str(parameters[1], "* T")
        + param2str(parameters[2], "* T2")
        + param2str(parameters[3], "* T3")
        + param2str(parameters[4], "* T4")
    )
    cw.writer(fstream, expression if expression else "0.0")


def dcpdtemp_nasa7(fstream, parameters):
    """Write NASA7 polynomial for dcpdtemp."""
    expression = (
        param2str(parameters[1])
        + param2str(parameters[2] * 2.0, "* T")
        + param2str(parameters[3] * 3.0, "* T2")
        + param2str(parameters[4] * 4.0, "* T3")
    )
    cw.writer(fstream, expression if expression else "0.0")


def gibbs_nasa7(fstream, parameters, syms=None):
    """Write NASA7 polynomial for Gibbs."""
    expression = (
        param2str(parameters[5], "* invT")
        + param2str(parameters[0] - parameters[6], "")
        + param2str(-parameters[0], "* logT")
        + param2str((-parameters[1] / 2), "* T")
        + param2str((-parameters[2] / 6), "* T2")
        + param2str((-parameters[3] / 12), "* T3")
        + param2str((-parameters[4] / 20), "* T4")
    )
    cw.writer(fstream, expression if expression else "0.0")

    if syms:
        return (
            parameters[5] * syms.invT_smp
            + parameters[0]
            - parameters[6]
            + (-parameters[0]) * syms.logT_smp
            + (-parameters[1] / 2) * syms.T_smp
            + (-parameters[2] / 6) * syms.T2_smp
            + (-parameters[3] / 12) * syms.T3_smp
            + (-parameters[4] / 20) * syms.T4_smp
        )


def helmholtz_nasa7(fstream, parameters, syms=None):
    """Write NASA7 polynomial for Helmholtz."""
    expression = (
        param2str(parameters[5], "* invT")
        + param2str(parameters[0] - parameters[6] - 1.0, "")
        + param2str(-parameters[0], "* logT")
        + param2str((-parameters[1] / 2), "* T")
        + param2str((-parameters[2] / 6), "* T2")
        + param2str((-parameters[3] / 12), "* T3")
        + param2str((-parameters[4] / 20), "* T4")
    )
    cw.writer(fstream, expression if expression else "0.0")

    if syms:
        return (
            parameters[5] * syms.invT_smp
            + parameters[0]
            - parameters[6]
            - 1.0
            + (-parameters[0]) * syms.logT_smp
            + (-parameters[1] / 2) * syms.T_smp
            + (-parameters[2] / 6) * syms.T2_smp
            + (-parameters[3] / 12) * syms.T3_smp
            + (-parameters[4] / 20) * syms.T4_smp
        )


def internal_energy_nasa7(fstream, parameters, syms=None):
    """Write NASA7 polynomial for internal energy."""
    expression = (
        param2str(parameters[0] - 1.0, "")
        + param2str(parameters[1] / 2, "* T")
        + param2str(parameters[2] / 3, "* T2")
        + param2str(parameters[3] / 4, "* T3")
        + param2str(parameters[4] / 5, "* T4")
        + param2str(parameters[5], "* invT")
    )
    cw.writer(fstream, expression if expression else "0.0")

    if syms:
        return (
            parameters[0]
            - 1.0
            + (parameters[1] / 2) * syms.T_smp
            + (parameters[2] / 3) * syms.T2_smp
            + (parameters[3] / 4) * syms.T3_smp
            + (parameters[4] / 5) * syms.T4_smp
            + (parameters[5]) * syms.invT_smp
        )


def enthalpy_nasa7(fstream, parameters, syms=None):
    """Write NASA7 polynomial for enthalpy."""
    expression = (
        param2str(parameters[0], "")
        + param2str(parameters[1] / 2, "* T")
        + param2str(parameters[2] / 3, "* T2")
        + param2str(parameters[3] / 4, "* T3")
        + param2str(parameters[4] / 5, "* T4")
        + param2str(parameters[5], "* invT")
    )
    cw.writer(fstream, expression if expression else "0.0")

    if syms:
        return (
            parameters[0]
            + (parameters[1] / 2) * syms.T_smp
            + (parameters[2] / 3) * syms.T2_smp
            + (parameters[3] / 4) * syms.T3_smp
            + (parameters[4] / 5) * syms.T4_smp
            + (parameters[5]) * syms.invT_smp
        )


def entropy_nasa7(fstream, parameters, syms=None):
    """Write NASA7 polynomial for entropy."""
    expression = (
        param2str(parameters[0], "* logT")
        + param2str(parameters[1], "* T")
        + param2str(parameters[2] / 2, "* T2")
        + param2str(parameters[3] / 3, "* T3")
        + param2str(parameters[4] / 4, "* T4")
        + param2str(parameters[6])
    )
    cw.writer(fstream, expression if expression else "0.0")

    if syms:
        return (
            (parameters[0]) * syms.logT_smp
            + (parameters[1]) * syms.T_smp
            + (parameters[2] / 2) * syms.T2_smp
            + (parameters[3] / 3) * syms.T3_smp
            + (parameters[4] / 4) * syms.T4_smp
            + parameters[6]
        )


def cv_nasa9(fstream, parameters):
    """Write NASA9 polynomial for cv."""
    expression = (
        param2str(parameters[0], "* invT2")
        + param2str(parameters[1], "* invT")
        + param2str(parameters[2] - 1.0)
        + param2str(parameters[3], "* T")
        + param2str(parameters[4], " * T2")
        + param2str(parameters[5], "* T3")
        + param2str(parameters[6], "* T4")
    )
    cw.writer(fstream, expression if expression else "0.0")


def eval_cv_species_nasa9(model, temp):
    """Evaluate cv with NASA9 polynomial."""
    nzones = int(model.coeffs[0])
    coeffs = [model.coeffs[3 + 11 * x : 12 + 11 * x] for x in range(nzones)]
    for c in coeffs:
        assert len(c) == 9, "NASA9 polynomial coefficients must be of length 9."
    min_bounds = [model.coeffs[1 + 11 * x] for x in range(nzones)]
    max_bounds = [model.coeffs[2 + 11 * x] for x in range(nzones)]
    interval = sorted(list(set(min_bounds) & set(max_bounds)))
    idx = bisect.bisect_left(interval, temp)
    parameters = coeffs[idx]

    return (
        parameters[0] * 1.0 / (temp * temp)
        + parameters[1] * 1.0 / (temp)
        + (parameters[2] - 1.0)
        + parameters[3] * temp
        + parameters[4] * temp * temp
        + parameters[5] * temp * temp * temp
        + parameters[6] * temp * temp * temp * temp
    )


def cp_nasa9(fstream, parameters):
    """Write NASA9 polynomial for cp."""
    expression = (
        param2str(parameters[0], "* invT2")
        + param2str(parameters[1], "* invT")
        + param2str(parameters[2])
        + param2str(parameters[3], "* T")
        + param2str(parameters[4], " * T2")
        + param2str(parameters[5], "* T3")
        + param2str(parameters[6], "* T4")
    )
    cw.writer(fstream, expression if expression else "0.0")


def dcpdtemp_nasa9(fstream, parameters):
    """Write NASA9 polynomial for dcpdtemp."""
    expression = (
        param2str(-parameters[0] * 2.0, "* invT3")
        + param2str(-parameters[1], "* invT2")
        + param2str(parameters[3])
        + param2str(parameters[4] * 2.0, "* T")
        + param2str(parameters[5] * 3.0, "* T2")
        + param2str(parameters[6] * 4.0, "* T3")
    )
    cw.writer(fstream, expression if expression else "0.0")


def gibbs_nasa9(fstream, parameters, syms=None):
    """Write NASA9 polynomial for Gibbs."""
    expression = (
        param2str(-parameters[0] / 2, "* invT2")
        + param2str(parameters[7] + parameters[1], "* invT")
        + param2str(parameters[1], "* logT * invT")
        + param2str(-parameters[2], "* logT")
        + param2str(parameters[2] - parameters[8], "")
        + param2str((-parameters[3] / 2), "* T")
        + param2str((-parameters[4] / 6), "* T2")
        + param2str((-parameters[5] / 12), "* T3")
        + param2str((-parameters[6] / 20), "* T4")
    )
    cw.writer(fstream, expression if expression else "0.0")

    if syms:
        return (
            -parameters[0] / 2 * syms.invT2_smp
            + (parameters[7] + parameters[1]) * syms.invT_smp
            + parameters[1] * syms.logT_smp * syms.invT_smp
            + (-parameters[2]) * syms.logT_smp
            + parameters[2]
            - parameters[8]
            + (-parameters[3] / 2) * syms.T_smp
            + (-parameters[4] / 6) * syms.T2_smp
            + (-parameters[5] / 12) * syms.T3_smp
            + (-parameters[6] / 20) * syms.T4_smp
        )


def helmholtz_nasa9(fstream, parameters, syms=None):
    """Write NASA9 polynomial for Helmholtz."""
    expression = (
        param2str(-parameters[0] / 2, "* invT2")
        + param2str(parameters[7] + parameters[1], "* invT")
        + param2str(parameters[1], "* logT * invT")
        + param2str(-parameters[2], "* logT")
        + param2str(parameters[2] - parameters[8] - 1.0, "")
        + param2str((-parameters[3] / 2), "* T")
        + param2str((-parameters[4] / 6), "* T2")
        + param2str((-parameters[5] / 12), "* T3")
        + param2str((-parameters[6] / 20), "* T4")
    )
    cw.writer(fstream, expression if expression else "0.0")

    if syms:
        return (
            -parameters[0] / 2 * syms.invT2_smp
            + (parameters[7] + parameters[1]) * syms.invT_smp
            + parameters[1] * syms.logT_smp * syms.invT_smp
            + (-parameters[2]) * syms.logT_smp
            + parameters[2]
            - parameters[8]
            - 1.0
            + (-parameters[3] / 2) * syms.T_smp
            + (-parameters[4] / 6) * syms.T2_smp
            + (-parameters[5] / 12) * syms.T3_smp
            + (-parameters[6] / 20) * syms.T4_smp
        )


def internal_energy_nasa9(fstream, parameters, syms=None):
    """Write NASA9 polynomial for internal energy."""
    expression = (
        param2str(-parameters[0], "* invT2")
        + param2str(parameters[1], "* logT * invT")
        + param2str(parameters[2] - 1.0)
        + param2str(parameters[3] / 2, "* T")
        + param2str(parameters[4] / 3, "* T2")
        + param2str(parameters[5] / 4, "* T3")
        + param2str(parameters[6] / 5, "* T4")
        + param2str(parameters[7], "* invT")
    )
    cw.writer(fstream, expression if expression else "0.0")

    if syms:
        return (
            -parameters[0] * syms.invT2_smp
            + parameters[1] * syms.logT_smp * syms.invT_smp
            + parameters[2]
            - 1.0
            + (parameters[3] / 2) * syms.T_smp
            + (parameters[4] / 3) * syms.T2_smp
            + (parameters[5] / 4) * syms.T3_smp
            + (parameters[6] / 5) * syms.T4_smp
            + parameters[7] * syms.invT_smp
        )


def enthalpy_nasa9(fstream, parameters, syms=None):
    """Write NASA9 polynomial for enthalpy."""
    expression = (
        param2str(-parameters[0], "* invT2")
        + param2str(parameters[1], "* logT * invT")
        + param2str(parameters[2])
        + param2str(parameters[3] / 2, "* T")
        + param2str(parameters[4] / 3, "* T2")
        + param2str(parameters[5] / 4, "* T3")
        + param2str(parameters[6] / 5, "* T4")
        + param2str(parameters[7], "* invT")
    )
    cw.writer(fstream, expression if expression else "0.0")

    if syms:
        return (
            -parameters[0] * syms.invT2_smp
            + parameters[1] * syms.logT_smp * syms.invT_smp
            + parameters[2]
            + (parameters[3] / 2) * syms.T_smp
            + (parameters[4] / 3) * syms.T2_smp
            + (parameters[5] / 4) * syms.T3_smp
            + (parameters[6] / 5) * syms.T4_smp
            + parameters[7] * syms.invT_smp
        )


def entropy_nasa9(fstream, parameters, syms=None):
    """Write NASA9 polynomial for entropy."""
    expression = (
        param2str(-parameters[0] / 2, "* invT2")
        + param2str(-parameters[1], "* invT")
        + param2str(parameters[2], "* logT")
        + param2str(parameters[3], "* T")
        + param2str(parameters[4] / 2, "* T2")
        + param2str(parameters[5] / 3, "* T3")
        + param2str(parameters[6] / 4, "* T4")
        + param2str(parameters[8])
    )
    cw.writer(fstream, expression if expression else "0.0")

    if syms:
        return (
            (-parameters[0] / 2) * syms.invT2_smp
            + (-parameters[1]) * syms.invT_smp
            + (parameters[2]) * syms.logT_smp
            + (parameters[3]) * syms.T_smp
            + (parameters[4] / 2) * syms.T2_smp
            + (parameters[5] / 3) * syms.T3_smp
            + (parameters[6] / 4) * syms.T4_smp
            + parameters[8]
        )
