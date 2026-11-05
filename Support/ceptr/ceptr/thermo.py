"""Thermodynamics functions."""

import bisect
import itertools
import warnings

import numpy as np
from cantera.speciesthermo import Nasa9PolyMultiTempRegion, NasaPoly2

import ceptr.writer as cw

# Exponents of T in the cp/R basis of each polynomial form
CP_EXPONENTS = {"nasa7": np.arange(0, 5), "nasa9": np.arange(-2, 5)}
# h and s are pinned to their original values at this temperature when smoothing
SMOOTH_TREF = 298.15
# Sample points per temperature range in the smoothing fit
SMOOTH_NSAMPLES = 200
# Jumps at a polynomial break above these limits are treated as bad input data
SMOOTH_MAX_JUMPS = {"cp/R": 0.5, "h/RT": 0.05, "s/R": 0.05}
# Jumps above these limits are reported by the continuity check. They sit above the
# rounding of published coefficients, which is around 1e-6 for cp, h and s.
CONTINUITY_TOLS = {"cp/R": 1e-3, "dcp/RdT": 1e-6, "h/RT": 1e-3, "s/R": 1e-3}
# Largest relative change in cp that smoothing may make to a species
SMOOTH_MAX_CP_ERROR = 0.05


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


def thermo_zones(model):
    """Return the temperature bounds and per-range coefficients of a model.

    Coefficients are ordered by increasing temperature range, each holding the
    cp/R coefficients followed by the enthalpy and entropy constants. Returns
    None for a model with a single polynomial, which has no break to smooth.
    """
    if model_type(model) == "nasa7":
        high, low = model.coeffs[1:8], model.coeffs[8:15]
        # Same test as analyze_thermodynamics, which emits one polynomial for these
        if np.allclose(high, low, atol=1e-22):
            return None
        return [model.min_temp, model.coeffs[0], model.max_temp], [low, high]

    order = _zone_order(model)
    if len(order) < 2:
        return None
    lows = [model.coeffs[1 + 11 * x] for x in order]
    highs = [model.coeffs[2 + 11 * x] for x in order]
    assert lows[1:] == highs[:-1], "NASA9 temperature ranges must be contiguous."
    return lows + highs[-1:], [model.coeffs[3 + 11 * x : 12 + 11 * x] for x in order]


def _zone_order(model):
    """Return NASA9 zone indices by increasing temperature."""
    nzones = int(model.coeffs[0])
    return sorted(range(nzones), key=lambda x: model.coeffs[1 + 11 * x])


def _with_zones(model, coeffs):
    """Return a copy of a model with new per-range coefficients."""
    new_coeffs = np.array(model.coeffs)
    if model_type(model) == "nasa7":
        new_coeffs[8:15], new_coeffs[1:8] = coeffs
    else:
        for x, zone in zip(_zone_order(model), coeffs):
            new_coeffs[3 + 11 * x : 12 + 11 * x] = zone
    return type(model)(
        model.min_temp, model.max_temp, model.reference_pressure, new_coeffs
    )


def _cp_r(exps, coeffs, temp):
    """Evaluate cp/R."""
    return sum(a * temp ** float(e) for a, e in zip(coeffs, exps))


def _dcp_r_dt(exps, coeffs, temp):
    """Evaluate d(cp/R)/dT."""
    return sum(e * a * temp ** float(e - 1) for a, e in zip(coeffs, exps))


def _h_rt(exps, coeffs, temp):
    """Evaluate h/RT."""
    integral = sum(
        a * (np.log(temp) if e == -1 else temp ** float(e + 1) / (e + 1))
        for a, e in zip(coeffs, exps)
    )
    return (integral + coeffs[len(exps)]) / temp


def _s_r(exps, coeffs, temp):
    """Evaluate s/R."""
    integral = sum(
        a * (np.log(temp) if e == 0 else temp ** float(e) / e)
        for a, e in zip(coeffs, exps)
    )
    return integral + coeffs[len(exps) + 1]


def break_jumps(exps, bounds, coeffs):
    """Return the largest absolute jumps in each property across polynomial breaks."""
    jumps = dict.fromkeys(("cp/R", "dcp/RdT", "h/RT", "s/R"), 0.0)
    funcs = {"cp/R": _cp_r, "dcp/RdT": _dcp_r_dt, "h/RT": _h_rt, "s/R": _s_r}
    for temp, low, high in zip(bounds[1:-1], coeffs, coeffs[1:]):
        for name, func in funcs.items():
            jumps[name] = max(
                jumps[name], abs(func(exps, high, temp) - func(exps, low, temp))
            )
    return jumps


def _fit_c1_cp(exps, bounds, coeffs):
    """Fit cp/R over all ranges, with value and slope continuous at each break.

    This is an equality-constrained least-squares problem, solved through its KKT
    system. Temperature is scaled to keep that system well conditioned.
    """
    scale = 1000.0
    nexps, nzones = len(exps), len(coeffs)

    def basis(temp, deriv=0):
        temp = np.atleast_1d(temp) / scale
        if deriv == 0:
            return np.stack([temp ** float(e) for e in exps], axis=1)
        return np.stack([e * temp ** float(e - 1) for e in exps], axis=1)

    lhs = np.zeros((SMOOTH_NSAMPLES * nzones, nexps * nzones))
    rhs = np.zeros(SMOOTH_NSAMPLES * nzones)
    for k, zone in enumerate(coeffs):
        temps = np.linspace(bounds[k], bounds[k + 1], SMOOTH_NSAMPLES)
        rows = slice(k * SMOOTH_NSAMPLES, (k + 1) * SMOOTH_NSAMPLES)
        lhs[rows, k * nexps : (k + 1) * nexps] = basis(temps)
        rhs[rows] = _cp_r(exps, zone, temps)

    constraints = np.zeros((2 * (nzones - 1), nexps * nzones))
    for k, temp in enumerate(bounds[1:-1]):
        for deriv in (0, 1):
            row = basis(temp, deriv)[0]
            constraints[2 * k + deriv, k * nexps : (k + 1) * nexps] = row
            constraints[2 * k + deriv, (k + 1) * nexps : (k + 2) * nexps] = -row

    ncons = constraints.shape[0]
    kkt = np.block(
        [[lhs.T @ lhs, constraints.T], [constraints, np.zeros((ncons, ncons))]]
    )
    solution = np.linalg.solve(kkt, np.concatenate([lhs.T @ rhs, np.zeros(ncons)]))
    fitted = solution[: nexps * nzones].reshape(nzones, nexps)
    return fitted * scale ** (-exps.astype(float))


def smooth_coefficients(exps, bounds, coeffs):
    """Return coefficients giving C1 cp and C2 enthalpy and entropy across breaks.

    Heat capacity is refit with its value and slope continuous at each break, and
    the breaks themselves do not move. The integration constants reproduce the
    original h and s at SMOOTH_TREF (or the nearest temperature in range), then
    carry continuity of h and s out to the other ranges.
    """
    fitted = _fit_c1_cp(exps, bounds, coeffs)
    nexps = len(exps)
    smoothed = [np.concatenate([zone, [0.0, 0.0]]) for zone in fitted]

    def match_constants(idx, temp, target):
        smoothed[idx][nexps] = (
            _h_rt(exps, target, temp) - _h_rt(exps, smoothed[idx], temp)
        ) * temp
        smoothed[idx][nexps + 1] = _s_r(exps, target, temp) - _s_r(
            exps, smoothed[idx], temp
        )

    tref = np.clip(SMOOTH_TREF, bounds[0], bounds[-1])
    pinned = min(bisect.bisect_right(bounds, tref) - 1, len(coeffs) - 1)
    match_constants(pinned, tref, coeffs[pinned])
    for idx in range(pinned + 1, len(coeffs)):
        match_constants(idx, bounds[idx], smoothed[idx - 1])
    for idx in range(pinned - 1, -1, -1):
        match_constants(idx, bounds[idx + 1], smoothed[idx + 1])
    return smoothed


def _jumps_over(jumps, limits):
    """Describe the jumps over their limits, or return an empty string if none are."""
    return ", ".join(
        f"{name} jump {jumps[name]:.3g} > {limit:g}"
        for name, limit in limits.items()
        if jumps[name] > limit
    )


def check_thermo_continuity(mechanism):
    """Warn if thermodynamic polynomials are not C1 at their temperature breaks."""
    bad_data = []
    worst = {name: (0.0, None) for name in CONTINUITY_TOLS}
    nspecies = 0
    nflagged = 0
    for species in mechanism.species():
        zones = thermo_zones(species.thermo)
        if zones is None:
            continue
        nspecies += 1
        jumps = break_jumps(CP_EXPONENTS[model_type(species.thermo)], *zones)
        if not _jumps_over(jumps, CONTINUITY_TOLS):
            continue
        nflagged += 1
        for name in worst:
            if jumps[name] > worst[name][0]:
                worst[name] = (jumps[name], species.name)
        bad = _jumps_over(jumps, SMOOTH_MAX_JUMPS)
        if bad:
            bad_data.append(f"{species.name}: {bad}")

    if not nflagged:
        return

    largest = ", ".join(
        f"{name} {jump:.3g} ({name_sp})"
        for name, (jump, name_sp) in worst.items()
        if jump > CONTINUITY_TOLS[name]
    )
    message = (
        f"{nflagged} of {nspecies} multi-range species in {mechanism.source} are not C1"
        f" at their temperature breaks. Largest jumps: {largest}."
    )
    if bad_data:
        details = "\n  ".join(bad_data)
        message += (
            f"\nThese jumps are too large to be smoothed, so check the data of these"
            f" species:\n  {details}\nThen rerun CEPTR with --smooth_thermo to refit"
            " the polynomials."
        )
    else:
        message += " Rerun CEPTR with --smooth_thermo to refit the polynomials."
    warnings.warn(message, stacklevel=2)


def smooth_thermo(mechanism):
    """Refit the thermodynamic polynomials of a mechanism in place to be C1.

    Raises ValueError, before changing anything, if a species has jumps too large
    to be anything but bad input data, or would need too large a change in cp. A
    refit would otherwise hide such an error by spreading it over the whole range.
    """
    updates = {}
    errors = []
    worst_cp_error = 0.0
    for species in mechanism.species():
        model = species.thermo
        mtype = model_type(model)
        zones = thermo_zones(model)
        if zones is None:
            continue
        bounds, coeffs = zones
        exps = CP_EXPONENTS[mtype]

        jumps = break_jumps(exps, bounds, coeffs)
        bad = _jumps_over(jumps, SMOOTH_MAX_JUMPS)
        if bad:
            errors.append(f"{species.name}: {bad}")
            continue

        smoothed = smooth_coefficients(exps, bounds, coeffs)
        cp_error = 0.0
        for k, (old, new) in enumerate(zip(coeffs, smoothed)):
            temps = np.linspace(bounds[k], bounds[k + 1], SMOOTH_NSAMPLES)
            old_cp = _cp_r(exps, old, temps)
            cp_error = max(
                cp_error,
                np.max(np.abs(_cp_r(exps, new, temps) - old_cp) / np.abs(old_cp)),
            )
        if cp_error > SMOOTH_MAX_CP_ERROR:
            errors.append(
                f"{species.name}: cp change {cp_error:.3g} >"
                f" {SMOOTH_MAX_CP_ERROR:g}"
            )
            continue

        worst_cp_error = max(worst_cp_error, cp_error)
        updates[species.name] = (species, _with_zones(model, smoothed))

    if errors:
        details = "\n  ".join(errors)
        raise ValueError(
            "Cannot smooth thermodynamic data, check these species in the"
            f" mechanism:\n  {details}"
        )

    for name, (species, model) in updates.items():
        species.thermo = model
        mechanism.modify_species(mechanism.species_index(name), species)
    print(
        f"Smoothed thermodynamic data of {len(updates)} species, largest change in"
        f" cp {100 * worst_cp_error:.3g}%"
    )


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
                    f"species with inflection points at T = {*interval,} kelvin"
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
                        f"""else if ( ({interval[k-1]:g} <= T) && (T < {interval[k]:g})) {{""",
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
