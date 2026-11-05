"""Tests for cepter."""

import filecmp
import math
import pathlib
import random
import shutil

import cantera as ct
import numpy as np
import pytest

import ceptr.ceptr as ceptr
import ceptr.converter as converter
import ceptr.thermo as cth
import ceptr.utilities as cu
from ceptr import __version__


def mechanism_path(mname):
    """Determine mechanism path."""
    this_file_dir = pathlib.Path(__file__).parent.resolve()
    model_path = "Mechanisms"
    return this_file_dir.parents[2] / model_path / mname


def test_version():
    """Test the version."""
    assert __version__ == "0.1.0"


def test_air():
    """Test mechanism generation of air."""
    mech_path = mechanism_path("air")
    fname = mech_path / "mechanism.yaml"
    mechanism = ct.Solution(fname)
    interface = None
    chemistry = "homogeneous"
    conv = converter.Converter(mechanism, interface, chemistry)
    conv.writer()
    conv.formatter()


def test_lidryer():
    """Test mechanism generation of LiDryer."""
    mech_path = mechanism_path("LiDryer")
    fname = mech_path / "mechanism.yaml"
    mechanism = ct.Solution(fname)
    interface = None
    chemistry = "homogeneous"
    conv = converter.Converter(mechanism, interface, chemistry)
    conv.writer()
    conv.formatter()


def test_dodecane_lu():
    """Test mechanism generation of dodecane_lu."""
    mech_path = mechanism_path("dodecane_lu")
    fname = mech_path / "mechanism.yaml"
    mechanism = ct.Solution(fname)
    interface = None
    chemistry = "homogeneous"
    conv = converter.Converter(mechanism, interface, chemistry)
    conv.writer()
    conv.formatter()


def test_kc_exp():
    """Check the product form of the equilibrium constants against exp."""
    mech_path = mechanism_path("dodecane_lu")
    mechanism = ct.Solution(mech_path / "mechanism.yaml")
    conv = converter.Converter(mechanism, None, "homogeneous")
    species_info = conv.species_info
    rng = random.Random(0)
    g_rt = [rng.uniform(-50.0, 50.0) for _ in range(species_info.n_species)]
    env = {
        "g_RT": g_rt,
        "ge": [math.exp(-g) for g in g_rt],
        "inv_ge": [math.exp(g) for g in g_rt],
        "exp": math.exp,
    }
    for reaction in mechanism.reactions():
        arg = eval(cu.sorted_kc_exp_arg(mechanism, species_info, reaction), env)
        for sign in (-1, 1):
            product, exp_form = cu.kc_exp_parts(mechanism, species_info, reaction, sign)
            assert product is not None and "exp" not in product
            assert math.isclose(eval(exp_form, env), math.exp(sign * arg))
            assert math.isclose(eval(product, env), math.exp(sign * arg), rel_tol=1e-12)


def test_kc_product_no_overflow():
    """Check that Kc products cannot overflow where they are used.

    Regression test: in LiDryer at 79 K, H + H2O2 <=> H2O + OH has a finite
    Kc (about 1e190), but a naive factor order overflows on its first two
    factors. The product form must be off there (exp fallback), and on the
    range where it is on, every left-to-right partial product must be finite
    and the result must match exp of the summed exponent.
    """
    mech_path = mechanism_path("LiDryer")
    mechanism = ct.Solution(mech_path / "mechanism.yaml")
    conv = converter.Converter(mechanism, None, "homogeneous")
    species_info = conv.species_info
    t_low, t_high, _ = cu.kc_product_plan(mechanism, species_info)
    assert t_low > 79.0
    for temp in [t_low, 250.0, 300.0, 1000.0, 3000.0, t_high]:
        g_rt = {
            s: float(cu.gibbs_rt(mechanism, s, [temp])[0])
            for s in species_info.nonqssa_species_list
        }
        g_list = [g_rt[s] for s in species_info.nonqssa_species_list]
        env = {
            "g_RT": g_list,
            "ge": [math.exp(-g) for g in g_list],
            "inv_ge": [math.exp(g) for g in g_list],
            "exp": math.exp,
        }
        for reaction in mechanism.reactions():
            arg = eval(cu.sorted_kc_exp_arg(mechanism, species_info, reaction), env)
            for sign in (-1, 1):
                product, _ = cu.kc_exp_parts(mechanism, species_info, reaction, sign)
                assert product is not None
                value = 1.0
                for factor in product.split(" * "):
                    value *= eval(factor, env)
                    assert math.isfinite(value) and value > 0.0
                assert math.isclose(value, math.exp(sign * arg), rel_tol=1e-10)


def assert_smooth(mechanism):
    """Check every multi-range species is continuous up to the slope of cp."""
    nchecked = 0
    for species in mechanism.species():
        zones = cth.thermo_zones(species.thermo)
        if zones is None:
            continue
        nchecked += 1
        exps = cth.CP_EXPONENTS[cth.model_type(species.thermo)]
        jumps = cth.break_jumps(exps, *zones)
        assert all(jump < 1e-9 for jump in jumps.values()), (species.name, jumps)
    assert nchecked > 0


def load_copy(mname, tmp_path, fname="mechanism.yaml"):
    """Copy a mechanism out of the source tree, as generation writes next to it."""
    copy = tmp_path / fname
    shutil.copy(mechanism_path(mname) / fname, copy)
    return copy


@pytest.mark.parametrize("mname", ["LiDryer", "dodecane_lu", "grimech30"])
def test_smooth_thermo_is_continuous(mname):
    """Test smoothing removes jumps and leaves h and s at the reference point."""
    mechanism = ct.Solution(mechanism_path(mname) / "mechanism.yaml")
    original = mechanism.species()
    cth.smooth_thermo(mechanism)
    assert_smooth(mechanism)

    for before, after in zip(original, mechanism.species()):
        if cth.thermo_zones(before.thermo) is None:
            continue
        tref = max(cth.SMOOTH_TREF, before.thermo.min_temp)
        assert after.thermo.h(tref) == pytest.approx(before.thermo.h(tref), rel=1e-9)
        assert after.thermo.s(tref) == pytest.approx(before.thermo.s(tref), rel=1e-9)
        assert after.thermo.cp(1500.0) == pytest.approx(
            before.thermo.cp(1500.0), rel=cth.SMOOTH_MAX_CP_ERROR
        )


def test_smooth_thermo_nasa9():
    """Test smoothing with more than one break, which NASA9 models can have."""
    temps = [200.0, 1000.0, 6000.0, 20000.0]
    zones = [
        [1e4, -300.0, 6.0, -5e-3, 8e-6, -6e-9, 2e-12, 3e3, -8.0],
        [6e5, -2e3, 8.0, -3e-4, 1e-7, -1e-11, 5e-16, 1e4, -20.0],
        [8e8, -5e5, 90.0, -5e-3, 1e-7, -1e-11, 5e-16, 4e6, -700.0],
    ]
    exps = cth.CP_EXPONENTS["nasa9"]
    smoothed = cth.smooth_coefficients(exps, temps, [np.array(z) for z in zones])
    jumps = cth.break_jumps(exps, temps, smoothed)
    assert all(jump < 1e-8 for jump in jumps.values()), jumps


@pytest.mark.parametrize("mname", ["LiDryer", "H2-CO-CO2-3spec"])
def test_smooth_thermo_evaluators_match_cantera(mname):
    """Test the zones and evaluators used for the fit, for NASA7 and NASA9."""
    mechanism = ct.Solution(mechanism_path(mname) / "mechanism.yaml")
    for species in mechanism.species():
        model = species.thermo
        zones = cth.thermo_zones(model)
        if zones is None:
            continue
        exps = cth.CP_EXPONENTS[cth.model_type(model)]
        for k, zone in enumerate(zones[1]):
            temp = 0.5 * (zones[0][k] + zones[0][k + 1])
            rgas = ct.gas_constant
            assert cth._cp_r(exps, zone, temp) == pytest.approx(model.cp(temp) / rgas)
            assert cth._h_rt(exps, zone, temp) == pytest.approx(
                model.h(temp) / (rgas * temp)
            )
            assert cth._s_r(exps, zone, temp) == pytest.approx(model.s(temp) / rgas)


def test_smooth_thermo_nasa9_mechanism():
    """Test smoothing a mechanism with NASA9 species, and that it reaches Cantera."""
    mechanism = ct.Solution(mechanism_path("H2-CO-CO2-3spec") / "mechanism.yaml")
    assert any(
        cth.model_type(species.thermo) == "nasa9" for species in mechanism.species()
    )
    before = mechanism.species()
    cth.smooth_thermo(mechanism)
    assert_smooth(mechanism)
    for old, new in zip(before, mechanism.species()):
        assert isinstance(new.thermo, type(old.thermo))
        assert new.thermo.cp(800.0) == pytest.approx(old.thermo.cp(800.0), rel=0.05)


@pytest.mark.filterwarnings("ignore:NasaPoly2")
def test_smooth_thermo_rejects_bad_data():
    """Test smoothing refuses a large jump, and leaves the mechanism unchanged."""
    mechanism = ct.Solution(mechanism_path("LuEthylene") / "mechanism.yaml")
    before = mechanism.species("CH4").thermo.coeffs
    with pytest.raises(ValueError, match="CH2CHO"):
        cth.smooth_thermo(mechanism)
    assert np.array_equal(mechanism.species("CH4").thermo.coeffs, before)


def test_smooth_thermo_reaches_transport_fits():
    """Test the cv evaluation used for transport fits sees the smoothed data."""
    mechanism = ct.Solution(mechanism_path("dodecane_lu") / "mechanism.yaml")
    species = mechanism.species("OC12H23OOH")
    before = cth.eval_cv_species(mechanism, species, 1000.0)
    cth.smooth_thermo(mechanism)
    assert cth.eval_cv_species(mechanism, species, 1000.0) != before


def test_convert_smooth_thermo(tmp_path):
    """Test converting with smoothing writes a mechanism that reloads smooth."""
    fname = load_copy("LiDryer", tmp_path)
    ceptr.convert(
        fname, True, None, False, "homogeneous", "gas", None, None, smooth_thermo=True
    )
    assert (tmp_path / "mechanism.H").exists()
    assert_smooth(ct.Solution(tmp_path / "mechanism_smoothed.yaml"))
    # The input is not changed
    assert filecmp.cmp(fname, mechanism_path("LiDryer") / "mechanism.yaml")


def test_smooth_thermo_keeps_qssa_data(tmp_path):
    """Test the QSSA data the converter reads survives smoothing and writing."""
    fname = load_copy("dodecane_lu_qss", tmp_path, "qssa.yaml")
    mechanism = ct.Solution(fname)
    cth.smooth_thermo(mechanism)
    assert mechanism.input_data["qssa_species"]
    smoothed = tmp_path / "qssa_smoothed.yaml"
    mechanism.write_yaml(smoothed, header=True)
    reloaded = ct.Solution(smoothed)
    assert_smooth(reloaded)
    for key in ("qssa_species", "n_qssa_species", "forward_to_remove_idx"):
        if key in mechanism.input_data:
            assert reloaded.input_data[key] == mechanism.input_data[key]


def test_check_continuity_suggests_smoothing():
    """Test the check warns about slope jumps and suggests smoothing."""
    mechanism = ct.Solution(mechanism_path("LiDryer") / "mechanism.yaml")
    with pytest.warns(UserWarning, match="Rerun CEPTR with --smooth_thermo") as record:
        cth.check_thermo_continuity(mechanism)
    assert "too large" not in str(record[0].message)


def test_check_continuity_silent_when_smooth(recwarn):
    """Test the check says nothing once a mechanism has been smoothed."""
    mechanism = ct.Solution(mechanism_path("LiDryer") / "mechanism.yaml")
    cth.smooth_thermo(mechanism)
    cth.check_thermo_continuity(mechanism)
    assert len(recwarn) == 0


@pytest.mark.filterwarnings("ignore:NasaPoly2")
def test_check_continuity_names_bad_data():
    """Test the check points at data that smoothing would refuse."""
    mechanism = ct.Solution(mechanism_path("LuEthylene") / "mechanism.yaml")
    with pytest.warns(UserWarning, match="too large to be smoothed") as record:
        cth.check_thermo_continuity(mechanism)
    assert "CH2CHO: cp/R jump" in str(record[0].message)


def test_convert_checks_by_default(tmp_path):
    """Test a default conversion warns, and still leaves the thermo unchanged."""
    fname = load_copy("LiDryer", tmp_path)
    with pytest.warns(UserWarning, match="--smooth_thermo"):
        ceptr.convert(fname, True, None, False, "homogeneous", "gas", None, None)
    assert (tmp_path / "mechanism.H").exists()
    assert not (tmp_path / "mechanism_smoothed.yaml").exists()
