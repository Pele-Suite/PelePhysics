"""Tests for cepter."""

import math
import pathlib
import random
import types

import cantera as ct
import pytest
import symengine

import ceptr.converter as converter
import ceptr.reaction_info as cri
import ceptr.utilities as cu
from ceptr import __version__
from ceptr.ceptr import parse_lst_file, parse_qss_lst_file
from ceptr.symbolic_math import SymbolicMath


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


def test_kc_unit_conversions():
    """Check the refC powers of the Kc unit conversions for any mole change."""
    # The real symbolic conversion helpers, without building a mechanism
    syms = SymbolicMath.__new__(SymbolicMath)
    syms.refC_smp = symengine.Symbol("refC")
    syms.refCinv_smp = symengine.Symbol("refCinv")
    syms.remove_1 = False
    refc = 2.0
    env = {"refC": refc, "refCinv": 1.0 / refc, "pow": pow, "max": max}
    subs = {syms.refC_smp: refc, syms.refCinv_smp: 1.0 / refc}

    def cpp(expr):
        return eval(expr.replace("std::max", "max"), env)

    # Integer mole changes, and fractional ones, which go through pow in C++
    # (with 6 decimals, so the coefficients here are exact in 6 decimals)
    moles = [(1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1)]
    moles += [(1, 1.5), (1, 2.5), (1, 2.75), (1.5, 1), (2.5, 1), (2.75, 1)]
    for n_reac, n_prod in moles:
        reaction = types.SimpleNamespace(
            reactants={"A": float(n_reac)}, products={"B": float(n_prod)}
        )
        dim = n_prod - n_reac
        # Kc = Kp * refC**dim, and fkc_conv_inv converts by its inverse
        assert math.isclose(cpp(cu.kc_conv(None, reaction)), refc**dim)
        conv, conv_smp = cu.fkc_conv_inv(None, None, reaction, syms)
        assert math.isclose(cpp(conv), refc**-dim)
        assert math.isclose(float(symengine.sympify(conv_smp).subs(subs)), refc**-dim)


PLOG_YAML = """
phases:
- name: gas
  thermo: ideal-gas
  elements: [H, O, Ar]
  species: [{gri30.yaml/species: [H, O, OH, AR]}]
  kinetics: gas
  reactions: all
reactions:
- equation: %s
  type: pressure-dependent-Arrhenius
  rate-constants:
  - {P: 0.01 atm, A: 1.0e12, b: 0, Ea: 0}
  - {P: 1 atm, A: 1.0e13, b: 0, Ea: 0}
"""


@pytest.mark.parametrize("equation", ["H + O <=> OH", "H + O + AR <=> OH + AR"])
def test_plog_detection(equation):
    """PLOG reactions are found with and without a third body."""
    mechanism = ct.Solution(yaml=PLOG_YAML % equation)
    assert cri.has_plog_reactions(mechanism.reactions())
    assert cri.ReactionInfo(mechanism, None).has_plog_reactions
    assert not cri.has_plog_reactions(ct.Solution("h2o2.yaml").reactions())


def test_parse_lst_file(tmp_path):
    """Blank and comment lines in a mechanism list file are skipped."""
    lst = tmp_path / "list_mech"
    lst.write_text(
        "# comment\n\nair/mechanism.yaml\n  \nplog/mechanism.yaml --plog=1e5\n\n"
    )
    names, pressures = parse_lst_file(lst)
    assert names == [tmp_path / "air/mechanism.yaml", tmp_path / "plog/mechanism.yaml"]
    assert pressures == [None, 1e5]

    lst.write_text("\nqss/mechanism.yaml qss/format.toml\n\n")
    names, formats, pressures = parse_qss_lst_file(lst)
    assert names == [tmp_path / "qss/mechanism.yaml"]
    assert formats == [tmp_path / "qss/format.toml"]
    assert pressures == [None]

    # Malformed entries name the file and line
    for text, parse in [
        ("qss/mechanism.yaml\n", parse_qss_lst_file),
        ("air/mechanism.yaml --plog=\n", parse_lst_file),
        ("air/mechanism.yaml --plog=1 atm\n", parse_lst_file),
    ]:
        lst.write_text("# comment\n" + text)
        with pytest.raises(ValueError, match=f"{lst}:2: "):
            parse(lst)
