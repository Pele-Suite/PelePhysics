"""Tests for cepter."""

import math
import pathlib
import random

import cantera as ct

import ceptr.converter as converter
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


def test_kc_unit_conversions():
    """Check the refC powers of the Kc unit conversions for any mole change."""
    import types

    import sympy

    refc = sympy.Symbol("refC")
    refcinv = sympy.Symbol("refCinv")
    syms = types.SimpleNamespace(
        refC_smp=refc,
        refCinv_smp=refcinv,
        remove_1=False,
        convert_symb_to_int=lambda x: x,
    )
    env = {"refC": 2.0, "refCinv": 0.5}
    for n_reac, n_prod in [(1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1)]:
        reaction = types.SimpleNamespace(
            reactants={"A": float(n_reac)}, products={"B": float(n_prod)}
        )
        dim = n_prod - n_reac
        # Kc = Kp * refC**dim, and fkc_conv_inv converts by its inverse
        assert eval(cu.kc_conv(None, reaction), env) == 2.0**dim
        conv, conv_smp = cu.fkc_conv_inv(None, None, reaction, syms)
        assert eval(conv, env) == 2.0**-dim
        assert conv_smp.subs({refc: 2.0, refcinv: 0.5}) == 2.0**-dim

    # Fractional coefficients go through pow in C++, but the symbolic form
    # must still be the inverse of refC**dim
    for n_prod in [2.5, 0.5, 1.75]:
        reaction = types.SimpleNamespace(reactants={"A": 1.0}, products={"B": n_prod})
        dim = n_prod - 1.0
        _, conv_smp = cu.fkc_conv_inv(None, None, reaction, syms)
        assert math.isclose(float(conv_smp.subs({refc: 2.0, refcinv: 0.5})), 2.0**-dim)


def test_parse_lst_file(tmp_path):
    """Blank and comment lines in a mechanism list file are skipped."""
    from ceptr.ceptr import parse_lst_file, parse_qss_lst_file

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
