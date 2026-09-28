"""Tests for cepter."""

import pathlib

import cantera as ct

from ceptr import __version__
import ceptr.converter as converter


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


def test_polimi2020():
    """Test mechanism generation of POLIMI2020 at 1 atm."""
    mech_path = mechanism_path("POLIMI2020")
    fname = mech_path / "mechanism.yaml"
    mechanism = ct.Solution(fname)
    interface = None
    chemistry = "homogeneous"
    conv = converter.Converter(mechanism, interface, chemistry, plog_pressure=101325.0)
    conv.writer()
    conv.formatter()
