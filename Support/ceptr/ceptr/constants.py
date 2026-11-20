"""Constants."""

import cantera as ct
from pint import UnitRegistry

smallnum = 1e-100
ureg = UnitRegistry()
ureg.default_system = "mks"
kb = (ct.boltzmann * ureg.joule / ureg.kelvin).to(ureg.erg / ureg.kelvin).m
R = 8.31446261815324e7 * (ureg.erg / (ureg.mole / ureg.kelvin))
# Same gas constant as R (CODATA 2018, as in Cantera), expressed in cal/mol/K
Rc = (8.31446261815324 * ureg.joule / ureg.mole / ureg.kelvin).to(
    ureg.cal / ureg.mole / ureg.kelvin
)
Na = (ct.avogadro / ureg.kmole).to(1.0 / ureg.mole).m
qc = ct.electron_charge
Patm = 1013250.0
Patm_pa = Patm / 10.0
sym = ""
