"""
Code-unit to cgs conversions, matching AthenaK's ``src/units/units.cpp``.

    units = Units.from_params(sim.params)
    T = units.temperature_cgs * gm1 * frame["eint"] / frame["dens"]   # K
    v = frame["velx"] * units.factor("velocity", "km/s")               # km/s
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants, as in AthenaK's units.hpp
ATOMIC_MASS_UNIT_CGS = 1.660538921e-24  # g
K_BOLTZMANN_CGS = 1.3806488e-16         # erg/K
PC_CGS = 3.0856775809623245e18          # cm
YR_CGS = 3.15576e7                      # s
MYR_CGS = 3.15576e13                    # s

# Units a quantity of each dimension can be shown in, as their value in cgs.
# Density can also be shown as a number density, see Units.factor.
UNIT_NAMES = {
    "length": {"cm": 1.0, "pc": PC_CGS, "kpc": 1e3 * PC_CGS},
    "time": {"s": 1.0, "yr": YR_CGS, "kyr": 1e3 * YR_CGS, "Myr": MYR_CGS},
    "velocity": {"cm/s": 1.0, "km/s": 1e5},
    "density": {"g/cm^3": 1.0, "cm^-3": None},
    "pressure": {"dyne/cm^2": 1.0, "erg/cm^3": 1.0},
    "temperature": {"K": 1.0},
    "magnetic": {"G": 1.0, "uG": 1e-6},
    "emissivity": {"erg/cm^3/s": 1.0},
}


@dataclass(frozen=True)
class Units:
    """Code units of one run, from the <units> block of its athinput."""

    length_cgs: float
    mass_cgs: float
    time_cgs: float
    mu: float = 1.0  # mean molecular weight; AthenaK's default is 1

    @classmethod
    def from_params(cls, params: dict[str, dict[str, str]]) -> Units:
        """Build from parsed athinput params; raises if there is no <units> block."""
        block = params.get("units")
        if not block:
            raise ValueError("athinput has no <units> block; cgs conversions unavailable")
        return cls(
            length_cgs=float(block["length_cgs"]),
            mass_cgs=float(block["mass_cgs"]),
            time_cgs=float(block["time_cgs"]),
            mu=float(block.get("mu", 1.0)),
        )

    @property
    def velocity_cgs(self) -> float:
        return self.length_cgs / self.time_cgs

    @property
    def density_cgs(self) -> float:
        return self.mass_cgs / self.length_cgs**3

    @property
    def pressure_cgs(self) -> float:
        return self.mass_cgs * self.velocity_cgs**2 / self.length_cgs**3

    @property
    def temperature_cgs(self) -> float:
        """Temperature in K of P/rho = 1 in code units."""
        return self.velocity_cgs**2 * self.mu * ATOMIC_MASS_UNIT_CGS / K_BOLTZMANN_CGS

    @property
    def number_density_cgs(self) -> float:
        """Number density in cm^-3 of rho = 1 in code units."""
        return self.density_cgs / (self.mu * ATOMIC_MASS_UNIT_CGS)

    @property
    def emissivity_cgs(self) -> float:
        """Energy loss rate per volume in erg/cm^3/s of 1 in code units."""
        return self.pressure_cgs / self.time_cgs

    @property
    def magnetic_cgs(self) -> float:
        """Field in G of B = 1 in code units (code P_mag = B^2/2, Gaussian B^2/8pi)."""
        return math.sqrt(4.0 * math.pi * self.pressure_cgs)

    def factor(self, dimension: str, unit: str) -> float:
        """
        Multiply a code-unit value of ``dimension`` by this to get it in ``unit``
        (a key of ``UNIT_NAMES[dimension]``, or ``"code"``).
        """
        if unit == "code":
            return 1.0
        names = UNIT_NAMES.get(dimension, {})
        if unit not in names:
            raise ValueError(f"cannot show a {dimension} quantity in {unit!r}; "
                             f"use one of {['code', *names]}")
        if dimension == "density" and unit == "cm^-3":
            return self.number_density_cgs
        code_cgs = {
            "length": self.length_cgs,
            "time": self.time_cgs,
            "velocity": self.velocity_cgs,
            "density": self.density_cgs,
            "pressure": self.pressure_cgs,
            "temperature": self.temperature_cgs,
            "magnetic": self.magnetic_cgs,
            "emissivity": self.emissivity_cgs,
        }[dimension]
        return code_cgs / names[unit]
