"""
Raw and derived quantities of a Frame, as arrays of shape ``(nx3, nx2, nx1)``.

    T = get(frame, "temp", sim.params, units="K")
    v = get(frame, "velx", sim.params, units="km/s")

Raw variables (``dens``, ``velx``, ``s_00``, ...) are read as stored; derived
ones are computed from them, in code units.  ``units`` converts a quantity
of known dimension (see ``utils.units.UNIT_NAMES``); "code" leaves it as is.
Derived quantities assume the primitive variables of ``hydro_w``/``mhd_w``
(``eint`` is internal energy per volume).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

if "." in __package__:  # imported as vis_athenak.plotting
    from ..utils.ism_cooling import ISMCoolFn
    from ..utils.units import Units
else:  # run from the repo root
    from utils.ism_cooling import ISMCoolFn
    from utils.units import Units

# Dimension of the raw variables; any other variable is dimensionless.
RAW_DIMENSIONS = {
    "dens": "density",
    "velx": "velocity", "vely": "velocity", "velz": "velocity",
    "eint": "pressure",
    "bcc1": "magnetic", "bcc2": "magnetic", "bcc3": "magnetic",
}


def _gamma(params) -> float:
    block = params.get("mhd") or params.get("hydro") or {}
    return float(block.get("gamma", 5.0 / 3.0))


def pres(frame, params):
    return (_gamma(params) - 1.0) * frame["eint"]


def temp(frame, params):
    """P/rho; in K with units="K", as AthenaK computes it for cooling."""
    return pres(frame, params) / frame["dens"]


def vmag(frame, params):
    return np.sqrt(frame["velx"]**2 + frame["vely"]**2 + frame["velz"]**2)


def bmag(frame, params):
    return np.sqrt(frame["bcc1"]**2 + frame["bcc2"]**2 + frame["bcc3"]**2)


def beta(frame, params):
    """Plasma beta, P_gas / (B^2/2)."""
    return pres(frame, params) / (0.5 * bmag(frame, params)**2)


def entropy(frame, params):
    """ln P - gamma ln rho, in code units; NaN where P or rho <= 0."""
    p = np.asarray(pres(frame, params), dtype=float)
    rho = np.asarray(frame["dens"], dtype=float)
    out = np.full_like(rho, np.nan)
    ok = (p > 0) & (rho > 0)
    out[ok] = np.log(p[ok]) - _gamma(params) * np.log(rho[ok])
    return out


def t_cool(frame, params):
    """Isobaric cooling time gamma P / ((gamma-1) n^2 Lambda(T)), ISM cooling, code units."""
    u = Units.from_params(params)
    gamma = _gamma(params)
    p_cgs = np.asarray(pres(frame, params), dtype=float) * u.pressure_cgs
    n_cgs = np.asarray(frame["dens"], dtype=float) * u.number_density_cgs
    T_cgs = np.asarray(temp(frame, params), dtype=float) * u.temperature_cgs
    t_s = gamma * p_cgs / ((gamma - 1.0) * n_cgs**2 * ISMCoolFn(T_cgs))
    return t_s / u.time_cgs


def emissivity(frame, params):
    """Cooling emissivity n^2 Lambda(T), ISM cooling, code units."""
    u = Units.from_params(params)
    n_cgs = np.asarray(frame["dens"], dtype=float) * u.number_density_cgs
    T_cgs = np.asarray(temp(frame, params), dtype=float) * u.temperature_cgs
    return n_cgs**2 * ISMCoolFn(T_cgs) / u.emissivity_cgs


# name -> (dimension, function(frame, params) returning code units)
DERIVED: dict[str, tuple[str, Callable]] = {
    "pres": ("pressure", pres),
    "temp": ("temperature", temp),
    "vmag": ("velocity", vmag),
    "bmag": ("magnetic", bmag),
    "beta": ("dimensionless", beta),
    "entropy": ("dimensionless", entropy),
    "t_cool": ("time", t_cool),
    "emissivity": ("emissivity", emissivity),
}


def available(frame) -> list[str]:
    """Raw variables of the frame plus every derived quantity."""
    return [*frame.fields, *DERIVED]


def dimension(name: str) -> str:
    if name in DERIVED:
        return DERIVED[name][0]
    return RAW_DIMENSIONS.get(name.split("/")[-1], "dimensionless")


def get(frame, name: str, params: dict, units: str = "code") -> np.ndarray:
    """Quantity ``name`` of ``frame`` in ``units``; ``params`` is the parsed athinput."""
    if name in frame.fields:
        data = frame[name]
    elif name in DERIVED:
        data = np.asarray(DERIVED[name][1](frame, params))
    else:
        raise KeyError(f"unknown quantity {name!r}; available: {available(frame)}")
    if units != "code":
        data = data * Units.from_params(params).factor(dimension(name), units)
    return data
