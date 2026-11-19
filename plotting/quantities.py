"""
Raw and derived quantities of a Frame, as arrays of shape ``(nx3, nx2, nx1)``.

    T = get(frame, "temp", sim.params, units="K")
    v = get(frame, "velx", sim.params, units="km/s")

Raw variables (``dens``, ``velx``, ``s_00``, ...) are read as stored; derived
ones are computed from them, in code units.  ``units`` converts a quantity
of known dimension (see ``utils.units.UNIT_NAMES``); "code" leaves it as is.
Derived quantities assume primitive variables: AthenaK's ``hydro_w``/``mhd_w``
(``eint`` is internal energy per volume) or Athena++'s ``prim`` (``press``).

AthenaK names work for Athena++ outputs too: ``dens``, ``velx``, ``bcc1``,
``s_00`` read Athena++'s ``rho``, ``vel1``, ``Bcc1``, ``r0`` (see `ALIASES`),
so the same ``PLOT_VARS`` serve both codes.  The names of the frame's own
variables always work as well.

``line_<name>`` is the volume emissivity of an emission line from the table in
config.EMISSION (see ``emission/table.py``): ``line_Ha``, ``line_OIII_5007``,
or a sum of lines from ``EMISSION["sums"]`` such as ``line_SII``; dimension
"emissivity", so units="erg/cm^3/s".
"""

from __future__ import annotations

from typing import Callable

import re

import numpy as np

if "." in __package__:  # imported as vis_athenak.plotting
    from ..emission import table as emission_table
    from ..utils.ism_cooling import ISMCoolFn
    from ..utils.units import Units
else:  # run from the repo root
    from emission import table as emission_table
    from utils.ism_cooling import ISMCoolFn
    from utils.units import Units

M_H_CGS = 1.6735575e-24  # g, hydrogen atom
LINE_PREFIX = "line_"    # line_<name>: emissivity of a table line or sum

# Dimension of the raw variables (AthenaK, then Athena++ names); any other
# variable is dimensionless.
RAW_DIMENSIONS = {
    "dens": "density",
    "velx": "velocity", "vely": "velocity", "velz": "velocity",
    "eint": "pressure",
    "bcc1": "magnetic", "bcc2": "magnetic", "bcc3": "magnetic",
    "rho": "density",
    "vel1": "velocity", "vel2": "velocity", "vel3": "velocity",
    "press": "pressure",
    "Bcc1": "magnetic", "Bcc2": "magnetic", "Bcc3": "magnetic",
}

# AthenaK name -> Athena++ name of the same primitive variable.  The velocity
# and field components are along x1, x2, x3 in both codes (R, phi, z in
# cylindrical coordinates).  Passive scalars s_NN -> rN are handled in `_raw`.
ALIASES = {
    "dens": "rho",
    "velx": "vel1", "vely": "vel2", "velz": "vel3",
    "bcc1": "Bcc1", "bcc2": "Bcc2", "bcc3": "Bcc3",
}


def _stored_name(frame, name: str) -> str | None:
    """Name under which raw variable `name` is stored in `frame`, or None."""
    if name in frame.fields:
        return name
    alias = ALIASES.get(name)
    m = re.fullmatch(r"s_(\d+)", name)
    if m:
        alias = f"r{int(m.group(1))}"
    if alias is not None and alias in frame.fields:
        return alias
    return None


def _raw(frame, name: str):
    """Raw variable `name` (AthenaK name, or as stored) of `frame`."""
    stored = _stored_name(frame, name)
    return frame[stored if stored is not None else name]


def _gamma(params) -> float:
    block = params.get("mhd") or params.get("hydro") or {}
    return float(block.get("gamma", 5.0 / 3.0))


def pres(frame, params):
    """Athena++ stores the pressure; AthenaK the internal energy density."""
    if "press" in frame.fields:
        return frame["press"]
    return (_gamma(params) - 1.0) * frame["eint"]


def temp(frame, params):
    """P/rho; in K with units="K", as AthenaK computes it for cooling."""
    return pres(frame, params) / _raw(frame, "dens")


def vmag(frame, params):
    return np.sqrt(_raw(frame, "velx")**2 + _raw(frame, "vely")**2 + _raw(frame, "velz")**2)


def _cylindrical(frame, name: str):
    """Velocity component `name` (velx/vely/velz = v_R/v_phi/v_z) of a cylindrical run."""
    if frame.coordinates != "cylindrical":
        raise ValueError(f"v_R and v_phi need cylindrical coordinates, not "
                         f"{frame.coordinates!r}; use velx/vely")
    return _raw(frame, name)


def velR(frame, params):
    """Cylindrical radial velocity v_R (x1 component)."""
    return _cylindrical(frame, "velx")


def velR_signed(frame, params):
    """v_R signed along the vertical axis of a meridional (R-z) slice: positive
    upward on both halves, like vely of a Cartesian x-y slice.  Equals v_R
    elsewhere (all R >= 0)."""
    return getattr(frame, "side", 1.0) * _cylindrical(frame, "velx")


def velphi(frame, params):
    """Cylindrical azimuthal velocity v_phi (x2 component)."""
    return _cylindrical(frame, "vely")


def bmag(frame, params):
    return np.sqrt(_raw(frame, "bcc1")**2 + _raw(frame, "bcc2")**2 + _raw(frame, "bcc3")**2)


def beta(frame, params):
    """Plasma beta, P_gas / (B^2/2)."""
    return pres(frame, params) / (0.5 * bmag(frame, params)**2)


def entropy(frame, params):
    """ln P - gamma ln rho, in code units; NaN where P or rho <= 0."""
    p = np.asarray(pres(frame, params), dtype=float)
    rho = np.asarray(_raw(frame, "dens"), dtype=float)
    out = np.full_like(rho, np.nan)
    ok = (p > 0) & (rho > 0)
    out[ok] = np.log(p[ok]) - _gamma(params) * np.log(rho[ok])
    return out


def t_cool(frame, params):
    """Isobaric cooling time gamma P / ((gamma-1) n^2 Lambda(T)), ISM cooling, code units."""
    u = Units.from_params(params)
    gamma = _gamma(params)
    p_cgs = np.asarray(pres(frame, params), dtype=float) * u.pressure_cgs
    n_cgs = np.asarray(_raw(frame, "dens"), dtype=float) * u.number_density_cgs
    T_cgs = np.asarray(temp(frame, params), dtype=float) * u.temperature_cgs
    t_s = gamma * p_cgs / ((gamma - 1.0) * n_cgs**2 * ISMCoolFn(T_cgs))
    return t_s / u.time_cgs


def emissivity(frame, params):
    """Cooling emissivity n^2 Lambda(T), ISM cooling, code units."""
    u = Units.from_params(params)
    n_cgs = np.asarray(_raw(frame, "dens"), dtype=float) * u.number_density_cgs
    T_cgs = np.asarray(temp(frame, params), dtype=float) * u.temperature_cgs
    return n_cgs**2 * ISMCoolFn(T_cgs) / u.emissivity_cgs


def hydrogen_density(frame, params, X_H: float):
    """n_H = X_H rho / m_H in cm^-3."""
    u = Units.from_params(params)
    return X_H * np.asarray(_raw(frame, "dens"), dtype=float) * (u.density_cgs / M_H_CGS)


def line_emissivity(frame, params, line: str):
    """
    Volume emissivity of emission line ``line`` (a line of the emission table,
    or a sum of them, see emission.table.components), code units; from the
    table's eps/n_H^2 at n_H = X_H rho / m_H (X_H of the table) and T.
    """
    table = emission_table.default_table()
    u = Units.from_params(params)
    nH = hydrogen_density(frame, params, table.X_H)
    T = np.asarray(temp(frame, params), dtype=float) * u.temperature_cgs
    eps = emission_table.line_emissivities(table, [line], nH, T)[line]
    return eps / u.emissivity_cgs


def _line_name(name: str) -> str | None:
    """'Ha' for 'line_Ha'; None for any other name."""
    return name[len(LINE_PREFIX):] if name.startswith(LINE_PREFIX) else None


# name -> (dimension, function(frame, params) returning code units)
DERIVED: dict[str, tuple[str, Callable]] = {
    "pres": ("pressure", pres),
    "temp": ("temperature", temp),
    "vmag": ("velocity", vmag),
    "velR": ("velocity", velR),
    "velR_signed": ("velocity", velR_signed),
    "velphi": ("velocity", velphi),
    "bmag": ("magnetic", bmag),
    "beta": ("dimensionless", beta),
    "entropy": ("dimensionless", entropy),
    "t_cool": ("time", t_cool),
    "emissivity": ("emissivity", emissivity),
}


def available(frame) -> list[str]:
    """Raw variables of the frame, the AthenaK names they alias, and every
    derived quantity."""
    stored = list(frame.fields)
    aliased = [a for a in ALIASES if a not in stored and ALIASES[a] in stored]
    aliased += [f"s_{int(n[1:]):02d}" for n in stored if re.fullmatch(r"r\d+", n)]
    try:  # line emissivities, if an emission table is set up
        table = emission_table.default_table()
        lines = [*table.lines, *emission_table.sums()]
    except (RuntimeError, OSError):
        lines = []
    return [*stored, *aliased, *DERIVED, *(LINE_PREFIX + n for n in lines)]


def dimension(name: str) -> str:
    if name in DERIVED:
        return DERIVED[name][0]
    if _line_name(name) is not None:
        return "emissivity"
    return RAW_DIMENSIONS.get(name.split("/")[-1], "dimensionless")


def get(frame, name: str, params: dict, units: str = "code") -> np.ndarray:
    """Quantity ``name`` of ``frame`` in ``units``; ``params`` is the parsed athinput."""
    stored = _stored_name(frame, name)
    if stored is not None:
        data = frame[stored]
    elif name in DERIVED:
        data = np.asarray(DERIVED[name][1](frame, params))
    elif _line_name(name) is not None:
        data = line_emissivity(frame, params, _line_name(name))
    else:
        raise KeyError(f"unknown quantity {name!r}; available: {available(frame)}")
    if units != "code":
        data = data * Units.from_params(params).factor(dimension(name), units)
    return data
