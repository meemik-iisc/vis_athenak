"""
Line emissivity tables (format "emission_table", version 1).

The tables are made outside vis_athenak (e.g. by ``cloudy_tables``); this
reader is the reference definition of the format.  One HDF5 file:

    attrs   format="emission_table", format_version=1, X_H, log_floor, ...
    axes/   log_nH (cm^-3), log_T (K): 1D, increasing, any spacing
            attrs: axis_order = ["log_nH", "log_T"]
    lines/  <name>: (n_nH, n_T) array of log10(eps / n_H^2) [erg cm^3 s^-1]
            attrs: wavelength_A, label, units

Values at or below ``log_floor`` mean eps = 0.  Outside the grid the table
is clamped to its edge, so eps / n_H^2 is constant there (eps ~ n_H^2 in n_H).

    table = EmissionTable("emis_KS19_z0.96_Z0.2.h5")
    eps = table.emissivity("Ha", nH, T)   # erg s^-1 cm^-3, nH in cm^-3, T in K
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import map_coordinates


class EmissionTable:
    FORMAT, VERSION = "emission_table", 1

    def __init__(self, path):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            if f.attrs.get("format") != self.FORMAT:
                raise ValueError(f"{path}: not an {self.FORMAT} file")
            version = int(f.attrs["format_version"])
            if version != self.VERSION:
                raise ValueError(f"{path}: format_version {version} unsupported "
                                 f"(this reader knows {self.VERSION})")
            axes = [str(a) for a in f["axes"].attrs["axis_order"]]
            if axes != ["log_nH", "log_T"]:
                raise ValueError(f"{path}: axes {axes}, version 1 needs [log_nH, log_T]")
            self.log_nH = f["axes/log_nH"][:]
            self.log_T = f["axes/log_T"][:]
            self.data = {k: f["lines"][k][:] for k in f["lines"]}
            self.wavelength = {k: float(f["lines"][k].attrs["wavelength_A"])
                               for k in f["lines"]}
            self.meta = {k: f.attrs[k] for k in f.attrs}
        for name, axis in (("log_nH", self.log_nH), ("log_T", self.log_T)):
            if axis.ndim != 1 or np.any(np.diff(axis) <= 0):
                raise ValueError(f"{path}: axis {name} is not 1D and increasing")
        shape = (self.log_nH.size, self.log_T.size)
        for k, v in self.data.items():
            if v.shape != shape:
                raise ValueError(f"{path}: line {k} has shape {v.shape}, axes give {shape}")
        self.X_H = float(self.meta["X_H"])
        self.log_floor = float(self.meta["log_floor"])

    @property
    def lines(self) -> list[str]:
        return list(self.data)

    def __repr__(self) -> str:
        return (f"EmissionTable({self.path.name}: {len(self.lines)} lines, "
                f"log nH {self.log_nH[0]:g}..{self.log_nH[-1]:g}, "
                f"log T {self.log_T[0]:g}..{self.log_T[-1]:g}, X_H={self.X_H:g})")

    def emissivity(self, line: str, nH, T):
        """
        Volume emissivity of ``line`` [erg s^-1 cm^-3] at hydrogen number
        density ``nH`` [cm^-3] and temperature ``T`` [K]; arrays of any
        (matching) shape.  Bilinear in (log nH, log T) on log(eps/nH^2).
        """
        return self.emissivities([line], nH, T)[line]

    def emissivities(self, lines: list[str], nH, T) -> dict[str, np.ndarray]:
        """`emissivity` of several lines, locating (nH, T) in the table once."""
        for line in lines:
            if line not in self.data:
                raise KeyError(f"no line {line!r} in {self.path.name}; it has {self.lines}")
        nH = np.asarray(nH, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        shape = np.broadcast_shapes(nH.shape, T.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            ln = np.log10(np.broadcast_to(nH, shape)).ravel()
            lt = np.log10(np.broadcast_to(T, shape)).ravel()
        # fractional indices (fine for uneven axes); np.interp clamps at the edges
        fi = np.interp(ln, self.log_nH, np.arange(self.log_nH.size))
        fj = np.interp(lt, self.log_T, np.arange(self.log_T.size))
        del ln, lt
        nH2 = (np.broadcast_to(nH, shape) ** 2).ravel()
        out = {}
        for line in lines:
            q = map_coordinates(self.data[line], [fi, fj], order=1, mode="nearest")
            out[line] = np.where(q > self.log_floor, 10.0**q * nH2, 0.0).reshape(shape)
        return out


@lru_cache(maxsize=None)
def load(path) -> EmissionTable:
    """EmissionTable of ``path``, read once per process."""
    return EmissionTable(Path(path).expanduser().resolve())


# ── Default table and line sums, used by the line_* quantities ───────────

_defaults: dict = {"path": None, "sums": None}


def configure(path=None, sums: dict | None = None) -> None:
    """
    Table and line sums used by `default_table` / `components` from now on in
    this process (None = config.EMISSION's).  Worker processes started with
    fork inherit it; pass the same values to workers started otherwise.
    """
    _defaults["path"] = path
    _defaults["sums"] = sums


def default_table() -> EmissionTable:
    """The table set by `configure`, else config.EMISSION["table"]."""
    if _defaults["path"] is not None:
        return load(_defaults["path"])
    config = _config()
    if not hasattr(config, "emission_table_path"):
        raise RuntimeError("no emission table: config.py has no EMISSION / "
                           "emission_table_path (needed for line_* quantities)")
    return load(config.emission_table_path())


def sums() -> dict[str, list[str]]:
    """Line name -> table lines summed for it (e.g. SII -> SII_6716, SII_6731)."""
    if _defaults["sums"] is not None:
        return _defaults["sums"]
    return getattr(_config(), "EMISSION", {}).get("sums", {})


def components(line: str) -> list[str]:
    """Table lines making up ``line``: its sum in `sums`, else the line itself."""
    return list(sums().get(line, [line]))


def line_emissivities(table: EmissionTable, lines: list[str], nH, T) -> dict[str, np.ndarray]:
    """Emissivity [erg s^-1 cm^-3] of each of ``lines`` (table lines or sums)."""
    needed = sorted({c for line in lines for c in components(line)})
    eps = table.emissivities(needed, nH, T)
    return {line: sum(eps[c] for c in components(line)) for line in lines}


def _config():
    if __package__ and "." in __package__:  # imported as vis_athenak.emission
        from .. import config
    else:
        import config
    return config
