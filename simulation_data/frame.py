"""A single, lazily loaded output snapshot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import athdf
from .field import Field
from .readers import (can_read_single_variable, read_coordinates, read_file, read_time,
                      read_variable_names)


class Frame:
    """
    One output snapshot, possibly spread over several files (one per output
    block).  Nothing is read from disk until a field, ``grid``, ``data``, or
    ``time`` is accessed.

    ``frame.fields`` maps each variable name to a lazily loaded `Field`,
    found from the file headers alone.  ``frame["dens"]`` is shorthand for
    ``frame.fields["dens"].data``, an array of shape ``(nx3, nx2, nx1)``.  The
    coordinate arrays ``x1f, x1v, x2f, ...`` and file attributes (``Time``,
    ``NumCycles``, ...) are in ``frame.grid`` and can be indexed the same way.

    From .athdf files only the requested fields are read.  A .bin file can
    only be read whole, so asking for one of its fields loads all of them,
    but the files of the other outputs in the frame are still not read.

    If two outputs hold a variable of the same name (``dens`` in both
    ``hydro_w`` and ``hydro_u``), the first output's is ``fields["dens"]`` and
    the other is ``fields["hydro_u/dens"]``.
    """

    def __init__(self, number: int, paths: dict[str, Path], dtype=None):
        self.number = number
        self.paths = paths  # {output id: path to that output's file}
        self._dtype = dtype
        self._fields: dict[str, Field] | None = None
        self._grid: dict | None = None
        self._time: float | None = None
        self._coordinates: str | None = None
        # Block layouts of .athdf files; kept by unload(), like the field list.
        self._layouts: dict[Path, athdf.AthdfLayout] = {}

    @property
    def time(self) -> float:
        """Simulation time in code units (read from the file header only)."""
        if self._time is None:
            if self._grid is not None:
                self._time = float(self._grid["Time"])
            else:
                self._time = read_time(next(iter(self.paths.values())))
        return self._time

    @property
    def coordinates(self) -> str:
        """Coordinate system: "cartesian", "cylindrical", "spherical_polar", ...
        (read from the file header only)."""
        if self._coordinates is None:
            self._coordinates = read_coordinates(next(iter(self.paths.values())))
        return self._coordinates

    # ── Fields ───────────────────────────────────────────────────────────

    @property
    def fields(self) -> dict[str, Field]:
        """Every variable of this frame, found from the file headers only."""
        if self._fields is None:
            fields = {}
            for output, path in self.paths.items():
                for name in read_variable_names(path):
                    key = name if name not in fields else f"{output}/{name}"
                    fields[key] = Field(name, output, path, self)
            self._fields = fields
        return self._fields

    def load(self, fields) -> None:
        """
        Read several fields (names or `Field`s) at once, one read per file,
        instead of one read per field.  Fields already loaded are skipped.
        """
        fields = [self.fields[f] if isinstance(f, str) else f for f in fields]
        by_path: dict[Path, list[Field]] = {}
        for field in fields:
            if not field.is_loaded:
                by_path.setdefault(field.path, []).append(field)

        for path, wanted in by_path.items():
            layout = self._athdf_layout(path)
            if layout is not None and layout.supported:
                # Fast path: no coordinates rebuilt, no grid stored.
                data = athdf.read_variables(path, [f.name for f in wanted], layout,
                                            self._dtype)
                for field in wanted:
                    field._data = data[field.name]
                continue
            if can_read_single_variable(path):
                data = read_file(path, self._dtype, quantities=[f.name for f in wanted])
            else:
                # The whole file is read anyway, so keep every field it holds.
                data = read_file(path, self._dtype)
                wanted = [f for f in self.fields.values() if f.path == path]
            for field in wanted:
                if not field.is_loaded:
                    field._data = data[field.name]
            self._keep_grid(data)

    def _athdf_layout(self, path: Path) -> athdf.AthdfLayout | None:
        """Block layout of an .athdf file, read once and cached; None otherwise."""
        if path.suffix != ".athdf":
            return None
        if path not in self._layouts:
            self._layouts[path] = athdf.read_layout(path)
        return self._layouts[path]

    @property
    def grid(self) -> dict:
        """Coordinates (``x1f``, ``x1v``, ...) and file attributes (``Time``, ...)."""
        if self._grid is None:
            path = next(iter(self.paths.values()))
            if can_read_single_variable(path):
                self._keep_grid(read_file(path, self._dtype, quantities=[]))
            else:
                self.load([next(f for f in self.fields.values() if f.path == path)])
        return self._grid

    def _keep_grid(self, data: dict) -> None:
        """Store the non-variable entries of a file read, if not stored yet."""
        if self._grid is None:
            names = {f.name for f in self.fields.values()}
            self._grid = {k: v for k, v in data.items() if k not in names}

    # ── Whole-frame access ───────────────────────────────────────────────

    @property
    def data(self) -> dict:
        """Every field and the grid in one dict; reads all of them."""
        self.load(self.fields.values())
        return {**self.grid, **{key: f.data for key, f in self.fields.items()}}

    @property
    def is_loaded(self) -> bool:
        """Whether any field is held in memory."""
        return self._fields is not None and any(f.is_loaded for f in self._fields.values())

    def unload(self) -> None:
        """Drop every loaded field and the grid; they are re-read on the next access."""
        if self._fields is not None:
            for field in self._fields.values():
                field.unload()
        self._grid = None

    def keys(self):
        return [*self.fields, *self.grid]

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self.fields:
            return self.fields[name].data
        try:
            return self.grid[name]
        except KeyError:
            raise KeyError(
                f"{name!r} not in frame {self.number}; fields: {sorted(self.fields)}, "
                f"grid: {sorted(self.grid)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self.fields or name in self.grid

    def __repr__(self) -> str:
        if self._fields is None:
            state = "fields not read"
        else:
            n = sum(f.is_loaded for f in self._fields.values())
            state = f"{n}/{len(self._fields)} fields loaded"
        return f"<Frame {self.number}  outputs={list(self.paths)}  {state}>"
