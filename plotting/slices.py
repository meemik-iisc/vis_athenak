"""
2D slices: cutting a slice out of a frame, and drawing it on an axes.

None of these use global settings, so they can be combined freely, e.g. one
panel per run when comparing cooling prescriptions:

    fig, axes = plt.subplots(1, 2)
    for ax, frame in zip(axes, frames):
        plane = Plane(frame, axis="z")                 # midplane
        T = get(plane, "temp", params, units="K")      # computed on the 2D slice only
        draw_slice(ax, plane.x, plane.y, T, cmap="inferno", norm="log", vmin=1e4, vmax=1e7)

`Plane` (``simulation_data.plane.FramePlane``) reads each raw field's slice
the first time it is used, from only the meshblocks crossing the slice, so
neither reading nor derived quantities touch the rest of the box.

For cylindrical runs, `MeridionalPlane` joins the slices at phi and phi + pi
into the full R-z plane, drawn with z across and R from -R to +R upwards.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colors

if "." in __package__:  # imported as vis_athenak.plotting
    from ..simulation_data.plane import FramePlane as Plane
else:  # run from the repo root
    from simulation_data.plane import FramePlane as Plane

__all__ = ["ANGULAR", "AXIS_LABELS", "COORDINATE_LABELS", "MeridionalPlane", "Plane",
           "draw_slice", "is_angular", "slice_index", "take_slice"]

# Slice normal -> (array axis cut, horizontal coordinate, vertical coordinate).
# Arrays are (nx3, nx2, nx1), so axis 0 is x3 (z).
_GEOMETRY = {
    "z": (0, "x1", "x2"),
    "y": (1, "x1", "x3"),
    "x": (2, "x2", "x3"),
}
AXIS_LABELS = {"x1": "x", "x2": "y", "x3": "z"}

# Axis labels of each coordinate system (the "Coordinates" of the output).
COORDINATE_LABELS = {
    "cartesian": AXIS_LABELS,
    "cylindrical": {"x1": "R", "x2": r"$\phi$", "x3": "z"},
    "spherical_polar": {"x1": "r", "x2": r"$\theta$", "x3": r"$\phi$"},
}
# Coordinates that are angles (radians in the files), not lengths.
ANGULAR = {"cylindrical": {"x2"}, "spherical_polar": {"x2", "x3"}}


def is_angular(coordinates: str, coord: str) -> bool:
    """Whether ``coord`` ("x1", "x2", "x3") is an angle in ``coordinates``."""
    return coord in ANGULAR.get(coordinates, set())


def normal_coordinate(axis: str) -> str:
    """The coordinate ("x1", "x2", "x3") held fixed by a slice normal to ``axis``."""
    _check_axis(axis)
    return {0: "x3", 1: "x2", 2: "x1"}[_GEOMETRY[axis][0]]


def _check_axis(axis: str) -> None:
    if axis not in _GEOMETRY:
        raise ValueError(f"axis must be one of {list(_GEOMETRY)}, got {axis!r}")


def slice_index(frame, axis: str = "z", position: float | None = None) -> int:
    """
    Index along ``axis`` of the cell containing ``position`` (code units);
    the middle cell if ``position`` is None.
    """
    _check_axis(axis)
    normal = {0: "x3", 1: "x2", 2: "x1"}[_GEOMETRY[axis][0]]
    faces = frame[f"{normal}f"]
    ncells = len(faces) - 1
    if position is None:
        return ncells // 2
    if not faces[0] <= position <= faces[-1]:
        raise ValueError(
            f"{AXIS_LABELS[normal]} = {position} outside [{faces[0]}, {faces[-1]}]"
        )
    return min(int(np.searchsorted(faces, position, side="right")) - 1, ncells - 1)


def take_slice(frame, data: np.ndarray, axis: str = "z", position: float | None = None):
    """
    Slice an already computed array ``data`` (shape ``(nx3, nx2, nx1)``)
    normal to ``axis``; see `Plane` to slice before computing instead.

    ``position`` is a coordinate in code units along ``axis``; the cell
    containing it is used.  Defaults to the middle of the domain.  For 2D
    runs use ``axis="z"``.

    Returns ``(x_faces, y_faces, slice)`` ready for `draw_slice`, with the
    slice of shape ``(len(y_faces) - 1, len(x_faces) - 1)``.
    """
    _check_axis(axis)
    cut, h, v = _GEOMETRY[axis]
    index = slice_index(frame, axis, position)
    plane = np.take(data, index, axis=cut)  # rows follow v, columns follow h
    return frame[f"{h}f"], frame[f"{v}f"], plane


class MeridionalPlane:
    """
    The R-z plane of a cylindrical run through the axis: the slice at phi
    (R > 0) and the one at phi + pi (mirrored to R < 0), transposed so that
    z runs along the horizontal axis and R along the vertical one.

    Works like a `Plane` for `plotting.quantities.get`: ``plane[name]`` is
    the joined 2D array of shape ``(len(y) - 1, len(x) - 1)``.  Values are
    as stored on both halves, so v_R > 0 points away from the axis on each.
    If phi + pi is not in the domain (a wedge, or a 2D run with one phi
    cell), the slice at phi is mirrored onto itself, i.e. axisymmetry.

    ``position`` is phi in radians; None = the middle phi cell.
    """

    def __init__(self, frame, position: float | None = None):
        self.frame = frame
        self.upper = Plane(frame, axis="y", position=position)
        faces = self.upper.grid["x2f"]
        phi = float(self.upper.grid["x2v"][self.upper.index])
        opposite = (phi + np.pi) % (2.0 * np.pi)
        if faces[0] <= opposite <= faces[-1] and len(faces) > 2:
            self.lower = Plane(frame, axis="y", position=opposite)
        else:
            self.lower = self.upper
        self.phi = phi
        # R faces from -R_max to R_max; with an inner hole (R_min > 0) one
        # masked row fills the gap between -R_min and R_min.
        r = np.asarray(self.upper.grid["x1f"])
        self._gap = bool(r[0] > 0)
        self._y = np.concatenate([-r[::-1], r if self._gap else r[1:]])

    @property
    def coordinates(self) -> str:
        return self.frame.coordinates

    @property
    def in_plane(self) -> tuple[str, str]:
        return "x3", "x1"

    @property
    def x(self) -> np.ndarray:
        """z faces."""
        return self.upper.grid["x3f"]

    @property
    def y(self) -> np.ndarray:
        """R faces, from -R_max to R_max."""
        return self._y

    @property
    def fields(self):
        return self.frame.fields

    @property
    def time(self) -> float:
        return self.frame.time

    @property
    def number(self) -> int:
        return self.frame.number

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self.frame.fields:
            return self.upper[name]  # coordinates and attributes
        upper = self.upper[name].T            # (nR, nz), R increasing
        lower = self.lower[name].T[::-1]      # R decreasing, i.e. -R increasing
        if self._gap:
            gap = np.full((1, upper.shape[1]), np.nan, dtype=upper.dtype)
            return np.concatenate([lower, gap, upper])
        return np.concatenate([lower, upper])

    def __contains__(self, name: str) -> bool:
        return name in self.upper

    def unload(self) -> None:
        self.upper.unload()
        self.lower.unload()

    def __repr__(self) -> str:
        return (f"<MeridionalPlane phi = {np.degrees(self.phi):.1f} deg of frame "
                f"{self.frame.number}>")


def _uniform(faces) -> bool:
    widths = np.diff(np.asarray(faces, dtype=float))
    return widths.size > 0 and np.allclose(widths, widths[0], rtol=1e-6, atol=0)


def draw_slice(ax, x, y, data, cmap="viridis", norm=None, vmin=None, vmax=None,
               colorbar=True, label=None, axis="z", length_units=None,
               coordinates="cartesian", in_plane=None):
    """
    Draw a slice on ``ax``; ``x`` and ``y`` are cell faces (as from `Plane`
    or `take_slice`).  Returns the image, for a colorbar.

    ``norm`` is "log" (non-positive values masked) or None/"linear".
    ``length_units`` is only used in the axis labels; scale ``x`` and ``y``
    to match (angles stay in radians).  ``coordinates`` (``frame.coordinates``)
    picks the axis labels, e.g. R and z for a cylindrical slice at fixed phi.
    ``in_plane`` (``plane.in_plane``) gives the (horizontal, vertical)
    coordinates when they differ from ``axis``'s, as for `MeridionalPlane`.
    Uniform grids are drawn with imshow (one image, fast); other grids with
    pcolormesh (one quad per cell).
    """
    if norm == "log":
        data = np.ma.masked_less_equal(data, 0)
        mpl_norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    elif norm in (None, "linear"):
        mpl_norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        raise ValueError(f"norm must be 'log' or None, got {norm!r}")

    h, v = in_plane or _GEOMETRY[axis][1:]
    # Equal aspect only when both axes are lengths; an angle against a length is not.
    aspect = "auto" if is_angular(coordinates, h) or is_angular(coordinates, v) else "equal"
    if _uniform(x) and _uniform(y):
        mesh = ax.imshow(data, origin="lower", extent=(x[0], x[-1], y[0], y[-1]),
                         cmap=cmap, norm=mpl_norm, aspect=aspect, interpolation="nearest")
    else:
        mesh = ax.pcolormesh(x, y, data, cmap=cmap, norm=mpl_norm, shading="flat")
        ax.set_aspect(aspect)
    labels = COORDINATE_LABELS.get(coordinates, AXIS_LABELS)
    length = f" [{length_units}]" if length_units and length_units != "code" else ""
    for coord, set_label in ((h, ax.set_xlabel), (v, ax.set_ylabel)):
        set_label(labels[coord] + (" [rad]" if is_angular(coordinates, coord) else length))
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=label)
    return mesh
