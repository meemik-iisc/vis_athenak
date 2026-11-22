"""
Split figure: one slice cut in two, a different variable on each side of the
cut, each with its own colorbar on its side (config.SPLIT).

    python main.py split                                  # config.SPLIT for config.RUN
    python main.py split --vars dens temp --frames 60 --xlim -10 3 --ylim -1.5 1.5
    python main.py split --cut vertical --at 0 --format png
    python main.py split --mirror                         # both show the first variable's half

A horizontal cut (along the plot's x axis, at y = --at) puts the first
variable above it and the second below, with their colorbars centred above
and below the plot; a vertical cut (along the plot's y axis, at x = --at)
puts the first on the left and the second on the right, colorbars likewise.
Each variable keeps its PLOT_VARS style (cmap, norm, vmin/vmax, units)
unless SPLIT["styles"] or --cmaps / --vmins / --vmaxs / --vcenters set it
for this figure; values below vmin take the style's ``under`` colour, and
``vcenter`` puts that value at the colormap's middle (diverging maps).  By default each
variable is drawn on its own true half of the slice; --mirror
instead shows the second variable on the first variable's half, reflected
across the cut (for flows symmetric about it).  The slice is the one of the
other plots (--axis / --position, as config.PLOT).  Images go to
``<outdir>/<name>.<axis>.<NNNNN>.<format>``, outdir defaulting to ``split/``
in the run's plot folder.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import ticker  # noqa: E402

if "." in __package__:  # imported as vis_athenak.plotting
    from .. import config
    from ..simulation_data import SimulationData
else:  # run from the repo root
    import config
    from simulation_data import SimulationData

from .quantities import available, dimension, has_quantity  # noqa: E402
from .slice2d import (  # noqa: E402
    apply_fonts, frame_plane, n_workers, non_negative_int, parse_frames, run_axis,
    slice_variable, time_title, unit_factor, var_settings,
)
from .slices import draw_slice  # noqa: E402


# ── Cutting the slice ────────────────────────────────────────────────────

def _centers(faces) -> np.ndarray:
    faces = np.asarray(faces, dtype=float)
    return 0.5 * (faces[1:] + faces[:-1])


def first_side(centers: np.ndarray, at: float, cut: str) -> np.ndarray:
    """Cells on the first variable's side: above a horizontal cut, left of a vertical one."""
    return centers >= at if cut == "horizontal" else centers < at


def reflect(data: np.ndarray, faces, at: float, axis: int) -> np.ma.MaskedArray:
    """
    ``data`` reflected across ``at`` along ``axis`` (0 = rows/y, 1 = columns/x):
    each cell gets the value of the cell holding its mirror point 2 at - c.
    Cells whose mirror point is outside the slice are masked.  Works on
    non-uniform grids too.
    """
    faces = np.asarray(faces, dtype=float)
    source = np.searchsorted(faces, 2.0 * at - _centers(faces), side="right") - 1
    valid = (source >= 0) & (source < len(faces) - 1)
    out = np.ma.masked_invalid(np.take(data, np.clip(source, 0, len(faces) - 2), axis=axis))
    invalid = ~valid[:, None] if axis == 0 else ~valid[None, :]
    return np.ma.masked_where(np.broadcast_to(invalid, out.shape), out)


def split_data(x, y, first: np.ndarray, second: np.ndarray, cut: str, at: float,
               mirror: bool) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    """
    The two arrays to draw: ``first`` kept on its side of the cut and
    ``second`` on the other (masked elsewhere).  With ``mirror``, ``second``
    is first reflected across the cut, so both show the first side's cells.
    """
    axis, faces = (0, y) if cut == "horizontal" else (1, x)
    keep = first_side(_centers(faces), at, cut)
    keep = keep[:, None] if axis == 0 else keep[None, :]
    if mirror:
        second = reflect(second, faces, at, axis)
    first = np.ma.masked_where(np.broadcast_to(~keep, first.shape), first)
    second = np.ma.masked_where(np.broadcast_to(keep, np.shape(second)), second)
    return first, second


def cut_position(at, lim, faces) -> float:
    """``at``, or the middle of the shown range (``lim``, else the slice) when None."""
    if at is not None:
        return float(at)
    lo, hi = lim if lim is not None else (faces[0], faces[-1])
    return 0.5 * (float(lo) + float(hi))


# ── Figure ───────────────────────────────────────────────────────────────



CBAR_THICKNESS = 0.1  # inches
CBAR_PAD = 0.08       # inches between a colorbar and the labels next to it


def _figure_bbox(fig, artist):
    """Tight bounding box of ``artist`` (with its labels), in figure coordinates."""
    renderer = fig.canvas.get_renderer()
    return artist.get_tightbbox(renderer).transformed(fig.transFigure.inverted())


def add_colorbars(fig, ax, meshes, labels, cut: str, length: float) -> None:
    """
    Each variable's colorbar centred on its own side of the plot, outside the
    axis labels: the first above and the second below for a horizontal cut,
    left and right for a vertical one; ticks and label on the outer side.
    Placed by hand from the drawn axes, since the equal aspect shrinks the
    axes after a layout engine would have placed them.
    """
    fig.canvas.draw()  # applies the aspect and lays out the tick labels
    box, tight = ax.get_position(), _figure_bbox(fig, ax)
    w_in, h_in = fig.get_size_inches()
    if cut == "horizontal":
        thick, pad = CBAR_THICKNESS / h_in, CBAR_PAD / h_in
        width = length * box.width
        x0 = box.x0 + 0.5 * (box.width - width)
        rects = [(x0, tight.y1 + pad, width, thick),
                 (x0, tight.y0 - pad - thick, width, thick)]
        sides, orientation = ("top", "bottom"), "horizontal"
    else:
        thick, pad = CBAR_THICKNESS / w_in, CBAR_PAD / w_in
        height = length * box.height
        y0 = box.y0 + 0.5 * (box.height - height)
        rects = [(tight.x0 - pad - thick, y0, thick, height),
                 (tight.x1 + pad, y0, thick, height)]
        sides, orientation = ("left", "right"), "vertical"
    for mesh, label, rect, side in zip(meshes, labels, rects, sides):
        cax = fig.add_axes(rect)
        cbar = fig.colorbar(mesh, cax=cax, orientation=orientation, label=label)
        axis = cbar.ax.xaxis if orientation == "horizontal" else cbar.ax.yaxis
        axis.set_ticks_position(side)
        axis.set_label_position(side)
        if isinstance(mesh.norm, (matplotlib.colors.LogNorm, LogCenterNorm)):
            log_ticks(axis, mesh.norm.vmin, mesh.norm.vmax)


def log_ticks(axis, vmin: float, vmax: float) -> None:
    """
    Labels of a log colorbar that do not overlap: 1, 2, 5 x 10^n over less
    than two decades (matplotlib would label every minor tick there), else
    the decades only; the other ticks are left unlabelled.
    """
    subs = (1.0, 2.0, 5.0) if np.log10(vmax / vmin) < 2 else (1.0,)
    decades = range(int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax))) + 1)
    ticks = [m * 10.0**e for e in decades for m in subs
             if vmin * (1 - 1e-9) <= m * 10.0**e <= vmax * (1 + 1e-9)]
    axis.set_major_locator(ticker.FixedLocator(ticks))
    axis.set_major_formatter(ticker.FuncFormatter(_sci_label))
    axis.set_minor_formatter(ticker.NullFormatter())


def add_title(fig, ax, text: str) -> None:
    """``text`` centred over ``ax``, above everything drawn so far."""
    fig.canvas.draw()
    top = max(_figure_bbox(fig, a).y1 for a in fig.axes)
    box = ax.get_position()
    fig.text(0.5 * (box.x0 + box.x1), top + CBAR_PAD / fig.get_figheight(), text,
             ha="center", va="bottom", fontsize=plt.rcParams["figure.titlesize"],
             fontweight=plt.rcParams["figure.titleweight"])


class LogCenterNorm(matplotlib.colors.Normalize):
    """
    Log scale with ``vcenter`` at the middle of the colormap: vmin..vcenter
    fills the lower half and vcenter..vmax the upper half, each evenly in
    log, however many decades each side spans (a log TwoSlopeNorm).
    Non-positive values are masked; values outside [vmin, vmax] map below 0
    or above 1 (the colormap's under/over colours).
    """

    def __init__(self, vmin: float, vcenter: float, vmax: float):
        if not 0 < vmin < vcenter < vmax:
            raise ValueError(f"need 0 < vmin < vcenter < vmax, got {vmin}, {vcenter}, {vmax}")
        super().__init__(vmin=vmin, vmax=vmax)
        self.vcenter = vcenter

    def _logs(self):
        return np.log10(self.vmin), np.log10(self.vcenter), np.log10(self.vmax)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        lo, c, hi = self._logs()
        log = np.ma.log10(np.ma.masked_less_equal(result, 0))
        out = np.ma.where(log < c, 0.5 * (log - lo) / (c - lo),
                          0.5 + 0.5 * (log - c) / (hi - c))
        return out[0] if is_scalar else out

    def inverse(self, value):
        lo, c, hi = self._logs()
        value = np.asarray(value, dtype=float)
        log = np.where(value < 0.5, lo + 2.0 * value * (c - lo),
                       c + (2.0 * value - 1.0) * (hi - c))
        return 10.0 ** log


def make_norm(var: dict):
    """
    The norm of ``var``: its "log"/None, or with ``vcenter`` set a centred one
    (log: `LogCenterNorm`; linear: TwoSlopeNorm), so the colormap's middle
    colour is ``vcenter``.
    """
    if var.get("vcenter") is None:
        return var["norm"]
    if var["vmin"] is None or var["vmax"] is None:
        raise ValueError("vcenter needs vmin and vmax")
    if var["norm"] == "log":
        return LogCenterNorm(var["vmin"], var["vcenter"], var["vmax"])
    return matplotlib.colors.TwoSlopeNorm(var["vcenter"], var["vmin"], var["vmax"])


def split_style(name: str, opts: dict) -> dict:
    """
    PLOT_VARS settings of ``name`` with SPLIT["styles"][name] and the
    command-line overrides applied; ``cmap`` becomes a Colormap whose values
    below vmin are drawn in ``under`` (if set) and masked cells (the other
    half) are transparent, and ``norm`` a centred norm if ``vcenter`` is set.
    """
    var = {**var_settings(name, {}), **opts["styles"].get(name, {})}
    i = opts["vars"].index(name)
    for key in ("cmaps", "vmins", "vmaxs", "vcenters"):
        if opts[key] is not None and opts[key][i] is not None:
            var[key[:-1]] = opts[key][i]
    cmap = plt.get_cmap(var["cmap"]).copy()
    cmap.set_bad(alpha=0.0)
    if var.get("under"):
        cmap.set_under(var["under"])
    var["cmap"] = cmap
    var["norm"] = make_norm(var)
    return var


def _sci_label(value: float, _pos=None) -> str:
    """5e-4 -> '$5\\times10^{-4}$', 1e-3 -> '$10^{-3}$', 10**5.5 -> '$10^{5.5}$'."""
    log = np.log10(value)
    if abs(2 * log - round(2 * log)) < 1e-6 and round(2 * log) % 2:  # a half decade
        return f"$10^{{{round(2 * log) / 2:g}}}$"
    exponent = int(np.floor(log + 1e-9))
    mantissa = round(value / 10.0**exponent, 6)
    if mantissa == 1:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:g}\\times10^{{{exponent}}}$"


def plot_split(frame, params: dict, basename: str, opts: dict) -> Path:
    """Draw and save the split figure of one frame; returns the file written."""
    apply_fonts()
    plane = frame_plane(frame, params, opts)
    names = opts["vars"]
    settings = [split_style(n, opts) for n in names]
    (x, y, data0), (_, _, data1) = (slice_variable(plane, v, params, opts) for v in settings)

    cut = opts["cut"]
    faces, lim = (y, opts["ylim"]) if cut == "horizontal" else (x, opts["xlim"])
    at = cut_position(opts["at"], lim, faces)
    halves = split_data(x, y, data0, data1, cut, at, opts["mirror"])

    fig, ax = plt.subplots(figsize=opts["figsize"])
    meshes = [
        draw_slice(ax, x, y, half, cmap=var["cmap"], norm=var["norm"],
                   vmin=var["vmin"], vmax=var["vmax"], colorbar=False,
                   axis=opts["axis"], length_units=opts["length_units"],
                   coordinates=frame.coordinates, in_plane=plane.in_plane)
        for half, var in zip(halves, settings)
    ]
    if opts["xlim"] is not None:
        ax.set_xlim(*opts["xlim"])
    if opts["ylim"] is not None:
        ax.set_ylim(*opts["ylim"])
    if opts["cut_line"]:
        (ax.axhline if cut == "horizontal" else ax.axvline)(at, **opts["cut_line"])

    add_colorbars(fig, ax, meshes, [l or v["label"] for l, v in
                                     zip(opts["labels"] or [None, None], settings)],
                  cut, opts["cbar_length"])
    if opts["title"]:
        add_title(fig, ax, time_title(frame, params, opts))

    outdir = Path(opts["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{opts['name']}.{opts['axis']}.{frame.number:05d}.{opts['format']}"
    fig.savefig(path, dpi=opts["dpi"], bbox_inches="tight")
    plt.close(fig)
    frame.unload()
    return path


def _plot_split_star(job):
    return plot_split(*job)


# ── Command line ─────────────────────────────────────────────────────────

def _optional_float(text: str):
    return None if text.lower() == "none" else float(text)


def main(argv=None):
    if not hasattr(config, "SPLIT"):
        raise SystemExit("main.py split: config.py has no SPLIT settings")
    sp, plot = config.SPLIT, config.PLOT
    parser = argparse.ArgumentParser(
        prog="main.py split",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=config.RUN, choices=list(config.SIMULATIONS),
                        help=f"simulation (default: {config.RUN})")
    parser.add_argument("--frames", nargs="+",
                        help="frame numbers and/or ranges like 0:21:5 (default: from --run)")
    parser.add_argument("--vars", nargs=2, default=sp["vars"], metavar=("FIRST", "SECOND"),
                        help="PLOT_VARS keys (or quantities): top and bottom for a horizontal "
                             f"cut, left and right for a vertical one (default: {sp['vars']})")
    parser.add_argument("--cut", choices=["horizontal", "vertical"], default=sp["cut"],
                        help=f"direction of the cut (default: {sp['cut']})")
    parser.add_argument("--at", type=_optional_float, default=sp["at"],
                        help="cut position along the plot's y (horizontal) or x (vertical) "
                             "axis in --length-units; none = middle of the shown range "
                             f"(default: {sp['at']})")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=sp["mirror"],
                        help="show the second variable on the first's half, reflected "
                             f"(default: {sp['mirror']})")
    parser.add_argument("--xlim", type=float, nargs=2, default=sp["xlim"],
                        help=f"range of the plot's x axis (default: {sp['xlim']} = full)")
    parser.add_argument("--ylim", type=float, nargs=2, default=sp["ylim"],
                        help=f"range of the plot's y axis (default: {sp['ylim']} = full)")
    parser.add_argument("--cmaps", nargs=2, metavar=("FIRST", "SECOND"),
                        help="colormaps of the two variables (default: SPLIT['styles'], "
                             "else PLOT_VARS); 'none' keeps one")
    parser.add_argument("--vmins", nargs=2, type=_optional_float, metavar=("FIRST", "SECOND"),
                        help="colour scale minima; 'none' keeps one")
    parser.add_argument("--vcenters", nargs=2, type=_optional_float, metavar=("FIRST", "SECOND"),
                        help="value at the colormap's middle (e.g. the ambient temperature), "
                             "for diverging maps; 'none' keeps one")
    parser.add_argument("--vmaxs", nargs=2, type=_optional_float, metavar=("FIRST", "SECOND"),
                        help="colour scale maxima; 'none' keeps one")
    parser.add_argument("--labels", nargs=2, default=sp.get("labels"),
                        help="colorbar labels of the two variables (default: their "
                             "PLOT_VARS labels)")
    parser.add_argument("--cbar-length", type=float, default=sp.get("cbar_length", 1.0),
                        help="colorbar length, fraction of the plot's side "
                             f"(default: {sp.get('cbar_length', 1.0)})")
    parser.add_argument("--cut-line", action=argparse.BooleanOptionalAction,
                        default=bool(sp["cut_line"]), help="draw a line along the cut")
    parser.add_argument("--title", action=argparse.BooleanOptionalAction,
                        default=sp.get("title", True), help="time as the title")
    parser.add_argument("--axis", choices=["x", "y", "z"],
                        help="slice normal (default: the run's axis in config.SIMULATIONS, "
                             f"else {plot['axis']})")
    parser.add_argument("--position", type=float, default=plot["position"],
                        help="slice position along --axis in --length-units (degrees for "
                             "an angle); default: midplane")
    parser.add_argument("--length-units", default=plot["length_units"],
                        help=f"units of the axes (default: {plot['length_units']})")
    parser.add_argument("--time-units", default=plot["time_units"],
                        help=f"units of the time in the title (default: {plot['time_units']})")
    parser.add_argument("--time-decimals", type=non_negative_int,
                        default=plot.get("time_decimals", 2))
    parser.add_argument("--figsize", type=float, nargs=2, default=sp["fig_size"],
                        help=f"inches (default: {sp['fig_size']})")
    parser.add_argument("--dpi", type=int, default=sp["dpi"])
    parser.add_argument("--format", default=sp["format"],
                        help=f"image format (default: {sp['format']})")
    parser.add_argument("--name", default=sp["name"], help="image file name prefix")
    parser.add_argument("--outdir", help="where images go (default: <run out>/split)")
    parser.add_argument("-w", "--workers", "-c", "--cores", dest="workers", type=int,
                        default=n_workers(),
                        help="parallel worker processes, 1 = serial "
                             f"(default: config.N_WORKERS = {n_workers()})")
    args = parser.parse_args(argv)
    args.axis = args.axis or run_axis(args.run)

    run = config.resolve(args.run)
    args.outdir = args.outdir or run["out"] / "split"
    sim = SimulationData(run["athinput"], run["data"], outputs=run["outputs"])
    wanted = parse_frames(args.frames)
    if wanted is None and run["frames"] is not None:
        wanted = set(run["frames"])
    frames = [f for f in sim.frames if wanted is None or f.number in wanted]
    if not frames:
        parser.error(f"no frames selected; run has frames {sim.frame_numbers}")

    unknown = [v for v in args.vars
               if not has_quantity(frames[0], config.plot_var(v)["quantity"])]
    if unknown:
        parser.error(f"unknown variables {unknown}; config.PLOT_VARS has "
                     f"{list(config.PLOT_VARS)}, quantities: {available(frames[0])}")
    try:  # Fail on a bad unit before spawning any work.
        unit_factor(sim.params, "length", args.length_units)
        unit_factor(sim.params, "time", args.time_units)
        for v in args.vars:
            var = config.plot_var(v)
            unit_factor(sim.params, dimension(var["quantity"]), var["units"])
    except ValueError as err:
        parser.error(str(err))

    opts = {k: getattr(args, k) for k in
            ("vars", "cmaps", "vmins", "vmaxs", "vcenters", "cut", "at", "mirror", "xlim", "ylim", "labels", "cbar_length", "title", "axis",
             "position", "length_units", "time_units", "time_decimals", "figsize", "dpi",
             "format", "name", "outdir")}
    opts["styles"] = sp.get("styles") or {}
    if args.cmaps:
        opts["cmaps"] = [None if c.lower() == "none" else c for c in args.cmaps]
    opts["cut_line"] = (sp["cut_line"] or dict(color="k", linewidth=0.5)) if args.cut_line else None

    jobs = [(f, sim.params, sim.basename, opts) for f in frames]
    print(f"{sim}\nsplit {args.vars[0]} | {args.vars[1]} ({args.cut} cut) "
          f"for {len(frames)} frames -> {args.outdir}")
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            for path in pool.map(_plot_split_star, jobs):
                print(f"  {path.name}")
    else:
        for job in jobs:
            print(f"  {plot_split(*job).name}")


if __name__ == "__main__":
    main()
