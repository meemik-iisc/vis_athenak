"""
Combined figure: a grid of slice panels per frame, laid out by config.COMBINED.

    python main.py combined                          # config.COMBINED["layout"]
    python main.py combined --frames 0:21:5 -w 4 --video
    python main.py combined --layout "dens,temp;pres,t_cool"
    python main.py combined --layout "cbox_02Myr:temp,cbox_tabcool:temp"
    python main.py combined --layout "dens;split:temp:tracer;vely" --colorbars horizontal

Each cell of the layout is a variable (plotted for config.RUN, or --run), a
(run, variable) pair, a split panel ("split", first, second) or
("split", run, first, second), or None for an empty panel; on the command
line rows are separated by ";", cells by ",", a run is given as "run:var",
a split panel as "split:first:second" or "run:split:first:second", and "-"
is empty.  A split panel is one slice cut in two as in ``main.py split``
(config.SPLIT: cut, at, mirror, cut_line, styles, labels), with each
variable's colorbar on its own side (above and below for a horizontal cut).
Variables are styled by config.PLOT_VARS.  With colorbars "auto"
(COMBINED["colorbars"] / --colorbars), if every row holds one variable,
each row shares a colorbar; else if every column does, each column does;
otherwise each panel has its own.  With "horizontal", every panel has its
own horizontal colorbar below it, labelled with the variable (no panel
titles).  Frames are matched by frame number across
runs.  Each run is sliced normal to its ``axis`` in config.SIMULATIONS if it
has one, else to --axis (e.g. a cylindrical R-z plane next to a 2D Cartesian
x-y run).  Images go to ``<outdir>/<name>.<axis>.<NNNNN>.<format>``, outdir
defaulting to ``combined/`` in the plot folder of the layout's first run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if "." in __package__:  # imported as vis_athenak.plotting
    from .. import config
    from ..simulation_data import SimulationData
else:  # run from the repo root
    import config
    from simulation_data import SimulationData

from .quantities import available, has_quantity, dimension  # noqa: E402
from .slice2d import (  # noqa: E402
    apply_fonts, frame_plane, n_workers, non_negative_int, parse_frames, slice_variable, time_title,
    unit_factor,
)
from .slices import draw_slice, is_angular, set_fixed_ticks  # noqa: E402
from .split import (  # noqa: E402
    CBAR_PAD, CBAR_THICKNESS, LogCenterNorm, _figure_bbox, cut_position, log_ticks,
    split_data, split_style,
)


CBAR_ROOM = 0.6  # inches of figure height added per horizontal colorbar
SPAN_GAP = 0.04  # gap between bars sharing a side (fraction of the panel's side)
BOTTOM_BAR_GAP = 0.8  # inches between the colorbars of a bottom row
RIGHT_BAR_GAP = 0.25  # inches between a right-hand colorbar's label and the next bar
# Margins (inches) of panels stacked by hand (hspace set); trimmed at savefig.
MARGIN_LEFT, MARGIN_BOTTOM, MARGIN_TOP, MARGIN_TITLE = 1.2, 0.8, 0.15, 0.6
MARGIN_RIGHT, MARGIN_RIGHT_BARS = 0.3, 3.0
TITLE_GAP = 0.1  # inches between the panels and the time above them


# ── Layout ───────────────────────────────────────────────────────────────

def is_split(name) -> bool:
    """Whether a normalized cell's variable is a split panel ("split", first, second)."""
    return isinstance(name, tuple)


def cell_variables(name) -> list[str]:
    """Variables drawn in a cell: the variable, or a split panel's two."""
    return list(name[1:]) if is_split(name) else [name]


def normalize_layout(layout, default_run: str) -> list[list[tuple | None]]:
    """
    Layout cells -> (run, variable), (run, ("split", first, second)) or None;
    rows padded to equal length.
    """
    rows = []
    for row in layout:
        cells = []
        for cell in row:
            if cell is None:
                cells.append(None)
            elif isinstance(cell, str):
                cells.append((default_run, cell))
            elif cell[0] == "split":
                if len(cell) == 3:
                    cells.append((default_run, ("split", cell[1], cell[2])))
                elif len(cell) == 4:
                    cells.append((cell[1], ("split", cell[2], cell[3])))
                else:
                    raise ValueError(f"a split cell is (\"split\", [run,] first, second), "
                                     f"got {cell!r}")
            else:
                run, var = cell
                cells.append((run, var))
        rows.append(cells)
    ncols = max((len(r) for r in rows), default=0)
    return [r + [None] * (ncols - len(r)) for r in rows]


def parse_layout(text: str) -> list[list]:
    """
    'dens,temp;run:pres,-' -> [["dens", "temp"], [("run", "pres"), None]];
    'split:temp:tracer' -> ("split", "temp", "tracer"), and
    'run:split:temp:tracer' -> ("split", "run", "temp", "tracer").
    """
    rows = []
    for row in text.split(";"):
        cells = []
        for cell in row.split(","):
            cell = cell.strip()
            parts = cell.split(":")
            if cell in ("", "-"):
                cells.append(None)
            elif "split" in parts[:2]:
                if parts[0] == "split" and len(parts) == 3:
                    cells.append(tuple(parts))
                elif parts[1] == "split" and len(parts) == 4:
                    cells.append(("split", parts[0], *parts[2:]))
                else:
                    raise argparse.ArgumentTypeError(
                        f"split cell must be split:first:second or run:split:first:second, "
                        f"got {cell!r}")
            elif ":" in cell:
                cells.append(tuple(cell.split(":", 1)))
            else:
                cells.append(cell)
        rows.append(cells)
    return rows


def layout_runs(layout) -> list[str]:
    """Runs used in a normalized layout, in order of first appearance."""
    return list(dict.fromkeys(c[0] for row in layout for c in row if c is not None))


def default_outdir(layout) -> Path:
    return config.resolve(layout_runs(layout)[0])["out"] / "combined"


def _uniform(cells) -> str | None:
    """The variable if all non-empty cells (two or more) hold the same one."""
    if any(c is not None and is_split(c[1]) for c in cells):
        return None
    names = {c[1] for c in cells if c is not None}
    filled = sum(c is not None for c in cells)
    return names.pop() if len(names) == 1 and filled > 1 else None


# ── Drawing ──────────────────────────────────────────────────────────────

def shared_limits(planes, var: dict) -> tuple:
    """vmin/vmax for panels of one variable: configured, else their joint range."""
    vmin, vmax = var["vmin"], var["vmax"]
    if vmin is None or vmax is None:
        values = np.concatenate([np.asarray(p, dtype=float).ravel() for p in planes])
        values = values[np.isfinite(values)]
        if var["norm"] == "log":
            values = values[values > 0]
        if values.size:
            vmin = values.min() if vmin is None else vmin
            vmax = values.max() if vmax is None else vmax
    return vmin, vmax


def _shown(faces, lim) -> tuple[float, float]:
    """Range of ``faces`` shown with axis limits ``lim`` ((min, max), either may be None)."""
    lo, hi = float(faces[0]), float(faces[-1])
    if lim is not None:
        lo = lo if lim[0] is None else float(lim[0])
        hi = hi if lim[1] is None else float(lim[1])
    return lo, hi


def plot_combined(layout, frames: dict, params: dict, number: int, opts) -> Path:
    """
    Draw frame ``number`` for a normalized ``layout``; ``frames`` and ``params``
    map each run to its Frame and athinput params.  Returns the file written.
    """
    apply_fonts()
    nrows, ncols = len(layout), len(layout[0])
    pw, ph = opts["panel_size"]
    bars_mode = opts.get("colorbars", "auto")
    horizontal, vertical = bars_mode == "horizontal", bars_mode == "vertical"
    bottom_row, right_column = bars_mode == "bottom", bars_mode == "right"
    own_bars = horizontal or vertical or bottom_row or right_column  # one bar per variable
    split_opts = opts.get("split") or {}
    styles = opts.get("styles") or {}
    xlim, ylim, hspace = opts.get("xlim"), opts.get("ylim"), opts.get("hspace")

    def style(name, extra=None):
        # PLOT_VARS, then the split styles (split panels), then COMBINED["styles"].
        sopts = dict(styles={name: {**(extra or {}).get(name, {}), **styles.get(name, {})}},
                     vars=[name], cmaps=None, vmins=None, vmaxs=None, vcenters=None)
        return split_style(name, sopts)

    # Slice every panel first, so a variable shares its colour range across panels.
    cells = [(r, c, run, name) for r, row in enumerate(layout)
             for c, cell in enumerate(row) if cell is not None for run, name in [cell]]
    plain = [cell for cell in cells if not is_split(cell[3])]
    split_cells = [cell for cell in cells if is_split(cell[3])]
    settings = {name: style(name) for _, _, _, name in plain}
    axes_of = opts.get("run_axes", {})
    planes = {run: frame_plane(frames[run], params[run], opts, axes_of.get(run))
              for run in frames}
    slices = {(r, c): slice_variable(planes[run], settings[name], params[run], opts)
              for r, c, run, name in plain}
    limits = {name: shared_limits([slices[r, c][2] for r, c, _, n in plain if n == name],
                                  var)
              for name, var in settings.items()}
    split_styles = {(r, c): [style(v, split_opts.get("styles")) for v in name[1:]]
                    for r, c, _, name in split_cells}
    for r, c, run, name in split_cells:
        (x, y, d0), (_, _, d1) = (slice_variable(planes[run], v, params[run], opts)
                                  for v in split_styles[r, c])
        slices[r, c] = (x, y, (d0, d1))

    # Figure: constrained layout, or (hspace set) panels stacked by hand with
    # gaps of hspace x the panel height, the figure as tall as the panels'
    # aspect needs, so equal-aspect panels fill their slots without gaps.
    sharex = opts.get("sharex", False)
    if hspace is None:
        split_bars = 2 if split_opts.get("cut", "horizontal") == "horizontal" and not vertical else 0

        def n_bars(cell):  # horizontal colorbars above/below a cell
            if cell is None:
                return 0
            return split_bars if is_split(cell[1]) else int(horizontal)

        # Room for the horizontal colorbars, so they do not take it from the panels.
        extra = sum(max(n_bars(cell) for cell in row) for row in layout) * CBAR_ROOM
        title_at = {}  # suptitle where the constrained layout puts it
        fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, ph * nrows + extra),
                                 squeeze=False, sharex=sharex, layout="constrained")
    else:
        if horizontal:
            raise ValueError("hspace (panels stacked by hand) needs colorbars "
                             "'right', 'bottom', 'vertical' or 'auto'")
        ratios = []
        for row in range(nrows):
            ratio = []
            for col in range(ncols):
                if layout[row][col] is None:
                    continue
                run = layout[row][col][0]
                x, y, _ = slices[row, col]
                h, v = planes[run].in_plane
                coords = frames[run].coordinates
                if is_angular(coords, h) or is_angular(coords, v):
                    ratio.append(ph / pw)
                else:
                    (x0, x1), (y0, y1) = _shown(x, xlim), _shown(y, ylim)
                    ratio.append(abs(y1 - y0) / abs(x1 - x0))
            ratios.append(max(ratio, default=ph / pw))
        left, bottom = MARGIN_LEFT, MARGIN_BOTTOM + (CBAR_ROOM if bottom_row else 0.0)
        bar_room = opts.get("cbar_room") or MARGIN_RIGHT_BARS  # inches, bar + its label
        right = bar_room if vertical or right_column else MARGIN_RIGHT
        top = MARGIN_TOP + (MARGIN_TITLE if opts.get("title", True) else 0.0)
        gap = right + 0.3 if vertical else 0.3  # between columns, inches
        width = pw * ncols
        aw = (width - left - right - (ncols - 1) * gap) / ncols
        heights = [aw * q for q in ratios]
        height = top + bottom + sum(heights) + (nrows - 1) * hspace * np.mean(heights)
        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(nrows, ncols, left=left / width, right=1 - right / width,
                              bottom=bottom / height, top=1 - top / height,
                              hspace=hspace, wspace=gap / aw, height_ratios=ratios)
        axes = gs.subplots(sharex=sharex, squeeze=False)
        # Suptitle just above the panels, not at the top of the generous margin.
        title_at = dict(y=1 - (top - TITLE_GAP) / height, va="bottom")

    row_bars = not own_bars and all(_uniform(row) for row in layout if any(row))
    col_bars = (not own_bars and not row_bars
                and all(_uniform(col) for col in zip(*layout) if any(col)))
    runs = layout_runs(layout)
    run_per_column = all(len({c[0] for c in col if c is not None}) <= 1
                         for col in zip(*layout))
    top_row = {c: min((r for r in range(nrows) if layout[r][c] is not None), default=None)
               for c in range(ncols)}

    def finish_panel(r, c, ax):
        if xlim is not None:
            ax.set_xlim(*_shown(ax.get_xlim(), xlim))
        if ylim is not None:
            ax.set_ylim(*_shown(ax.get_ylim(), ylim))
        if r + 1 < nrows and layout[r + 1][c] is not None:
            ax.set_xlabel("")
        if c > 0 and layout[r][c - 1] is not None:
            ax.set_ylabel("")
        if hspace is not None and r != nrows // 2:
            ax.set_ylabel("")  # stacked by hand: one y label, on the middle row

    def panel_title(r, c, run, label):
        # The variable names the panel (unless its colorbar does); with several
        # runs, the run does too (once per column if each column is one run).
        title = label or ""
        if len(runs) > 1 and (not run_per_column or r == top_row[c]):
            title = f"{run}\n{title}" if title else run
        return title

    # side_bars: (ax, mesh, label, location, span along the panel, ticks), placed below
    meshes, side_bars = {}, []
    own_side = "bottom" if horizontal else "right" if vertical else "row"
    for r, c, run, name in plain:
        var, ax = settings[name], axes[r, c]
        x, y, plane = slices[r, c]
        vmin, vmax = limits[name]
        own_bar = not (row_bars or col_bars or own_bars)
        meshes[r, c] = draw_slice(ax, x, y, plane, cmap=var["cmap"], norm=var["norm"],
                                  vmin=vmin, vmax=vmax, colorbar=own_bar,
                                  linthresh=var.get("linthresh"),
                                  axis=axes_of.get(run, opts["axis"]),
                                  length_units=opts["length_units"],
                                  coordinates=frames[run].coordinates,
                                  in_plane=planes[run].in_plane)
        finish_panel(r, c, ax)
        if own_bars:
            side_bars.append((ax, meshes[r, c], var["label"], own_side, (0.0, 1.0),
                              var.get("ticks")))
        ax.set_title(panel_title(r, c, run, None if own_bars else var["label"]))

    for r, c, run, name in split_cells:
        ax, sp, vars_ = axes[r, c], split_opts, split_styles[r, c]
        x, y, (data0, data1) = slices[r, c]
        cut = sp["cut"]
        at = cut_position(sp["at"], None, y if cut == "horizontal" else x)
        halves = split_data(x, y, data0, data1, cut, at, sp["mirror"])
        halves_meshes = [
            draw_slice(ax, x, y, half, cmap=var["cmap"], norm=var["norm"],
                       vmin=var["vmin"], vmax=var["vmax"], colorbar=False,
                       linthresh=var.get("linthresh"),
                       axis=axes_of.get(run, opts["axis"]),
                       length_units=opts["length_units"],
                       coordinates=frames[run].coordinates,
                       in_plane=planes[run].in_plane)
            for half, var in zip(halves, vars_)
        ]
        if sp["cut_line"]:
            (ax.axhline if cut == "horizontal" else ax.axvline)(at, **sp["cut_line"])
        finish_panel(r, c, ax)
        labels = [l or v["label"] for l, v in zip(sp["labels"] or [None, None], vars_)]
        if bottom_row or right_column:
            places = [("row", (0.0, 1.0)), ("row", (0.0, 1.0))]
        elif vertical and cut == "horizontal":
            # Two short bars on the right, each beside its own half.
            lo, hi = ax.get_ylim()
            f = min(max((at - lo) / (hi - lo), 0.0), 1.0)
            places = [("right", (f, 1.0)), ("right", (0.0, f))]
        elif cut == "horizontal":
            places = [("top", (0.0, 1.0)), ("bottom", (0.0, 1.0))]
        else:
            places = [("left", (0.0, 1.0)), ("right", (0.0, 1.0))]
        side_bars += [(ax, m, l, loc, span, v.get("ticks"))
                      for m, l, (loc, span), v in zip(halves_meshes, labels, places, vars_)]
        ax.set_title(panel_title(r, c, run, None))

    for r in range(nrows):
        for c in range(ncols):
            if layout[r][c] is None:
                axes[r, c].set_axis_off()
    if opts.get("panel_letters"):
        filled = [axes[r, c] for r, c, _, _ in sorted(cells)]
        add_panel_letters(filled, opts["panel_letters"])

    if row_bars:
        for r, row in enumerate(layout):
            filled = [c for c, cell in enumerate(row) if cell is not None]
            if filled:
                fig.colorbar(meshes[r, filled[0]], ax=[axes[r, c] for c in filled])
    elif col_bars:
        for c, col in enumerate(zip(*layout)):
            filled = [r for r, cell in enumerate(col) if cell is not None]
            if filled:
                fig.colorbar(meshes[filled[0], c], ax=[axes[r, c] for r in filled],
                             orientation="horizontal")

    # One time if the runs agree (to the shown precision), else one per run.
    times = {run: time_title(frames[run], params[run], opts) for run in runs}
    if opts.get("title", True):
        if opts.get("title_text"):  # given by hand, for every frame
            fig.suptitle(opts["title_text"], **title_at)
        elif len(set(times.values())) == 1:
            fig.suptitle(times[runs[0]], **title_at)
        else:
            fig.suptitle("    ".join(f"{run}: {t}" for run, t in times.items()), **title_at)

    if bottom_row:
        add_bottom_colorbars(fig, [ax for ax in axes.flat if ax.axison], side_bars)
    elif right_column:
        add_right_colorbars(fig, [ax for ax in axes.flat if ax.axison], side_bars)
    elif side_bars:
        add_side_colorbars(fig, side_bars, opts.get("cbar_length", 1.0),
                           reserve=hspace is None)

    path = Path(opts["outdir"]) / f"{opts['name']}.{opts['axis']}.{number:05d}.{opts['format']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if hspace is None:
        fig.savefig(path, dpi=opts["dpi"])
    else:  # margins are generous; trim them to what is drawn
        fig.savefig(path, dpi=opts["dpi"], bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    for frame in frames.values():
        frame.unload()
    return path


PANEL_CORNERS = {"upper left": (0, 1), "upper right": (1, 1),
                 "lower left": (0, 0), "lower right": (1, 0)}


def add_panel_letters(axes, corner: str) -> None:
    """(a), (b), ... in ``corner`` (a key of PANEL_CORNERS) of each of ``axes``, in order."""
    fx, fy = PANEL_CORNERS[corner]
    pad = 4  # points from the panel's edges
    for i, ax in enumerate(axes):
        ax.annotate(f"({chr(ord('a') + i)})", xy=(fx, fy), xycoords="axes fraction",
                    xytext=(pad if fx == 0 else -pad, pad if fy == 0 else -pad),
                    textcoords="offset points", ha="left" if fx == 0 else "right",
                    va="bottom" if fy == 0 else "top",
                    fontsize=plt.rcParams["ytick.labelsize"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.7))


def add_side_colorbars(fig, bars, length: float, reserve: bool = True) -> None:
    """
    Colorbars on the given side ("top", "bottom", "left", "right") of their
    panels, each over the ``span`` (fractions, e.g. (0.5, 1) for the upper
    half) of its panel's side.  With ``reserve`` the constrained layout makes
    room for them first and is then frozen; otherwise the figure's margins
    must have room.  Each is fitted to its panel as drawn (the equal aspect
    shrinks a panel inside its slot): ``length`` x its span, centred on it,
    just outside the panel (top/bottom: outside its tick and axis labels).
    Labels of left/right bars are written across, beside the bar, so short
    bars stay readable, centred on one line for all bars in a column (see
    `align_side_labels`); ``ticks`` (a list) sets the bar's ticks.
    """
    cbars, side_labels = [], []
    for ax, mesh, label, location, _, _ in bars:
        across = location in ("top", "bottom")
        if reserve:
            long_in = (ax.get_position().width * fig.get_figwidth() if across
                       else ax.get_position().height * fig.get_figheight())
            cbar = fig.colorbar(mesh, ax=ax, location=location, label=label,
                                aspect=max(long_in / CBAR_THICKNESS, 20.0))
        else:
            cbar = fig.colorbar(mesh, cax=fig.add_axes([0.0, 0.0, 0.01, 0.01]),
                                orientation="horizontal" if across else "vertical")
        cbars.append(cbar)
    fig.canvas.draw()
    if reserve:
        fig.set_layout_engine("none")  # keep the positions set below at savefig
    # All boxes first: moving one colorbar does not move the panels.
    boxes = [(ax.get_position(), _figure_bbox(fig, ax)) for ax, *_ in bars]
    w_in, h_in = fig.get_size_inches()
    for (ax, mesh, label, location, span, ticks), cbar, (box, tight) in zip(bars, cbars, boxes):
        cax = cbar.ax
        cax.set_box_aspect(None)
        s0, s1 = span
        if location in ("top", "bottom"):
            thick, pad = CBAR_THICKNESS / h_in, CBAR_PAD / h_in
            width = length * (s1 - s0) * box.width
            x0 = box.x0 + box.width * (s0 + 0.5 * (s1 - s0)) - 0.5 * width
            y0 = tight.y1 + pad if location == "top" else tight.y0 - pad - thick
            cax.set_position([x0, y0, width, thick])
            axis = cax.xaxis
            axis.set_ticks_position(location)
            axis.set_label_position(location)
            cbar.set_label(label)
        else:
            thick, pad = CBAR_THICKNESS / w_in, CBAR_PAD / w_in
            # Bars sharing a side (a split panel's halves) keep a gap between them.
            inset = SPAN_GAP if (s0, s1) != (0.0, 1.0) else 0.0
            height = max(length * (s1 - s0) - 2 * inset, 0.05) * box.height
            y0 = box.y0 + box.height * (s0 + 0.5 * (s1 - s0)) - 0.5 * height
            x0 = box.x1 + pad if location == "right" else tight.x0 - pad - thick
            cax.set_position([x0, y0, thick, height])
            axis = cax.yaxis
            axis.set_ticks_position(location)
            axis.set_label_position(location)
            side_labels.append((cax, label, location))
        set_bar_ticks(cbar, axis, mesh, ticks)
    if side_labels:
        align_side_labels(fig, side_labels)


def align_side_labels(fig, labels) -> None:
    """
    Labels written across beside vertical colorbars, ``labels`` being
    (colorbar axes, text, "left"/"right").  The labels of the bars in one
    column share one centre line: the nearest to the bars that keeps every
    label clear of its own bar's tick labels.
    """
    fig.canvas.draw()  # tick labels laid out, for their extent
    renderer = fig.canvas.get_renderer()
    pad = CBAR_PAD / fig.get_figwidth()
    columns = {}
    for cax, text, location in labels:
        box, edge = cax.get_position(), _figure_bbox(fig, cax)
        artist = fig.text(0.0, 0.5 * (box.y0 + box.y1), text, ha="center", va="center",
                          multialignment="center", fontsize=plt.rcParams["axes.labelsize"],
                          fontweight=plt.rcParams["axes.labelweight"])
        width = artist.get_window_extent(renderer).transformed(fig.transFigure.inverted()).width
        # Centre needed to clear this bar's ticks, outward from the bar.
        need = (edge.x1 + pad + 0.5 * width if location == "right"
                else edge.x0 - pad - 0.5 * width)
        columns.setdefault((location, round(box.x0, 4)), []).append((artist, need))
    for (location, _), column in columns.items():
        centre = (max if location == "right" else min)(need for _, need in column)
        for artist, _ in column:
            artist.set_x(centre)


def add_bottom_colorbars(fig, panels, bars) -> None:
    """
    The ``bars`` (as for `add_side_colorbars`; side and span ignored) in one
    row under the figure's ``panels``, below their x labels, side by side
    across the panels' width in panel order (top to bottom, left to right;
    a split panel's two in its order).  The margins must have room.
    """
    fig.canvas.draw()
    bars = sorted(bars, key=lambda b: (-b[0].get_position().y0, b[0].get_position().x0))
    w_in, h_in = fig.get_size_inches()
    x0 = min(ax.get_position().x0 for ax in panels)
    x1 = max(ax.get_position().x1 for ax in panels)
    below = min(_figure_bbox(fig, ax).y0 for ax in panels)
    thick, gap = CBAR_THICKNESS / h_in, BOTTOM_BAR_GAP / w_in
    y0 = below - 2 * CBAR_PAD / h_in - thick
    width = (x1 - x0 - (len(bars) - 1) * gap) / len(bars)
    for i, (_, mesh, label, _, _, ticks) in enumerate(bars):
        cax = fig.add_axes([x0 + i * (width + gap), y0, width, thick])
        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cbar.set_label(label)
        set_bar_ticks(cbar, cax.xaxis, mesh, ticks)


def add_right_colorbars(fig, panels, bars) -> None:
    """
    The ``bars`` (as for `add_side_colorbars`; side and span ignored) as
    vertical bars spanning the full height of the figure's ``panels``, side by
    side to their right in panel order (top to bottom, left to right; a split
    panel's two in its order), each labelled underneath.  Each is placed after
    the previous one is drawn, just clear of its tick labels and label.
    """
    fig.canvas.draw()
    bars = sorted(bars, key=lambda b: (-b[0].get_position().y0, b[0].get_position().x0))
    w_in, _ = fig.get_size_inches()
    y0 = min(ax.get_position().y0 for ax in panels)
    y1 = max(ax.get_position().y1 for ax in panels)
    x0 = max(ax.get_position().x1 for ax in panels) + CBAR_PAD / w_in
    thick, gap = CBAR_THICKNESS / w_in, RIGHT_BAR_GAP / w_in
    for _, mesh, label, _, _, ticks in bars:
        cax = fig.add_axes([x0, y0, thick, y1 - y0])
        cbar = fig.colorbar(mesh, cax=cax, orientation="vertical")
        # Under the bar, written across, clear of the bottom tick label's lower half.
        cax.set_xlabel(label, labelpad=0.6 * plt.rcParams["ytick.labelsize"])
        set_bar_ticks(cbar, cax.yaxis, mesh, ticks)
        fig.canvas.draw()
        x0 = _figure_bbox(fig, cax).x1 + gap


def set_bar_ticks(cbar, axis, mesh, ticks) -> None:
    """
    ``ticks`` (a list) on a colorbar (powers of ten for log and symlog norms),
    else non-overlapping log ticks for a log norm.
    """
    log = isinstance(mesh.norm, (matplotlib.colors.LogNorm, LogCenterNorm))
    if ticks is not None:
        set_fixed_ticks(cbar, axis, ticks,
                        log=log or isinstance(mesh.norm, matplotlib.colors.SymLogNorm))
    elif log:
        log_ticks(axis, mesh.norm.vmin, mesh.norm.vmax)


def _plot_combined_star(job):
    return plot_combined(*job)


# ── Command line ─────────────────────────────────────────────────────────

def _optional_float(text: str):
    return None if text.lower() == "none" else float(text)


def main(argv=None):
    plot, combined = config.PLOT, config.COMBINED
    parser = argparse.ArgumentParser(
        description="Combined slices: a grid of panels per frame, laid out by "
                    "config.COMBINED['layout'].",
    )
    parser.add_argument("--layout", type=parse_layout,
                        help='grid as "a,b;c,d": rows by ";", cells by ",", "run:var" '
                             'for another run, "-" for empty (default: config.COMBINED)')
    parser.add_argument("--run", default=config.RUN, choices=list(config.SIMULATIONS),
                        help=f"run of cells that name no run (default: {config.RUN})")
    parser.add_argument("--frames", nargs="+",
                        help="frame numbers and/or ranges like 0:21:5 "
                             "(default: from the layout's first run)")
    parser.add_argument("--name", default=combined["name"],
                        help=f"file name prefix (default: {combined['name']})")
    parser.add_argument("--outdir", help="where images are written "
                                         "(default: <first run's out>/combined)")
    parser.add_argument("--axis", choices=["x", "y", "z"], default=plot["axis"],
                        help="slice normal of runs without their own axis in "
                             "config.SIMULATIONS; also names the images "
                             f"(default: {plot['axis']})")
    parser.add_argument("--position", type=float, default=plot["position"],
                        help="slice position along --axis in --length-units, or in degrees "
                             "for an angle such as phi of a cylindrical run (default: midplane)")
    parser.add_argument("--length-units", default=plot["length_units"])
    parser.add_argument("--time-units", default=plot["time_units"])
    parser.add_argument("--time-decimals", type=non_negative_int,
                        default=plot.get("time_decimals", 2),
                        help="decimal places of the time in titles "
                             f"(default: {plot.get('time_decimals', 2)})")
    parser.add_argument("--colorbars",
                        choices=["auto", "horizontal", "vertical", "bottom", "right"],
                        default=combined.get("colorbars", "auto"),
                        help="auto: shared by a row/column of one variable, else one per "
                             "panel on its right; horizontal: one per panel, below it; "
                             "vertical: one per panel on its right (a split panel: one "
                             "beside each half); bottom: all in one row under the figure; "
                             "right: all side by side right of the figure, each as tall as "
                             "the panels (bottom and right need --hspace) "
                             f"(default: {combined.get('colorbars', 'auto')})")
    parser.add_argument("--xlim", type=_optional_float, nargs=2,
                        default=combined.get("xlim"), metavar=("MIN", "MAX"),
                        help="range of the panels' x axis in --length-units; 'none' "
                             f"keeps that end (default: {combined.get('xlim')} = full)")
    parser.add_argument("--ylim", type=_optional_float, nargs=2,
                        default=combined.get("ylim"), metavar=("MIN", "MAX"),
                        help=f"range of the panels' y axis (default: {combined.get('ylim')})")
    parser.add_argument("--sharex", action=argparse.BooleanOptionalAction,
                        default=combined.get("sharex", False),
                        help="share the x axis: tick labels on the bottom row only")
    parser.add_argument("--hspace", type=_optional_float, default=combined.get("hspace"),
                        help="stack the panels by hand with this gap (fraction of a panel's "
                             "height); none = constrained layout "
                             f"(default: {combined.get('hspace')})")
    parser.add_argument("--title", action=argparse.BooleanOptionalAction,
                        default=combined.get("title", True), help="time as the title")
    parser.add_argument("--title-text", default=combined.get("title_text"),
                        help="title written instead of the computed time, on every frame "
                             f"(default: {combined.get('title_text')} = the time)")
    parser.add_argument("--panel-letters", type=lambda t: None if t.lower() == "none" else t,
                        choices=[*PANEL_CORNERS, None], default=combined.get("panel_letters"),
                        help="corner of each panel for its letter (a), (b), ... in reading "
                             f"order; none = no letters (default: {combined.get('panel_letters')})")
    parser.add_argument("--cbar-room", type=float, default=combined.get("cbar_room"),
                        help="inches right of each panel for its vertical colorbar and "
                             "label (hspace set; also the gap between columns) "
                             f"(default: {combined.get('cbar_room') or MARGIN_RIGHT_BARS})")
    parser.add_argument("--cbar-length", type=float, default=combined.get("cbar_length", 1.0),
                        help="length of horizontal (and split) colorbars, fraction of the "
                             f"panel's width (default: {combined.get('cbar_length', 1.0)})")
    parser.add_argument("--panel-size", type=float, nargs=2, default=plot["panel_size"])
    parser.add_argument("--dpi", type=int, default=plot["dpi"])
    parser.add_argument("--format", default=plot["format"])
    parser.add_argument("-w", "--workers", "-c", "--cores", dest="workers", type=int,
                        default=n_workers(),
                        help="parallel worker processes, 1 = serial "
                             f"(default: config.N_WORKERS = {n_workers()})")
    parser.add_argument("--video", action="store_true",
                        help="also make a video of the images (see python main.py video)")
    parser.add_argument("--save-png", action=argparse.BooleanOptionalAction, default=None,
                        help="keep the images once the video is made; --no-save-png "
                             f"deletes them (default: config.MAIN = {config.MAIN['save_png']})")
    parser.add_argument("--fps", type=float, default=combined["fps"],
                        help=f"video frames per second (default: {combined['fps']})")
    parser.add_argument("--max-size", type=int, default=combined.get("video_max_size", 4096),
                        help="scale video frames down to at most this many pixels per "
                             "side; 0 keeps the full size (default: config.COMBINED"
                             f"['video_max_size'] = {combined.get('video_max_size', 4096)})")
    args = parser.parse_args(argv)

    try:
        layout = normalize_layout(args.layout or combined["layout"], args.run)
    except ValueError as err:
        parser.error(str(err))
    if not any(cell for row in layout for cell in row):
        parser.error("the layout has no panels")
    has_split = any(c is not None and is_split(c[1]) for row in layout for c in row)
    if args.hspace is not None and args.colorbars == "horizontal":
        parser.error("--hspace (panels stacked by hand) needs --colorbars right, bottom, "
                     "vertical or auto")
    if args.colorbars in ("bottom", "right") and args.hspace is None:
        parser.error(f"--colorbars {args.colorbars} needs --hspace (panels stacked by hand)")
    if has_split and not hasattr(config, "SPLIT"):
        parser.error("the layout has a split panel but config.py has no SPLIT settings")
    runs = layout_runs(layout)
    unknown_runs = [r for r in runs if r not in config.SIMULATIONS]
    if unknown_runs:
        parser.error(f"unknown runs {unknown_runs}; config.SIMULATIONS has "
                     f"{list(config.SIMULATIONS)}")

    try:
        resolved = {name: config.resolve(name) for name in runs}
    except RuntimeError as err:
        parser.error(str(err))
    sims = {name: SimulationData(r["athinput"], r["data"], outputs=r["outputs"],
                                 basename=r.get("basename"))
            for name, r in resolved.items()}

    wanted = parse_frames(args.frames)
    if wanted is None and resolved[runs[0]]["frames"] is not None:
        wanted = set(resolved[runs[0]]["frames"])
    common = set.intersection(*(set(s.frame_numbers) for s in sims.values()))
    numbers = sorted(n for n in common if wanted is None or n in wanted)
    if not numbers:
        parser.error("no frames selected that every run has; frames: "
                     + ", ".join(f"{n}: {s.frame_numbers}" for n, s in sims.items()))

    by_number = {name: {f.number: f for f in s.frames} for name, s in sims.items()}
    for name, sim in sims.items():
        frame = by_number[name][numbers[0]]
        variables = {v for row in layout for c in row if c is not None and c[0] == name
                     for v in cell_variables(c[1])}
        unknown = sorted(v for v in variables
                         if not has_quantity(frame, config.plot_var(v)["quantity"]))
        if unknown:
            parser.error(f"run {name}: unknown variables {unknown}; config.PLOT_VARS has "
                         f"{list(config.PLOT_VARS)}, quantities: {available(frame)}")
        try:  # Fail on a bad unit before spawning any work.
            unit_factor(sim.params, "length", args.length_units)
            unit_factor(sim.params, "time", args.time_units)
            for v in variables:
                var = config.plot_var(v)
                unit_factor(sim.params, dimension(var["quantity"]), var["units"])
        except ValueError as err:
            parser.error(f"run {name}: {err}")

    opts = {k: getattr(args, k) for k in
            ("axis", "position", "length_units", "time_units", "time_decimals",
             "panel_size", "dpi",
             "format", "name", "colorbars", "cbar_length", "cbar_room", "xlim", "ylim", "sharex",
             "hspace", "title", "title_text", "panel_letters")}
    opts["styles"] = combined.get("styles") or {}
    if has_split:
        sp = config.SPLIT
        opts["split"] = dict(cut=sp["cut"], at=sp["at"], mirror=sp["mirror"],
                             cut_line=sp["cut_line"], styles=sp.get("styles") or {},
                             labels=sp.get("labels"))
    opts["outdir"] = Path(args.outdir) if args.outdir else default_outdir(layout)
    opts["run_axes"] = {name: resolved[name].get("axis") or args.axis for name in runs}
    params = {name: sim.params for name, sim in sims.items()}
    jobs = [(layout, {name: by_number[name][n] for name in runs}, params, n, opts)
            for n in numbers]
    shape = f"{len(layout)}x{len(layout[0])}"
    print(f"plotting a {shape} layout of {runs} for {len(numbers)} frames -> {opts['outdir']}")

    written = []
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            for n, path in zip(numbers, pool.map(_plot_combined_star, jobs)):
                written.append(path)
                print(f"  frame {n:5d}: {path.name}")
    else:
        for n, job in zip(numbers, jobs):
            written.append(plot_combined(*job))
            print(f"  frame {n:5d}: {written[-1].name}")

    if args.video:
        from .video import make_video
        # Only this run's images, not leftovers of other frames in the folder.
        out = make_video(written, Path(opts["outdir"]) / f"{args.name}.{args.axis}.mp4",
                         fps=args.fps, max_size=args.max_size or None)
        print(f"video: {out}")
        if not (config.MAIN["save_png"] if args.save_png is None else args.save_png):
            # Only the images written by this run; the video holds them now.
            for path in written:
                path.unlink(missing_ok=True)
            print(f"deleted the {len(written)} images (save_png = False)")
    elif args.save_png is False:  # asked for on the command line, not just config
        print("--no-save-png only applies with --video; keeping the images")


if __name__ == "__main__":
    main()
