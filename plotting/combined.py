"""
Combined figure: a grid of slice panels per frame, laid out by config.COMBINED.

    python main.py combined                          # config.COMBINED["layout"]
    python main.py combined --frames 0:21:5 -w 4 --video
    python main.py combined --layout "dens,temp;pres,t_cool"
    python main.py combined --layout "cbox_02Myr:temp,cbox_tabcool:temp"

Each cell of the layout is a variable (plotted for config.RUN, or --run), a
(run, variable) pair, or None for an empty panel; on the command line rows are
separated by ";", cells by ",", a run is given as "run:var", and "-" is empty.
Variables are styled by config.PLOT_VARS.  If every row holds one variable,
each row shares a colorbar; else if every column does, each column does;
otherwise each panel has its own.  Frames are matched by frame number across
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

from .quantities import available, dimension  # noqa: E402
from .slice2d import (  # noqa: E402
    apply_fonts, frame_plane, n_workers, non_negative_int, parse_frames, slice_variable, time_title,
    unit_factor, var_settings,
)
from .slices import draw_slice  # noqa: E402


# ── Layout ───────────────────────────────────────────────────────────────

def normalize_layout(layout, default_run: str) -> list[list[tuple | None]]:
    """Layout cells -> (run, variable) or None; rows padded to equal length."""
    rows = []
    for row in layout:
        cells = []
        for cell in row:
            if cell is None:
                cells.append(None)
            elif isinstance(cell, str):
                cells.append((default_run, cell))
            else:
                run, var = cell
                cells.append((run, var))
        rows.append(cells)
    ncols = max((len(r) for r in rows), default=0)
    return [r + [None] * (ncols - len(r)) for r in rows]


def parse_layout(text: str) -> list[list]:
    """'dens,temp;run:pres,-' -> [["dens", "temp"], [("run", "pres"), None]]."""
    rows = []
    for row in text.split(";"):
        cells = []
        for cell in row.split(","):
            cell = cell.strip()
            if cell in ("", "-"):
                cells.append(None)
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


def plot_combined(layout, frames: dict, params: dict, number: int, opts) -> Path:
    """
    Draw frame ``number`` for a normalized ``layout``; ``frames`` and ``params``
    map each run to its Frame and athinput params.  Returns the file written.
    """
    apply_fonts()
    nrows, ncols = len(layout), len(layout[0])
    pw, ph = opts["panel_size"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, ph * nrows),
                             squeeze=False, layout="constrained")

    # Slice every panel first, so a variable shares its colour range across panels.
    cells = [(r, c, run, name) for r, row in enumerate(layout)
             for c, cell in enumerate(row) if cell is not None for run, name in [cell]]
    settings = {name: var_settings(name, opts) for _, _, _, name in cells}
    axes_of = opts.get("run_axes", {})
    planes = {run: frame_plane(frames[run], params[run], opts, axes_of.get(run))
              for run in frames}
    slices = {(r, c): slice_variable(planes[run], settings[name], params[run], opts)
              for r, c, run, name in cells}
    limits = {name: shared_limits([slices[r, c][2] for r, c, _, n in cells if n == name],
                                  var)
              for name, var in settings.items()}

    row_bars = all(_uniform(row) for row in layout if any(row))
    col_bars = not row_bars and all(_uniform(col) for col in zip(*layout) if any(col))
    runs = layout_runs(layout)
    run_per_column = all(len({c[0] for c in col if c is not None}) <= 1
                         for col in zip(*layout))
    top = {c: min((r for r in range(nrows) if layout[r][c] is not None), default=None)
           for c in range(ncols)}

    meshes = {}
    for r, c, run, name in cells:
        var, ax = settings[name], axes[r, c]
        x, y, plane = slices[r, c]
        vmin, vmax = limits[name]
        own_bar = not (row_bars or col_bars)
        meshes[r, c] = draw_slice(ax, x, y, plane, cmap=var["cmap"], norm=var["norm"],
                                  vmin=vmin, vmax=vmax, colorbar=own_bar,
                                  axis=axes_of.get(run, opts["axis"]),
                                  length_units=opts["length_units"],
                                  coordinates=frames[run].coordinates,
                                  in_plane=planes[run].in_plane)
        if r + 1 < nrows and layout[r + 1][c] is not None:
            ax.set_xlabel("")
        if c > 0 and layout[r][c - 1] is not None:
            ax.set_ylabel("")
        # The variable names the panel; with several runs, the run does too
        # (once per column if each column is one run).
        title = var["label"]
        if len(runs) > 1 and (not run_per_column or r == top[c]):
            title = f"{run}\n{title}"
        ax.set_title(title)

    for r in range(nrows):
        for c in range(ncols):
            if layout[r][c] is None:
                axes[r, c].set_axis_off()

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
    if len(set(times.values())) == 1:
        fig.suptitle(times[runs[0]])
    else:
        fig.suptitle("    ".join(f"{run}: {t}" for run, t in times.items()))

    path = Path(opts["outdir"]) / f"{opts['name']}.{opts['axis']}.{number:05d}.{opts['format']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=opts["dpi"])
    plt.close(fig)
    for frame in frames.values():
        frame.unload()
    return path


def _plot_combined_star(job):
    return plot_combined(*job)


# ── Command line ─────────────────────────────────────────────────────────

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

    layout = normalize_layout(args.layout or combined["layout"], args.run)
    if not any(cell for row in layout for cell in row):
        parser.error("the layout has no panels")
    runs = layout_runs(layout)
    unknown_runs = [r for r in runs if r not in config.SIMULATIONS]
    if unknown_runs:
        parser.error(f"unknown runs {unknown_runs}; config.SIMULATIONS has "
                     f"{list(config.SIMULATIONS)}")

    try:
        resolved = {name: config.resolve(name) for name in runs}
    except RuntimeError as err:
        parser.error(str(err))
    sims = {name: SimulationData(r["athinput"], r["data"], outputs=r["outputs"])
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
        variables = {c[1] for row in layout for c in row if c is not None and c[0] == name}
        unknown = sorted(v for v in variables
                         if config.plot_var(v)["quantity"] not in available(frame))
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
             "format", "name")}
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
