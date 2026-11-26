"""
Cold gas mass in slabs along x (or per unit length, dM_cold/dx), for one or more runs.

    python main.py cold_mass                                   # config.COLD_MASS
    python main.py cold_mass --runs ucgm tcgm turb_tcgm --frames 60
    python main.py cold_mass --bin-width 1 --T-cold 1e5 --no-logy

With --bin-width (in --length-units) each step is the mass of cold gas
(T < T_cold, T from the run's fixed mu, as the `temp` quantity) in a slab of
that thickness spanning the whole y and z extent of the box; slabs start at
the box's x1min, a cell going to the slab holding its centre.  Without
--bin-width (or with 0) the slabs are the grid's own cells along x and the
cold mass per unit length, dM_cold/dx, is drawn as a continuous line.  Cartesian runs only.

All runs go on one set of axes, colour = run; with several frames per run
the line style tells the frames apart.  x without cold gas is a gap on a
log axis.

Frames are reduced in parallel, one whole 3D frame per worker.  Each run's
cold mass per cell column along x is cached in
``<run out>/cold_mass/cold_mass_T<T_cold>.npz`` (code units, by frame number),
so later plots, with any --bin-width, only read frames not yet in the cache;
--recompute ignores it.  The figure goes to
``<outdir>/<name>.<format>``, outdir defaulting to ``cold_mass/`` in the plot
folder of the first run.
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

from .quantities import get  # noqa: E402
from .slice2d import apply_fonts, n_workers, parse_frames, unit_factor  # noqa: E402
from .timeseries import _power, _tag  # noqa: E402

LINESTYLES = ["-", "--", ":", "-."]


# ── Reduction ────────────────────────────────────────────────────────────

def cold_mass_per_column(frame, params: dict, T_cold: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """(x1f, mass): cold gas mass (T < T_cold) in each cell column along x1,
    i.e. over the whole x2 and x3 extent; all in code units."""
    if frame.coordinates != "cartesian":
        raise NotImplementedError(f"cold_mass needs Cartesian coordinates, not {frame.coordinates}")
    v1, v2, v3 = frame.cell_volume_factors
    rho = np.asarray(frame["dens"], dtype=float)
    cold = get(frame, "temp", params, units="K") < T_cold
    mass = np.einsum("kji,j,k->i", np.where(cold, rho, 0.0), v2, v3) * v1
    return np.asarray(frame.grid["x1f"], dtype=float), mass


def rebin(x1f: np.ndarray, mass: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """(edges, mass) summed into slabs of ``width`` (same units as x1f) from
    x1f[0]; the last slab is cut at x1f[-1].  A cell goes to the slab holding
    its centre."""
    n = int(np.ceil((x1f[-1] - x1f[0]) / width - 1e-9))
    edges = np.minimum(x1f[0] + width * np.arange(n + 1), x1f[-1])
    centres = 0.5 * (x1f[1:] + x1f[:-1])
    slab = np.clip(np.searchsorted(edges, centres, side="right") - 1, 0, n - 1)
    return edges, np.bincount(slab, weights=mass, minlength=n)


def _reduce_frame(job) -> tuple[str, int, float, np.ndarray, np.ndarray]:
    """Worker: (run, frame number, time, x1f, mass per column) of one frame, code units."""
    run, frame, params, T_cold = job
    try:
        return (run, frame.number, frame.time, *cold_mass_per_column(frame, params, T_cold))
    finally:
        frame.unload()


# ── Cache ────────────────────────────────────────────────────────────────

def cache_path(outdir: Path, T_cold: float) -> Path:
    return Path(outdir) / "cold_mass" / f"cold_mass_T{_tag(T_cold)}.npz"


def read_cache(path: Path) -> dict[int, tuple[float, np.ndarray, np.ndarray]]:
    """{frame number: (time, x1f, mass per column)} in code units; empty if no cache."""
    if not path.is_file():
        return {}
    with np.load(path) as f:
        return {int(n): (float(f["time"][i]), f["x1f"].copy(), f["mass"][i].copy())
                for i, n in enumerate(f["number"])}


def write_cache(path: Path, values: dict, T_cold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numbers = sorted(values)
    np.savez(path, number=np.array(numbers),
             time=np.array([values[n][0] for n in numbers]),
             x1f=values[numbers[0]][1],  # same grid for every frame of a run
             mass=np.array([values[n][2] for n in numbers]), T_cold=T_cold)


# ── Plot ─────────────────────────────────────────────────────────────────

def plot_cold_mass(series: dict[str, list[dict]], labels: dict[str, str], opts: dict,
                   path: Path) -> Path:
    """
    Draw ``series`` {run: [{"time", "edges", "mass"}, ...]} (one dict per
    frame, already in opts' units; mass per interval of ``edges``): steps of
    the mass per slab if opts["bin_width"] is set, else dM/dx as a line
    through the interval centres (one interval per cell).  Colour = run, line style = frame; the
    legend gives each line's total cold mass.
    """
    apply_fonts()
    fig, ax = plt.subplots(figsize=opts["figsize"], layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    units = {"Msun": r"M$_\odot$"}.get(opts["mass_units"], opts["mass_units"])
    for n, (run, frames) in enumerate(series.items()):
        for k, d in enumerate(frames):
            # binned: mass per slab; else mass per unit length of each cell
            dmdx = d["mass"] if opts["bin_width"] else d["mass"] / np.diff(d["edges"])
            if opts["logy"]:
                dmdx = np.where(dmdx > 0, dmdx, np.nan)
            label = labels.get(run, run)
            if len(frames) > 1 or len(series) == 1:
                label += f", t = {d['time']:.{opts['time_decimals']}f} {opts['time_units']}"
            total = float(f"{d['mass'].sum():.2g}")
            label += "\n" + (rf"(${_power(total)}$ {units})" if total > 0 else f"(0 {units})")
            style = dict(color=colors[n % len(colors)],
                         linestyle=LINESTYLES[k % len(LINESTYLES)], label=label)
            if opts["bin_width"]:
                ax.stairs(dmdx, d["edges"], linewidth=1.5, **style)
            else:
                ax.plot(0.5 * (d["edges"][1:] + d["edges"][:-1]), dmdx, linewidth=1.0, **style)
    if opts["logy"]:
        ax.set_yscale("log")
    ax.set_xlabel(f"x [{opts['length_units']}]")
    if opts["bin_width"]:  # mass per slab (dM/dx x bin_width); the width goes in the caption
        ax.set_ylabel(rf"$M_{{\rm cold}}$ [{units}]")
    else:
        ax.set_ylabel(rf"$dM_{{\rm cold}}/dx$ [{units}/{opts['length_units']}]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=opts["legend_fontsize"], loc=opts["legend_loc"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=opts["dpi"])
    plt.close(fig)
    return path


# ── Command line ─────────────────────────────────────────────────────────

def main(argv=None):
    if not hasattr(config, "COLD_MASS"):
        raise SystemExit("main.py cold_mass: config.py has no COLD_MASS")
    cm, plot = config.COLD_MASS, config.PLOT
    parser = argparse.ArgumentParser(
        prog="main.py cold_mass",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runs_default = cm.get("runs") or [config.RUN]
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--runs", nargs="+", choices=list(config.SIMULATIONS),
                       help=f"runs to plot (default: {runs_default})")
    group.add_argument("--run", choices=list(config.SIMULATIONS),
                       help="a single run (same as --runs RUN)")
    parser.add_argument("--frames", nargs="+",
                        help="frame numbers and/or ranges like 0:21:5, for every run "
                             f"(default: {cm.get('frames') or 'each run`s frames in config.SIMULATIONS'})")
    parser.add_argument("--T-cold", dest="T_cold", type=float, default=cm["T_cold"],
                        help=f"cold means T below this, in K (default: {cm['T_cold']:g})")
    parser.add_argument("--bin-width", type=float, default=cm.get("bin_width"),
                        help="slab thickness along x, in --length-units; 0 = no binning, "
                             f"a continuous line per grid cell (default: {cm.get('bin_width')})")
    parser.add_argument("--length-units", default=cm["length_units"],
                        help=f"code, cm, pc or kpc (default: {cm['length_units']})")
    parser.add_argument("--mass-units", default=cm["mass_units"],
                        help=f"g or Msun (default: {cm['mass_units']})")
    parser.add_argument("--time-units", default=plot["time_units"],
                        help=f"time in the legend (default: {plot['time_units']})")
    parser.add_argument("--logy", action=argparse.BooleanOptionalAction, default=cm["logy"],
                        help=f"logarithmic mass axis (default: {cm['logy']})")
    parser.add_argument("--name", default=cm["name"],
                        help=f"figure file name, without extension (default: {cm['name']})")
    parser.add_argument("--outdir", help="figure folder (default: cold_mass/ in the plot "
                                         "folder of the first run)")
    parser.add_argument("--figsize", type=float, nargs=2, default=cm["figsize"],
                        help=f"figure size in inches (default: {cm['figsize']})")
    parser.add_argument("--dpi", type=int, default=plot["dpi"])
    parser.add_argument("--format", default=plot["format"], help="image format")
    parser.add_argument("--recompute", action="store_true",
                        help="ignore the cached values and read every frame again")
    parser.add_argument("-w", "--workers", "-c", "--cores", dest="workers", type=int,
                        default=n_workers(),
                        help="parallel worker processes, 1 = serial "
                             f"(default: config.N_WORKERS = {n_workers()})")
    args = parser.parse_args(argv)
    runs = args.runs or ([args.run] if args.run else runs_default)
    wanted_cli = parse_frames(args.frames) if args.frames else (
        set(cm["frames"]) if cm.get("frames") is not None else None)

    sims, frames, caches, cached = {}, {}, {}, {}
    for run in runs:
        settings = config.resolve(run)
        sim = SimulationData(settings["athinput"], settings["data"], outputs=settings["outputs"],
                             basename=settings.get("basename"))
        wanted = wanted_cli
        if wanted is None and settings["frames"] is not None:
            wanted = set(settings["frames"])
        frames[run] = [f for f in sim.frames if wanted is None or f.number in wanted]
        if not frames[run]:
            parser.error(f"{run}: no frames selected; run has frames {sim.frame_numbers}")
        try:
            for dim, unit in (("length", args.length_units), ("mass", args.mass_units),
                              ("time", args.time_units)):
                unit_factor(sim.params, dim, unit)
        except ValueError as err:
            parser.error(f"{run}: {err}")
        sims[run] = sim
        caches[run] = cache_path(settings["out"], args.T_cold)
        cached[run] = {} if args.recompute else read_cache(caches[run])

    jobs = []
    for run in runs:
        new = [f for f in frames[run] if f.number not in cached[run]]
        jobs += [(run, f, sims[run].params, args.T_cold) for f in new]
        print(f"{run}: {len(frames[run])} frames, {len(frames[run]) - len(new)} cached "
              f"({caches[run]})")

    if jobs:
        print(f"reading {len(jobs)} frames with {min(args.workers, len(jobs))} workers")

        def store(result):
            run, number, time, x1f, mass = result
            cached[run][number] = (time, x1f, mass)
            print(f"  {run} frame {number:5d}: t = {time:.4g}, total cold = "
                  f"{mass.sum():.4g} (code)", flush=True)

        try:
            if args.workers > 1 and len(jobs) > 1:
                with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
                    for result in pool.map(_reduce_frame, jobs):
                        store(result)
            else:
                for job in jobs:
                    store(_reduce_frame(job))
        finally:  # keep what was computed, even if interrupted
            for run in {j[0] for j in jobs}:
                if cached[run]:
                    write_cache(caches[run], cached[run], args.T_cold)

    series = {}
    for run in runs:
        params = sims[run].params
        lf = unit_factor(params, "length", args.length_units)
        mf = unit_factor(params, "mass", args.mass_units)
        tf = unit_factor(params, "time", args.time_units)
        series[run] = []
        for f in frames[run]:
            time, x1f, mass = cached[run][f.number]
            edges, mass = x1f * lf, mass * mf
            if args.bin_width:
                edges, mass = rebin(edges, mass, args.bin_width)
            series[run].append(dict(time=time * tf, edges=edges, mass=mass))

    outdir = Path(args.outdir) if args.outdir else config.resolve(runs[0])["out"] / "cold_mass"
    opts = {k: getattr(args, k) for k in
            ("T_cold", "bin_width", "length_units", "mass_units", "time_units",
             "logy", "figsize", "dpi")}
    opts["time_decimals"] = plot.get("time_decimals", 1)
    opts["legend_fontsize"] = cm.get("legend_fontsize", "medium")
    opts["legend_loc"] = cm.get("legend_loc", "best")
    path = plot_cold_mass(series, cm.get("labels") or {}, opts,
                          outdir / f"{args.name}.{args.format}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
