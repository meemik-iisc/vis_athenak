"""
Cold gas mass in the box as a function of time, for one or more runs.

    python main.py timeseries                                  # config.TIMESERIES
    python main.py timeseries --runs turb_sigv_30 turb_tcgm_8192 --overlay
    python main.py timeseries --runs turb_sigv_30 turb_tcgm_8192 --no-overlay
    python main.py timeseries --run turb_sigv_30 --T-cold 1e5 --frames 0:61:5

The cold mass of a frame is the sum of rho dV over cells with T < T_cold
(T from the run's fixed mu, as the `temp` quantity).  With --overlay every run
is a line on one set of axes; with --no-overlay each run gets its own panel,
stacked vertically with a shared time axis.  Each run is plotted against its
own snapshot times, so runs need not share frame numbers or output cadence.

Frames are reduced in parallel, one whole 3D frame per worker.  Each run's
values are cached in ``<run out>/timeseries/cold_mass_T<T_cold>.npz`` (code
units, by frame number), so later plots only read the frames not yet in the
cache; --recompute ignores it.  The figure goes to
``<outdir>/<name>.<format>``, outdir defaulting to ``timeseries/`` in the
plot folder of the first run.
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


# ── Reduction ────────────────────────────────────────────────────────────

def cold_mass(frame, params: dict, T_cold: float) -> float:
    """Mass of gas with T < T_cold [K] in the frame, in code units."""
    T = get(frame, "temp", params, units="K")
    rho = np.where(T < T_cold, np.asarray(frame["dens"], dtype=float), 0.0)
    v1, v2, v3 = frame.cell_volume_factors
    return float(np.einsum("kji,i,j,k->", rho, v1, v2, v3))


def _reduce_frame(job) -> tuple[str, int, float, float]:
    """Worker: (run, frame number, time, cold mass) of one frame, all in code units."""
    run, frame, params, T_cold = job
    try:
        return run, frame.number, frame.time, cold_mass(frame, params, T_cold)
    finally:
        frame.unload()


# ── Cache ────────────────────────────────────────────────────────────────

def cache_path(outdir: Path, T_cold: float) -> Path:
    """<run out>/timeseries/cold_mass_T<T>.npz, T written like 5e4."""
    mantissa, exponent = f"{T_cold:e}".split("e")
    tag = f"{float(mantissa):g}e{int(exponent)}"
    return Path(outdir) / "timeseries" / f"cold_mass_T{tag}.npz"


def read_cache(path: Path) -> dict[int, tuple[float, float]]:
    """{frame number: (time, cold mass)} in code units; empty if no cache."""
    if not path.is_file():
        return {}
    with np.load(path) as f:
        return {int(n): (float(t), float(m))
                for n, t, m in zip(f["number"], f["time"], f["mass"])}


def write_cache(path: Path, values: dict[int, tuple[float, float]], T_cold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numbers = sorted(values)
    np.savez(path, number=np.array(numbers),
             time=np.array([values[n][0] for n in numbers]),
             mass=np.array([values[n][1] for n in numbers]),
             T_cold=T_cold)


# ── Plot ─────────────────────────────────────────────────────────────────

def _T_label(T_cold: float) -> str:
    mantissa, exponent = f"{T_cold:e}".split("e")
    m = float(mantissa)
    return (f"10^{{{int(exponent)}}}" if m == 1
            else rf"{m:g}\times10^{{{int(exponent)}}}")


def plot_timeseries(series: dict[str, tuple[np.ndarray, np.ndarray]], labels: dict[str, str],
                    opts: dict, path: Path) -> Path:
    """
    Draw ``series`` {run: (time, mass)}, already in opts' time and mass
    units: one axes with a line per run if opts["overlay"], else one panel per
    run stacked vertically.
    """
    apply_fonts()
    runs = list(series)
    n_panels = 1 if opts["overlay"] else len(runs)
    width, height = opts["figsize"]
    fig, axes = plt.subplots(n_panels, 1, sharex=True, squeeze=False,
                             figsize=(width, height * n_panels), layout="constrained")
    axes = axes[:, 0]
    units = {"Msun": r"M$_\odot$"}.get(opts["mass_units"], opts["mass_units"])
    ylabel = rf"$M(T < {_T_label(opts['T_cold'])}\,\mathrm{{K}})$ [{units}]"
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for n, run in enumerate(runs):
        ax = axes[0 if opts["overlay"] else n]
        t, m = series[run]
        if opts["logy"]:  # e.g. no cold gas yet at t = 0: leave those points out
            m = np.where(m > 0, m, np.nan)
        ax.plot(t, m, marker="o", markersize=3, color=colors[n % len(colors)],
                label=labels.get(run, run))
        if opts["logy"]:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if opts["overlay"]:
            ax.legend()
        else:
            ax.set_title(labels.get(run, run))
    axes[-1].set_xlabel(f"t [{opts['time_units']}]")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=opts["dpi"])
    plt.close(fig)
    return path


# ── Command line ─────────────────────────────────────────────────────────

def main(argv=None):
    ts, plot = config.TIMESERIES, config.PLOT
    parser = argparse.ArgumentParser(
        prog="main.py timeseries",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runs_default = ts.get("runs") or [config.RUN]
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--runs", nargs="+", choices=list(config.SIMULATIONS),
                       help=f"runs to plot (default: {runs_default})")
    group.add_argument("--run", choices=list(config.SIMULATIONS),
                       help="a single run (same as --runs RUN)")
    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction,
                        default=ts["overlay"],
                        help="all runs on one axes; --no-overlay: one panel per run "
                             f"(default: {ts['overlay']})")
    parser.add_argument("--T-cold", dest="T_cold", type=float, default=ts["T_cold"],
                        help=f"cold means T below this, in K (default: {ts['T_cold']:g})")
    parser.add_argument("--frames", nargs="+",
                        help="frame numbers and/or ranges like 0:21:5, for every run "
                             "(default: each run's frames in config.SIMULATIONS)")
    parser.add_argument("--mass-units", default=ts["mass_units"],
                        help=f"g or Msun (default: {ts['mass_units']})")
    parser.add_argument("--time-units", default=ts.get("time_units") or plot["time_units"],
                        help="code, s, yr, kyr or Myr "
                             f"(default: {ts.get('time_units') or plot['time_units']})")
    parser.add_argument("--logy", action=argparse.BooleanOptionalAction, default=ts["logy"],
                        help=f"logarithmic mass axis (default: {ts['logy']})")
    parser.add_argument("--name", default=ts["name"],
                        help=f"figure file name, without extension (default: {ts['name']})")
    parser.add_argument("--outdir", help="figure folder (default: timeseries/ in the plot "
                                         "folder of the first run)")
    parser.add_argument("--figsize", type=float, nargs=2, default=ts["figsize"],
                        help="figure size in inches; with --no-overlay the height is "
                             f"per panel (default: {ts['figsize']})")
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
    wanted_cli = parse_frames(args.frames)

    # Open every run and check its units before reading any frame.
    sims, frames, caches, cached = {}, {}, {}, {}
    for run in runs:
        try:
            settings = config.resolve(run)
        except RuntimeError as err:
            parser.error(str(err))
        sim = SimulationData(settings["athinput"], settings["data"],
                             outputs=settings["outputs"])
        wanted = wanted_cli
        if wanted is None and settings["frames"] is not None:
            wanted = set(settings["frames"])
        frames[run] = [f for f in sim.frames if wanted is None or f.number in wanted]
        if not frames[run]:
            parser.error(f"{run}: no frames selected; run has frames {sim.frame_numbers}")
        try:
            unit_factor(sim.params, "mass", args.mass_units)
            unit_factor(sim.params, "time", args.time_units)
        except ValueError as err:
            parser.error(f"{run}: {err}")
        sims[run] = sim
        caches[run] = cache_path(settings["out"], args.T_cold)
        cached[run] = {} if args.recompute else read_cache(caches[run])

    jobs = [(run, f, sims[run].params, args.T_cold)
            for run in runs for f in frames[run] if f.number not in cached[run]]
    for run in runs:
        n_new = sum(1 for j in jobs if j[0] == run)
        print(f"{run}: {len(frames[run])} frames, {len(frames[run]) - n_new} cached "
              f"({caches[run]})")

    if jobs:
        print(f"reading {len(jobs)} frames with {min(args.workers, len(jobs))} workers")

        def store(result):
            run, number, time, mass = result
            cached[run][number] = (time, mass)
            print(f"  {run} frame {number:5d}: t = {time:.4g}, "
                  f"M_cold = {mass:.4g} (code)", flush=True)

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
                write_cache(caches[run], cached[run], args.T_cold)

    series = {}
    for run in runs:
        params = sims[run].params
        numbers = [f.number for f in frames[run]]
        t = np.array([cached[run][n][0] for n in numbers])
        m = np.array([cached[run][n][1] for n in numbers])
        series[run] = (t * unit_factor(params, "time", args.time_units),
                       m * unit_factor(params, "mass", args.mass_units))

    outdir = Path(args.outdir) if args.outdir else config.resolve(runs[0])["out"] / "timeseries"
    opts = {k: getattr(args, k) for k in
            ("overlay", "T_cold", "mass_units", "time_units", "logy", "figsize", "dpi")}
    path = plot_timeseries(series, ts.get("labels") or {}, opts,
                           outdir / f"{args.name}.{args.format}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
