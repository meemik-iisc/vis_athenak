"""
Gas masses in the box as a function of time, for one or more runs.

    python main.py timeseries                                  # config.TIMESERIES
    python main.py timeseries --quantities cold wind wind_tracer analytic
    python main.py timeseries --runs turb_sigv_30 turb_tcgm_8192 --no-overlay
    python main.py timeseries --run turb_sigv_30 --T-cold 1e5 --s-min 1e-4 --frames 0:61:5

Each quantity is a sum of rho dV over the box (T from the run's fixed mu, as
the `temp` quantity; s = s_00, the wind tracer, clipped to [0, 1]):

    cold         cells with T < T_cold
    wind         cells with s > s_min (wind-polluted gas, mostly CGM in mixed cells)
    cold_wind    cells with T < T_cold and s > s_min
    wind_tracer  every cell weighted by s (the wind material itself)
    analytic     wind mass injected so far for Mdot(t) = Mdot0 exp(-t/tau):
                 Mdot0 tau (1 - exp(-t/tau)), with Mdot0 [Msun/yr] and
                 tau = dt_mdot [code time] from the run's <problem> block
                 (turb_tcgm); left out for runs without them.  Not read
                 from the frames, drawn as a smooth dotted curve.
    entrained    entrained CGM mass, cold - analytic at each frame time, with
                 the cold mass as matched (see --match) if cold is matched,
                 else as is; zero at --match-at then, and values <= 0 are
                 left out on a log axis.  <factor> in its label shows
                 cold's factor

--match cold --match-at 1 rescales the cold mass of each run by a constant so
it equals the analytic injected mass at t = 1 (in --time-units; the frame
values are interpolated linearly to that time), to compare how fast each
grows; the factor is shown in the legend.  --match with no quantities turns
it off.

With --overlay every run goes on one set of axes (colour = run, line style =
quantity; runs with the same Mdot0 and dt_mdot share one analytic curve); with --no-overlay each run gets its own panel, stacked vertically
with a shared time axis.  Each run is plotted against its own snapshot times,
so runs need not share frame numbers or output cadence.

Frames are reduced in parallel, one whole 3D frame per worker, all quantities
in one read.  Each run's values are cached in
``<run out>/timeseries/masses_T<T_cold>_s<s_min>.npz`` (code units, by frame
number), so later plots only read frames not yet in the cache; --recompute
ignores it.  The figure goes to ``<outdir>/<name>.<format>``, outdir
defaulting to ``timeseries/`` in the plot folder of the first run.
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
    from ..utils.units import MSUN_CGS, YR_CGS, Units
else:  # run from the repo root
    import config
    from simulation_data import SimulationData
    from utils.units import MSUN_CGS, YR_CGS, Units

from .quantities import available, get  # noqa: E402
from .slice2d import apply_fonts, n_workers, parse_frames, unit_factor  # noqa: E402

QUANTITIES = ["cold", "wind", "cold_wind", "wind_tracer"]  # summed over each frame
ANALYTIC = "analytic"                                      # from the athinput only
ENTRAINED = "entrained"                                    # cold - analytic
CHOICES = [*QUANTITIES, ANALYTIC, ENTRAINED]


# ── Reduction ────────────────────────────────────────────────────────────

def masses(frame, params: dict, T_cold: float, s_min: float) -> dict[str, float]:
    """Every quantity in QUANTITIES for one frame, in code units; the wind
    ones are NaN if the run has no s_00 tracer."""
    v = frame.cell_volume_factors

    def total(q):
        return float(np.einsum("kji,i,j,k->", q, *v))

    rho = np.asarray(frame["dens"], dtype=float)
    cold = get(frame, "temp", params, units="K") < T_cold
    out = {"cold": total(np.where(cold, rho, 0.0))}
    if "s_00" in available(frame):
        s = np.clip(np.asarray(get(frame, "s_00", params), dtype=float), 0.0, 1.0)
        wind = s > s_min
        out["wind"] = total(np.where(wind, rho, 0.0))
        out["cold_wind"] = total(np.where(cold & wind, rho, 0.0))
        out["wind_tracer"] = total(rho * s)
    else:
        out.update(wind=np.nan, cold_wind=np.nan, wind_tracer=np.nan)
    return out


def _reduce_frame(job) -> tuple[str, int, float, dict[str, float]]:
    """Worker: (run, frame number, time, masses) of one frame, all in code units."""
    run, frame, params, T_cold, s_min = job
    try:
        return run, frame.number, frame.time, masses(frame, params, T_cold, s_min)
    finally:
        frame.unload()


def analytic_mass(params: dict, t: np.ndarray) -> np.ndarray | None:
    """Wind mass injected by time t (code units) for Mdot(t) = Mdot0 exp(-t/tau):
    Mdot0 tau (1 - exp(-t/tau)), from <problem> Mdot0 [Msun/yr] and
    dt_mdot = tau [code time]; None if absent."""
    problem = params.get("problem", {})
    if "Mdot0" not in problem or "dt_mdot" not in problem:
        return None
    u = Units.from_params(params)
    mdot0 = float(problem["Mdot0"]) * MSUN_CGS / YR_CGS / (u.mass_cgs / u.time_cgs)
    tau = float(problem["dt_mdot"])
    return mdot0 * tau * (1.0 - np.exp(-t / tau))


# ── Cache ────────────────────────────────────────────────────────────────

def _tag(x: float) -> str:
    """5e4 for 50000.0, 1e-3 for 0.001."""
    mantissa, exponent = f"{x:e}".split("e")
    return f"{float(mantissa):g}e{int(exponent)}"


def cache_path(outdir: Path, T_cold: float, s_min: float) -> Path:
    return Path(outdir) / "timeseries" / f"masses_T{_tag(T_cold)}_s{_tag(s_min)}.npz"


def read_cache(path: Path) -> dict[int, tuple[float, dict[str, float]]]:
    """{frame number: (time, masses)} in code units; frames missing any
    quantity are left out (so they are read again); empty if no cache."""
    if not path.is_file():
        return {}
    with np.load(path) as f:
        if not all(q in f for q in QUANTITIES):
            return {}
        return {int(n): (float(f["time"][i]), {q: float(f[q][i]) for q in QUANTITIES})
                for i, n in enumerate(f["number"])}


def write_cache(path: Path, values: dict, T_cold: float, s_min: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numbers = sorted(values)
    np.savez(path, number=np.array(numbers),
             time=np.array([values[n][0] for n in numbers]),
             **{q: np.array([values[n][1][q] for n in numbers]) for q in QUANTITIES},
             T_cold=T_cold, s_min=s_min)


# ── Plot ─────────────────────────────────────────────────────────────────

def _power(x: float) -> str:
    """LaTeX for x as m x 10^e (or 10^e when m = 1)."""
    mantissa, exponent = f"{x:e}".split("e")
    m = float(mantissa)
    return f"10^{{{int(exponent)}}}" if m == 1 else rf"{m:g}\times10^{{{int(exponent)}}}"


def _power_text(x: float) -> str:
    """Factor for a legend: 0.22, 4.5, or m x 10^e outside [0.01, 1000)."""
    if 0.01 <= abs(x) < 1000:
        return f"{x:.2g}"
    return f"${_power(float(f'{x:.2g}'))}$"


def match_factors(data: dict, quantities: list[str], t_match: float) -> dict[str, float]:
    """
    Per quantity, the factor making it equal data[ANALYTIC] at t_match (same
    time units as data["time"]); frame values are interpolated linearly.
    Quantities that are zero there, or missing, are left out with a message.
    """
    if ANALYTIC not in data:
        return {}
    t = data["time"]
    if not t.min() <= t_match <= t.max():
        print(f"  match_at {t_match} is outside the frame times {t.min():.3g}..{t.max():.3g}")
        return {}
    target = np.interp(t_match, data["time_" + ANALYTIC], data[ANALYTIC])
    factors = {}
    for q in quantities:
        if q not in data or q == ANALYTIC:
            continue
        value = np.interp(t_match, t, data[q])
        if value > 0:
            factors[q] = float(target / value)
        else:
            print(f"  {q} is 0 at t = {t_match}; not rescaled")
    return factors


def quantity_style(q: str, T_cold: float, s_min: float) -> dict:
    """config.timeseries_var(q) with <T_cold> and <s_min> in the label filled in
    (<factor> is filled in per run when plotting)."""
    style = config.timeseries_var(q)
    style["label"] = (style["label"].replace("<T_cold>", _power(T_cold))
                      .replace("<s_min>", _power(s_min)))
    return style


def plot_timeseries(series: dict[str, dict], labels: dict[str, str], opts: dict,
                    path: Path) -> Path:
    """
    Draw ``series`` {run: {"time": t, <quantity>: mass, ...}}, already in
    opts' time and mass units, styled by config.TIMESERIES_VARS.  Several
    runs overlaid: one axes, colour = run, line style = quantity.  Otherwise
    (one run, or one panel per run) each quantity has its own colour too.
    """
    apply_fonts()
    runs = list(series)
    n_panels = 1 if opts["overlay"] else len(runs)
    width, height = opts["figsize"]
    fig, axes = plt.subplots(n_panels, 1, sharex=True, squeeze=False,
                             figsize=(width, height * n_panels), layout="constrained")
    axes = axes[:, 0]
    units = {"Msun": r"M$_\odot$"}.get(opts["mass_units"], opts["mass_units"])
    styles = {q: quantity_style(q, opts["T_cold"], opts["s_min"]) for q in opts["quantities"]}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    names = opts["quantities"]
    by_run = opts["overlay"] and len(runs) > 1  # colour tells runs apart

    # Overlaid runs with the same injection share one analytic curve, drawn
    # with the run reaching the latest time; {drawing run: runs it stands for}.
    analytic_for = {}
    if by_run and ANALYTIC in names:
        groups = {}
        for run in runs:
            if ANALYTIC in series[run]:
                groups.setdefault(series[run]["analytic_key"], []).append(run)
        for group in groups.values():
            longest = max(group, key=lambda r: series[r]["time_" + ANALYTIC][-1])
            analytic_for[longest] = group
        shared = len(groups) == 1

    for n, run in enumerate(runs):
        ax = axes[0 if opts["overlay"] else n]
        data = series[run]
        for k, q in enumerate(names):
            if q not in data:
                continue
            if by_run and q == ANALYTIC:
                if run not in analytic_for:
                    continue  # drawn with another run of the same injection
                style = styles[q]
                group = ", ".join(labels.get(r, r) for r in analytic_for[run])
                ax.plot(data["time_" + q], data[q], linestyle=style["linestyle"],
                        color=(style["color"] or "k") if shared else colors[n % len(colors)],
                        marker=style["marker"], markersize=style["markersize"],
                        label=style["label"] if shared else f"{group}: {style['label']}")
                continue
            t = data["time_" + ANALYTIC] if q == ANALYTIC else data["time"]
            m = data[q]
            if opts["logy"]:  # e.g. no cold gas yet at t = 0: leave those points out
                m = np.where(m > 0, m, np.nan)
            style = styles[q]
            label = (f"{labels.get(run, run)}: {style['label']}" if by_run else style["label"])
            factor = data.get("factors", {}).get(q)
            label = label.replace("<factor>", "" if factor is None
                                  else rf" $\times$ {_power_text(factor)}")
            color = (colors[n % len(colors)] if by_run
                     else style["color"] or colors[k % len(colors)])
            ax.plot(t, m, linestyle=style["linestyle"], color=color, marker=style["marker"],
                    markersize=style["markersize"], label=label)
        if opts["logy"]:
            ax.set_yscale("log")
        ax.set_ylabel(f"M [{units}]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")
        if not opts["overlay"] or len(runs) == 1:
            ax.set_title(labels.get(run, run))
    axes[-1].set_xlabel(f"t [{opts['time_units']}]")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=opts["dpi"])
    plt.close(fig)
    return path


# ── Command line ─────────────────────────────────────────────────────────

def main(argv=None):
    missing = [n for n in ("TIMESERIES", "timeseries_var") if not hasattr(config, n)]
    if missing:
        raise SystemExit(f"main.py timeseries: config.py has no {', '.join(missing)} "
                         "(see config2.py for the time-series settings)")
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
    parser.add_argument("--quantities", nargs="+", choices=CHOICES,
                        default=ts["quantities"],
                        help=f"masses to plot (default: {ts['quantities']})")
    parser.add_argument("--match", nargs="*", choices=QUANTITIES,
                        default=ts.get("match", []),
                        help="quantities rescaled to equal `analytic` at --match-at; "
                             f"no value = none (default: {ts.get('match', [])})")
    parser.add_argument("--match-at", type=float, default=ts.get("match_at", 1.0),
                        help="time of the match, in --time-units "
                             f"(default: {ts.get('match_at', 1.0)})")
    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction,
                        default=ts["overlay"],
                        help="all runs on one axes; --no-overlay: one panel per run "
                             f"(default: {ts['overlay']})")
    parser.add_argument("--T-cold", dest="T_cold", type=float, default=ts["T_cold"],
                        help=f"cold means T below this, in K (default: {ts['T_cold']:g})")
    parser.add_argument("--s-min", dest="s_min", type=float, default=ts["s_min"],
                        help=f"wind-polluted means s_00 above this (default: {ts['s_min']:g})")
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
                             outputs=settings["outputs"], basename=settings.get("basename"))
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
        caches[run] = cache_path(settings["out"], args.T_cold, args.s_min)
        cached[run] = {} if args.recompute else read_cache(caches[run])

    jobs = [(run, f, sims[run].params, args.T_cold, args.s_min)
            for run in runs for f in frames[run] if f.number not in cached[run]]
    for run in runs:
        n_new = sum(1 for j in jobs if j[0] == run)
        print(f"{run}: {len(frames[run])} frames, {len(frames[run]) - n_new} cached "
              f"({caches[run]})")

    if jobs:
        print(f"reading {len(jobs)} frames with {min(args.workers, len(jobs))} workers")

        def store(result):
            run, number, time, m = result
            cached[run][number] = (time, m)
            print(f"  {run} frame {number:5d}: t = {time:.4g}, "
                  + ", ".join(f"{q} = {m[q]:.4g}" for q in QUANTITIES) + " (code)",
                  flush=True)

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
                write_cache(caches[run], cached[run], args.T_cold, args.s_min)

    series = {}
    for run in runs:
        params = sims[run].params
        tf = unit_factor(params, "time", args.time_units)
        mf = unit_factor(params, "mass", args.mass_units)
        numbers = [f.number for f in frames[run]]
        t = np.array([cached[run][n][0] for n in numbers])
        data = {"time": t * tf}
        for q in args.quantities:
            if q == ANALYTIC:
                t_fine = np.linspace(0.0, t.max(), 400)
                m = analytic_mass(params, t_fine)
                if m is None:
                    print(f"{run}: no Mdot0/dt_mdot in <problem>, {q} left out")
                else:
                    data[q], data["time_" + q] = m * mf, t_fine * tf
                    problem = params["problem"]  # same physical Mdot(t) <=> same curve
                    data["analytic_key"] = (float(problem["Mdot0"]),
                                            round(float(problem["dt_mdot"]) * tf, 6))
                continue
            if q == ENTRAINED:  # computed after the match below, from the matched cold mass
                continue
            m = np.array([cached[run][n][1][q] for n in numbers])
            if np.all(np.isnan(m)):
                print(f"{run}: no s_00 tracer, {q} left out")
                continue
            data[q] = m * mf
        want_entrained = ENTRAINED in args.quantities
        if want_entrained and "cold" not in data:  # needed even if cold is not plotted
            data["cold"] = np.array([cached[run][n][1]["cold"] for n in numbers]) * mf
        if args.match:
            if ANALYTIC not in data:  # the match needs the curve even if not plotted
                m = analytic_mass(params, np.linspace(0.0, t.max(), 400))
                if m is not None:
                    data["time_" + ANALYTIC] = np.linspace(0.0, t.max(), 400) * tf
                    data[ANALYTIC] = m * mf
            factors = match_factors(data, args.match, args.match_at)
            if not factors:
                print(f"{run}: nothing rescaled (needs Mdot0/dt_mdot and a nonzero "
                      f"{args.match} at t = {args.match_at})")
            for q, f in factors.items():
                data[q] = data[q] * f
                print(f"{run}: {q} x {f:.3g} to match analytic at t = {args.match_at} "
                      f"{args.time_units}")
            data["factors"] = factors
        if want_entrained:  # (matched) cold - injected, at the frame times
            injected = analytic_mass(params, t)
            if injected is None:
                print(f"{run}: no Mdot0/dt_mdot in <problem>, {ENTRAINED} left out")
            else:
                data[ENTRAINED] = data["cold"] - injected * mf
                if "cold" in data.get("factors", {}):
                    data["factors"][ENTRAINED] = data["factors"]["cold"]  # for <factor>
        if "cold" not in args.quantities:
            data.pop("cold", None)
        if ANALYTIC not in args.quantities:
            data.pop(ANALYTIC, None)
        series[run] = data

    outdir = Path(args.outdir) if args.outdir else config.resolve(runs[0])["out"] / "timeseries"
    opts = {k: getattr(args, k) for k in
            ("quantities", "overlay", "T_cold", "s_min", "mass_units",
             "time_units", "logy", "figsize", "dpi")}
    path = plot_timeseries(series, ts.get("labels") or {}, opts,
                           outdir / f"{args.name}.{args.format}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
