"""
Simulations the scripts can work on, and which one they use by default.

Paths are relative to root folders given by the environment variables
``ATHINPUT_DIR`` and ``DATA_DIR`` (machine-specific; ``DATA_DIR`` defaults to
``ATHENAK_DIR``), so this file is the
same on every machine.  They come from the shell or a ``.env`` file (copy
``.env.example``); see ``utils/env.py``.  An
absolute path in an entry is used as is.  Scripts take ``--run <name>`` to
pick another entry, and flags to override any single value.

The second half sets what the plotting scripts draw: PLOT (figure-wide
settings), PLOT_VARS (one entry per variable that can be plotted) and
PLOT_ORDER (which of them are plotted by default), COMBINED (the grid of
panels of the combined figure and its video), TIMESERIES (gas masses vs time)
and MAIN (what `python main.py` makes).

    from config import resolve
    run = resolve()            # paths of RUN
    run = resolve("cbox_02Myr")
    sim = SimulationData(run["athinput"], run["data"], outputs=run["outputs"])
"""

from __future__ import annotations

if __package__:  # imported as vis_athenak.config
    from .utils.env import env_path
else:            # run from the repo root
    from utils.env import env_path
    
# What `python main.py` (no command) makes; each switch can be flipped on the
# command line, e.g. --no-make-slices or --save-png.
MAIN = dict(
    make_slices=False,    # one image per frame and variable in PLOT_ORDER
    make_combined=False,  # one image per frame of COMBINED["layout"]
    make_video=False,     # a video of the combined images
    save_png=False,       # False: no slices, and the combined images are deleted once in the video
    make_timeseries=True,  # gas masses vs time of TIMESERIES["runs"]
)

# Parallel worker processes for plotting frames (slices and combined figures);
# 1 = serial.  Each worker holds one frame of every run it plots in memory, so
# lower this for large runs.  -w/--workers overrides it for one run.
N_WORKERS = 2

# RUN = "explicit"  # default simulation
RUN = "turb_sigv_75"  # default simulation

SIMULATIONS = {
    "no_turb": dict(
        athinput="/home/meemik/Meemik/athenak_MR/inputs/hydro/turb_tcgm/tcgm_noturb.athinput",          # relative to $ATHINPUT_DIR
        data="/data/data_meemik/RunawayBH/sim_data/tCGM/athenak_3D_runs/tcgm03_mdot05_vw2",       # relative to $DATA_DIR
        out="/data/data_meemik/RunawayBH/sim_data/tCGM/athenak_3D_runs/tcgm03_mdot05_vw2/plots",  # relative to $DATA_DIR
        outputs=None,  # output ids, e.g. ["hydro_w"]; None = auto (see SimulationData)
        frames=None,   # frame numbers, e.g. range(0, 21, 5); None = all
    ),
    "turb_sigv_30": dict(
            athinput="/home/meemik/Meemik/athenak_MR/inputs/hydro/turb_tcgm/turb_tcgm.athinput",          # relative to $ATHINPUT_DIR
            data="/data/data_meemik/RunawayBH/sim_data/turb_tcgm/athenak_3d_runs/turb_tcgm_single",       # relative to $DATA_DIR
            out="/data/data_meemik/RunawayBH/sim_data/turb_tcgm/athenak_3d_runs/turb_tcgm_single/plots",  # relative to $DATA_DIR
            outputs=None,  # output ids, e.g. ["hydro_w"]; None = auto (see SimulationData)
            frames=None,   # frame numbers, e.g. range(0, 21, 5); None = all
        ),
    "turb_sigv_75": dict(
            athinput="/home/meemik/Meemik/athenak_MR/inputs/hydro/turb_tcgm/turb_tcgm_sigv75.athinput",          # relative to $ATHINPUT_DIR
            data="/data/data_meemik/RunawayBH/sim_data/turb_tcgm/athenak_3d_runs/turb_tcgm_sigv_75",       # relative to $DATA_DIR
            out="/data/data_meemik/RunawayBH/sim_data/turb_tcgm/athenak_3d_runs/turb_tcgm_sigv_75/plots",  # relative to $DATA_DIR
            outputs=None,  # output ids, e.g. ["hydro_w"]; None = auto (see SimulationData)
            frames=None,   # frame numbers, e.g. range(0, 21, 5); None = all
        )
}


def resolve(run: str = RUN) -> dict:
    """Settings of ``run`` with ``athinput``, ``data``, ``out`` as absolute Paths."""
    if run not in SIMULATIONS:
        raise KeyError(f"unknown run {run!r}; config.SIMULATIONS has {list(SIMULATIONS)}")
    sim = dict(SIMULATIONS[run])
    sim["athinput"] = env_path("ATHINPUT_DIR") / sim["athinput"]
    data_root = env_path("DATA_DIR", fallback="ATHENAK_DIR")
    sim["data"] = data_root / sim["data"]
    sim["out"] = data_root / sim["out"]
    return sim


# ── Plotting ─────────────────────────────────────────────────────────────

PLOT = dict(
    axis="z",             # slice normal: x, y or z (z for 2D runs)
    position=None,        # slice position along axis, in length_units (degrees for an
                          # angle, e.g. phi with axis y in cylindrical); None = midplane
    length_units="kpc",    # axis coordinates: code, cm, pc, kpc
    time_units="Myr",     # time in titles: code, s, yr, kyr, Myr
    time_decimals=0,      # decimal places of the time in titles (0 = whole numbers)
    fig_size_single=(16.0, 1.5),
    panel_size=(16, 1.8),  # combined figure: size of each panel
    dpi=300,
    format="png",
)

# Fonts of every figure (sizes in points).
FONTS = dict(
    weight="bold",  # "bold" or "normal"; also applies to math text such as 10^{-4}
    title=16,       # panel titles (the variable labels)
    suptitle=18,    # time at the top of the figure
    label=14,       # axis labels
    ticks=12,       # tick labels of axes and colorbars
)

# One entry per plottable variable; the key names the output folder and the
# CLI argument.  Fields:
#   quantity  raw variable (dens, velx, eint, s_00, ...) or derived one
#             (pres, temp, vmag, bmag, beta, entropy, t_cool); see plotting/quantities.py
#   units     "code", or a unit of the quantity's dimension (utils/units.py UNIT_NAMES):
#             density g/cm^3 or cm^-3, pressure dyne/cm^2, velocity km/s or cm/s,
#             temperature K, time s/yr/kyr/Myr, magnetic G/uG
#   norm      "log" or None (linear)
#   vmin/vmax colour limits in those units; None = autoscale each frame
PLOT_VARS = {
    "dens": dict(
        label=r"Density [$\mathbf{m_p/cm^3}$]",
        quantity="dens", units="cm^-3",
        cmap="Greens", norm="log", vmin=1.0e-4, vmax=1.0e1,
    ),
    "pres": dict(
        label=r"Pressure [$\mathbf{dyne/cm^2}$]",
        quantity="pres", units="dyne/cm^2",
        cmap="viridis", norm="log", vmin=1.0e-12, vmax=1.0e-8,
    ),
    "entropy": dict(
        label="Entropy [code]",
        quantity="entropy", units="code",
        cmap="magma", norm=None, vmin=-20.0, vmax=0.0,
    ),
    "velx": dict(
        label="X Velocity [km/s]",
        quantity="velx", units="km/s",
        cmap="Blues_r", norm=None, vmin=-1.0e3, vmax=0.0,
    ),
    "vely": dict(
        label="Y Velocity [km/s]",
        quantity="vely", units="km/s",
        cmap="seismic", norm=None, vmin=-500.0, vmax=500.0,
    ),
    "temp": dict(
        label="Temperature [K]",
        quantity="temp", units="K",
        cmap="coolwarm", norm="log", vmin=1.0e4, vmax=1.0e7,
    ),
    "t_cool": dict(
        label="Cooling Time [Myr]",
        quantity="t_cool", units="Myr",
        cmap="turbo", norm="log", vmin=1.0e-3, vmax=1.0e3,
    ),
    "emissivity": dict(
        label=r"Emissivity $\mathbf{n^2\Lambda(T)}$ [$\mathbf{erg/cm^3/s}$]",
        quantity="emissivity", units="erg/cm^3/s",
        cmap="inferno", norm="log", vmin=1.0e-18, vmax=1.0e-22,
    ),
    "tracer": dict(
        label="Outflow Tracer",
        quantity="s_00", units="code",
        cmap="winter", norm="log", vmin=1.0e-5, vmax=1.0,
    ),
}

# Style of each quantity of the time series (python main.py timeseries), like
# PLOT_VARS for the slices but kept apart, since these are box totals, not
# fields that can be sliced.  Fields:
#   label      legend text, used as written except for these placeholders:
#              <T_cold>, <s_min>  TIMESERIES' thresholds (as LaTeX, so inside $...$)
#              <factor>           " x 0.22" if TIMESERIES["match"] rescaled the
#                                 quantity, else nothing; leave it out to never show it
#   color      matplotlib colour; when several runs are overlaid the colour
#              tells the runs apart instead, and the line style the quantity
#   linestyle  "-", "--", "-.", ":" or a dash tuple like (0, (6, 2, 1, 2))
#   marker     marker at each frame, or None; markersize in points
TIMESERIES_VARS = {
    "cold": dict(
        label=r"cold ($T < <T_cold>$ K)",
        color="C0", linestyle="-", marker="o", markersize=2.5,
    ),
    "wind": dict(
        label=r"wind-polluted ($s_{00} > <s_min>$)<factor>",
        color="C1", linestyle="--", marker="o", markersize=2.5,
    ),
    "cold_wind": dict(
        label="cold & polluted<factor>",
        color="C2", linestyle="-.", marker="o", markersize=2.5,
    ),
    "wind_tracer": dict(
        label=r"wind material ($\rho\,s_{00}$)<factor>",
        color="C3", linestyle=(0, (6, 2, 1, 2, 1, 2)), marker="o", markersize=2.5,
    ),
    "analytic": dict(
        label=r"injected $\int\dot M\,dt$, $\dot M_0 e^{-t/\tau}$",
        color="k", linestyle=":", marker=None, markersize=2.5,
    ),
    "entrained": dict(
        label=r"entrained CGM (cold $-$ injected)",
        color="C4", linestyle=(0, (3, 1, 1, 1)), marker="o", markersize=2.5,
    ),
}

# Variables plotted when none are named on the command line.
PLOT_ORDER = [
    "dens",
    "pres",
    "temp",
    "tracer",
    "velx",
    "vely",
    "entropy",
    "t_cool",
    "emissivity",
]

# Combined figure (python main.py combined) and its video (python main.py video).
# layout is the grid of panels as a list of rows; each cell is
#   "temp"                    a PLOT_VARS key (or quantity), plotted for RUN
#   ("cbox_tabcool", "temp")  a (run, variable) pair, for comparing runs
#   None                      an empty panel
# A row (or column) holding one variable throughout shares one colorbar.
# COMBINED = dict(
#     name="combined",  # images <name>.<axis>.<NNNNN>.<format>, video <name>.<axis>.mp4
#     layout=[
#         ["dens","temp"],
#         ["pres","t_cool"],
#         ["tracer","emissivity"],
#         ["velx","vely"],
#         # ["vely"],
#     ],
#     fps=24,           # video frames per second
#     video_max_size=4096,  # video frames scaled down to at most this many pixels per side
#                           # (the encoders fail above 4096); None/0 = full size
# )
# COMBINED = dict(
#     name="combined",  # images <name>.<axis>.<NNNNN>.<format>, video <name>.<axis>.mp4
#     layout=[
#         ["dens"],
#         ["temp"],
#         # ["pres"],
#         ["t_cool"],
#         ["tracer"],
#         # ["emissivity"],
#         ["velx"],
#         ["vely"],
#         # ["vely"],
#     ],
#     fps=24,           # video frames per second
#     video_max_size=4096,  # video frames scaled down to at most this many pixels per side
#                           # (the encoders fail above 4096); None/0 = full size
# )
# Comparing runs, one row per variable and one column per run:
# COMBINED = dict(
#     name="compare_sigv_30_75",  # images <name>.<axis>.<NNNNN>.<format>, video <name>.<axis>.mp4
#     layout=[[(run, var) for run in ["turb_sigv_30", "turb_sigv_75"]]
#                       for var in ["dens", "temp","velx","vely"]],
#     fps=24,           # video frames per second
#     video_max_size=4096,  # video frames scaled down to at most this many pixels per side
#                           # (the encoders fail above 4096); None/0 = full size
# )


# Gas masses vs time (python main.py timeseries).  Values are cached per run
# in <out>/timeseries/, so adding a run to the plot only reads that run.
TIMESERIES = dict(
    name="compare_mass_vs_time",  # figure <outdir>/<name>.<format>; outdir = timeseries/ of the first run
    # runs=["turb_sigv_75"],  # runs to plot; None = RUN
    runs=["no_turb","turb_sigv_30", "turb_sigv_75"],  # runs to plot; None = RUN
    quantities=["cold", "analytic"],  # any of: cold (T < T_cold),
    # quantities=["cold", "analytic","entrained"],  # any of: cold (T < T_cold),
    
                          # wind (s_00 > s_min), cold_wind (both), wind_tracer (sum of rho s_00 dV),
                          # analytic (injected wind mass Mdot0 dt_mdot (1 - e^-t/dt_mdot) for
                          # Mdot(t) = Mdot0 e^-t/dt_mdot, both from the athinput's <problem>),
                          # entrained (entrained CGM = cold - analytic, using the matched cold mass when cold is in match)
    match=["cold"],       # quantities rescaled to equal `analytic` at time match_at (a constant
                          # factor per run, shown in the legend); [] = no rescaling
    match_at=1.0,         # in time_units
    overlay=True,        # True: one axes (colour = run, line style = quantity);
                          # False: a panel per run, stacked
    labels={},            # run -> legend/panel label, e.g. {"turb_sigv_30": r"$\sigma_v$ = 30 km/s"}
    T_cold=2e4,           # K; cold gas is T < T_cold
    s_min=1e-3,           # wind-polluted gas is s_00 > s_min (as the run's cooling mask)
    mass_units="Msun",    # g or Msun
    time_units="Myr",     # None = PLOT["time_units"]
    logy=True,            # logarithmic mass axis
    figsize=(8.0, 4.5),   # inches; height is per panel when overlay is False
)


def plot_var(name: str) -> dict:
    """
    Settings of PLOT_VARS[name]; a name not in PLOT_VARS is taken as a
    quantity and plotted in code units with a linear viridis map.
    """
    defaults = dict(label=name, quantity=name, units="code",
                    cmap="viridis", norm=None, vmin=None, vmax=None)
    return {**defaults, **PLOT_VARS.get(name, {})}


def timeseries_var(name: str) -> dict:
    """Settings of TIMESERIES_VARS[name], with defaults for anything missing."""
    defaults = dict(label=name, color=None, linestyle="-", marker="o", markersize=2.5)
    return {**defaults, **TIMESERIES_VARS.get(name, {})}
