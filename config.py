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
panels of the combined figure and its video), TIMESERIES (cold mass vs time)
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
    make_timeseries=True,  # cold mass vs time of TIMESERIES["runs"]
)

# Parallel worker processes for plotting frames (slices and combined figures);
# 1 = serial.  Each worker holds one frame of every run it plots in memory, so
# lower this for large runs.  -w/--workers overrides it for one run.
N_WORKERS = 2

# RUN = "explicit"  # default simulation
RUN = "turb_sigv_50"  # default simulation

SIMULATIONS = {
    "turb_tcgm_8192": dict(
        athinput="/data/data_meemik/RunawayBH/sim_data/turb_tcgm/rudra_runs/turb_tcgm_highres/turb_tcgm_125998.athinput",          # relative to $ATHINPUT_DIR
        data="/data/data_meemik/RunawayBH/sim_data/turb_tcgm/rudra_runs/turb_tcgm_highres",       # relative to $DATA_DIR
        out="/data/data_meemik/RunawayBH/sim_data/turb_tcgm/rudra_runs/turb_tcgm_highres/plots",  # relative to $DATA_DIR
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
    "turb_sigv_50": dict(
            athinput="/home/meemik/Meemik/athenak_MR/inputs/hydro/turb_tcgm/turb_tcgm_sigv50.athinput",          # relative to $ATHINPUT_DIR
            data="/home/meemik/Meemik/athenak_MR/build_turb_tcgm_sp/src/turb_tcgm_sigv_50",       # relative to $DATA_DIR
            out="/home/meemik/Meemik/athenak_MR/build_turb_tcgm_sp/src/turb_tcgm_sigv_50/plots",  # relative to $DATA_DIR
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
COMBINED = dict(
    name="compare",  # images <name>.<axis>.<NNNNN>.<format>, video <name>.<axis>.mp4
    layout=[[(run, var) for run in ["turb_sigv_30", "turb_sigv_50"]]
                      for var in ["dens", "temp","velx","vely"]],
    fps=24,           # video frames per second
    video_max_size=4096,  # video frames scaled down to at most this many pixels per side
                          # (the encoders fail above 4096); None/0 = full size
)


# Cold gas mass vs time (python main.py timeseries).  Values are cached per run
# in <out>/timeseries/, so adding a run to the plot only reads that run.
TIMESERIES = dict(
    name="cold_mass",     # figure <outdir>/<name>.<format>; outdir = timeseries/ of the first run
    runs=["turb_sigv_30"],  # runs to plot; None = RUN
    overlay=True,         # True: one axes, a line per run; False: a panel per run, stacked
    labels={},            # run -> legend/panel label, e.g. {"turb_sigv_30": r"$\sigma_v$ = 30 km/s"}
    T_cold=5e4,           # K; cold gas is T < T_cold
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
