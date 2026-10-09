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
panels of the combined figure and its video) and MAIN (what `python main.py`
makes).

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
    make_slices=True,    # one image per frame and variable in PLOT_ORDER
    make_combined=True,  # one image per frame of COMBINED["layout"]
    make_video=True,     # a video of the combined images
    save_png=False,       # False: no slices, and the combined images are deleted once in the video
)

# Parallel worker processes for plotting frames (slices and combined figures);
# 1 = serial.  Each worker holds one frame of every run it plots in memory, so
# lower this for large runs.  -w/--workers overrides it for one run.
N_WORKERS = 4

RUN = "wind_bondi"  # default simulation

SIMULATIONS = {
    str(RUN): dict(
        athinput="wind_outflow/wind_rad_bondi_2d.athinput",          # relative to $ATHINPUT_DIR
        data="build_wind_rad_bondi/src/wind_rad_bondi_10kyr_2d/bin",       # relative to $DATA_DIR
        out="build_wind_rad_bondi/src/wind_rad_bondi_10kyr_2d/plots",  # relative to $DATA_DIR
        outputs=None,  # output ids, e.g. ["hydro_w"]; None = auto (see SimulationData)
        frames=None,   # frame numbers, e.g. range(0, 21, 5); None = all
    ),
    # "cbox_tabcool": dict(
    #     athinput="cooling_box/cbox.athinput",
    #     data="build_cooling_box/src/cbox_tabcool",
    #     out="build_cooling_box/src/cbox_tabcool/plots",
    #     outputs=None,
    #     frames=None,
    # ),
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
    position=None,        # slice position along axis, in length_units; None = midplane
    length_units="pc",    # axis coordinates: code, cm, pc, kpc
    time_units="kyr",     # time in titles: code, s, yr, kyr, Myr
    time_decimals=0,      # decimal places of the time in titles (0 = whole numbers)
    fig_size_single=(16.0, 4.0),
    panel_size=(16, 4),  # combined figure: size of each panel
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
        cmap="Greens", norm="log", vmin=1.0e-5, vmax=1.0e-1,
    ),
    "pres": dict(
        label=r"Pressure [$\mathbf{dyne/cm^2}$]",
        quantity="pres", units="dyne/cm^2",
        cmap="viridis", norm="log", vmin=1.0e-14, vmax=1.0e-10,
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
        cmap="coolwarm", norm="log", vmin=1.0e4, vmax=1.0e9,
    ),
    "t_cool": dict(
        label="Cooling Time [Myr]",
        quantity="t_cool", units="Myr",
        cmap="turbo", norm="log", vmin=1.0, vmax=1.0e6,
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
    # "tracer",
    "velx",
    "vely",
    # "entropy",
    "t_cool",
]

# Combined figure (python main.py combined) and its video (python main.py video).
# layout is the grid of panels as a list of rows; each cell is
#   "temp"                    a PLOT_VARS key (or quantity), plotted for RUN
#   ("cbox_tabcool", "temp")  a (run, variable) pair, for comparing runs
#   None                      an empty panel
# A row (or column) holding one variable throughout shares one colorbar.
COMBINED = dict(
    name="combined",  # images <name>.<axis>.<NNNNN>.<format>, video <name>.<axis>.mp4
    layout=[
        ["dens","temp"],
        ["pres","t_cool"],
        # ["entropy","tracer"],
        ["velx","vely"],
        # ["vely"],
    ],
    fps=24,           # video frames per second
    video_max_size=4096,  # video frames scaled down to at most this many pixels per side
                          # (the encoders fail above 4096); None/0 = full size
)
# Comparing runs, one row per variable and one column per run:
# COMBINED["layout"] = [[(run, var) for run in ["cbox_02Myr", "cbox_tabcool"]]
#                       for var in ["dens", "temp"]]


def plot_var(name: str) -> dict:
    """
    Settings of PLOT_VARS[name]; a name not in PLOT_VARS is taken as a
    quantity and plotted in code units with a linear viridis map.
    """
    defaults = dict(label=name, quantity=name, units="code",
                    cmap="viridis", norm=None, vmin=None, vmax=None)
    return {**defaults, **PLOT_VARS.get(name, {})}
