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
    make_slices=False,    # one image per frame and variable in PLOT_ORDER
    make_combined=False,  # one image per frame of COMBINED["layout"]
    make_split=True,    # True: the split figure (SPLIT) instead of the combined one and its video
    make_video=False,     # a video of the combined images
    save_png=False,       # False: no slices, and the combined images are deleted once in the video
    make_timeseries=False,  # gas masses vs time of TIMESERIES["runs"]
    make_emission=False,          # False: no slices, and the combined images are deleted once in the video
)

# Parallel worker processes for plotting frames (slices and combined figures);
# 1 = serial.  Each worker holds one frame of every run it plots in memory, so
# lower this for large runs.  -w/--workers overrides it for one run.
N_WORKERS = 4

RUN = "nowind_3d"  # default simulation

SIMULATIONS = {
    "nowind_3d": dict(
        athinput="no_wind/no_wind_zoom.athinput",          # relative to $ATHINPUT_DIR
        data="/home/meemik/Meemik/athenak_MR/build_no_wind_sp/src/no_wind_zoom/bin",       # relative to $DATA_DIR
        out="/home/meemik/Meemik/athenak_MR/build_no_wind_sp/src/no_wind_zoom/plots",  # relative to $DATA_DIR
        outputs=None,  # output ids, e.g. ["hydro_w"]; None = auto (see SimulationData)
        frames=[150],   # frame numbers, e.g. range(0, 21, 5); None = all
    ),
    "wind_2d": dict(
        athinput="wind_outflow/wind_rad_bondi_2d.athinput",          # relative to $ATHINPUT_DIR
        data="build_wind_rad_bondi/src/wind_rad_bondi_10kyr_2d/bin",       # relative to $DATA_DIR
        out="build_wind_rad_bondi/src/wind_rad_bondi_10kyr_2d/plots",  # relative to $DATA_DIR
        outputs=None,  # output ids, e.g. ["hydro_w"]; None = auto (see SimulationData)
        frames=None,   # frame numbers, e.g. range(0, 21, 5); None = all
    ),
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
    title=14,       # panel titles (the variable labels)
    suptitle=16,    # time at the top of the figure
    label=12,       # axis labels
    ticks=10,       # tick labels of axes and colorbars
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
        cmap="Greens", norm="log", vmin=1.0e-4, vmax=1.0e-2,
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
        cmap="coolwarm", norm="log", vmin=1.0e4, vmax=1.0e8,
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
# COMBINED = dict(
#     name="compare",  # images <name>.<axis>.<NNNNN>.<format>, video <name>.<axis>.mp4
#     layout=[[(run, var) for run in ["wind_3d", "wind_2d"]]
#                       for var in ["dens", "temp","velx","vely"]],
#     fps=24,           # video frames per second
#     video_max_size=4096,  # video frames scaled down to at most this many pixels per side
#                           # (the encoders fail above 4096); None/0 = full size
# )
# Comparing runs, one row per variable and one column per run:
# COMBINED["layout"] = [[(run, var) for run in ["cbox_02Myr", "cbox_tabcool"]]
#                       for var in ["dens", "temp"]]

# Split figure (python main.py split): one slice cut in two, a different
# variable on each side of the cut, each with its own colorbar on the right
# (e.g. density above y = 0 and temperature below).  Uses PLOT["axis"] /
# PLOT["position"] for the slice and length_units for the coordinates below.
SPLIT = dict(
    name="split",          # images <name>.<axis>.<NNNNN>.<format> in <out>/split/
    vars=["dens", "temp"], # PLOT_VARS keys (their cmap, norm, vmin/vmax, unless styled below):
                           # [top, bottom] for a horizontal cut, [left, right] for a vertical one
    # Style of each variable in this figure only (PLOT_VARS stays as it is for the
    # other plots): any of cmap, norm, vmin, vmax, label, plus `under`, the colour of
    # values below vmin (e.g. the sink, held far below the ambient density), and
    # `vcenter`, the value at the colormap's middle (for diverging maps; log or linear).
    # Ranges fitted to the data so the colormaps' full range is used: viridis/inferno
    # change brightness steadily and look clearly different from each other.
    styles={
        "dens": dict(cmap="viridis", vmin=1.0e-4, vmax=1.0e-2, under="k"),
        # Ambient T_cgm = 1e6 K at the map's middle: cooled gas blue, shocked gas red,
        # each side using its full half however many decades it spans.
        "temp": dict(cmap="coolwarm", vmin=1.0e4, vcenter=1.0e6, vmax=1.0e7),
    },
    cut="horizontal",      # "horizontal": cut along the plot's x axis (top/bottom halves);
                           # "vertical": cut along the plot's y axis (left/right halves)
    at=0.0,                # cut position along the plot's y (horizontal) or x (vertical)
                           # axis, in length_units; None = middle of the shown range
    mirror=False,          # False: each variable on its own true half;
                           # True: both show the first variable's half, the second reflected
    xlim=None,             # (min, max) of the plot's x axis in length_units; None = full domain
    ylim=None,             # (min, max) of the plot's y axis; None = full domain
    cut_line=dict(color="k", linewidth=0.5),  # line along the cut; None = no line
    labels=None,           # colorbar labels [first, second]; None = the PLOT_VARS labels
                           # (short ones read better on a paper figure, e.g. r"$n$ [cm$^{-3}$]")
    cbar_length=1.0,       # colorbar length, fraction of the plot's width (height for a
                           # vertical cut); horizontal cut: first's above, second's below
    title=True,            # time as the title
    fig_size=(7.0, 2.0),   # inches; ~7 in = two-column paper width, ~3.5 in = one column
    dpi=300,               # for raster formats
    format="png",          # pdf/svg for the paper (vector), png for quick looks
)


def plot_var(name: str) -> dict:
    """
    Settings of PLOT_VARS[name]; a name not in PLOT_VARS is taken as a
    quantity and plotted in code units with a linear viridis map.
    """
    defaults = dict(label=name, quantity=name, units="code",
                    cmap="viridis", norm=None, vmin=None, vmax=None)
    return {**defaults, **PLOT_VARS.get(name, {})}
