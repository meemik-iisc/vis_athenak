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
panels of the combined figure and its video), SPLIT (one slice cut in two,
a variable on each side), TIMESERIES (gas masses vs time) and MAIN (what
`python main.py` makes).

This version is set up for the fast_wind run (0.1c, 1e-4 Msun/yr wind, BH at
100 km/s through a uniform n = 1, T = 1e6 K CGM; pc box, SMR to 0.5 pc).
Copy it over config.py to use it:  cp config_fast_wind.py config.py

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
    make_slices=True,     # one image per frame and variable in PLOT_ORDER
    make_combined=True,   # one image per frame of COMBINED["layout"]
    make_split=False,     # True: the split figure (SPLIT) instead of the combined one and its video
    make_video=True,      # a video of the combined images
    save_png=False,        # False: no slices, and the combined images are deleted once in the video
    make_timeseries=False,  # gas masses vs time of TIMESERIES["runs"]
    make_cold_mass=False,  # cold gas mass in x slabs vs x of COLD_MASS["runs"]
    make_emission=False,   # projected emission-line maps of EMISSION["run"]
)

# Parallel worker processes for plotting frames (slices and combined figures);
# 1 = serial.  Each worker holds one frame of every run it plots in memory, so
# lower this for large runs.  -w/--workers overrides it for one run.
N_WORKERS = 4

RUN = "capture_cool_n10_nmin2"  # default simulation

SIMULATIONS = {
    # 0.1c wind, BH at 100 km/s, n = 1, T = 1e6 K; R_0 ~ 26 pc. Output every 20 code
    # units (19.6 kyr), so frame N is at t = 0.0196 N Myr; 51 frames to 1 Myr.
    # The data is still written into the build folder; move it to /data later and
    # change data/out here.
    "fast_wind": dict(
        athinput="/home/meemik/Meemik/athenak_MR/inputs/hydro/fast_wind/fast_wind.athinput",
        data="/home/meemik/Meemik/athenak_MR/build_fast_wind/src/fast_wind",
        out="/home/meemik/Meemik/athenak_MR/build_fast_wind/src/fast_wind/plots",
        outputs=None,  # output ids, e.g. ["hydro_w"]; None = auto (see SimulationData)
        frames=None,   # frame numbers, e.g. range(0, 51, 5); None = all
    ),
    # Capture test: 0.03c wind with pdot = 3.5e33 dyn whose shocked gas is cooled
    # (momentum-driven limit), BH at 1000 km/s, n = 1, T = 1e6 K; R_0 = 54 pc,
    # R_cap = 108 pc. Box x in [-1500, 500], y,z in [-500, 500] pc, SMR 7.8/3.9/1.95 pc
    # (1.95 pc in x in [-375, 125], y,z in [-125, 125]). Output every 40 code units
    # (39.1 kyr): frame N at t = 0.0391 N Myr, 51 frames to 1.96 Myr.
    # s_00 = wind fraction, s_01 = impact parameter b [pc] of the inflowing CGM.
    # Captured-gas history (M_halt, Mdot_up, M_halt per b bin): capture_cool.user.hst.
    "capture_cool": dict(
        athinput="/home/meemik/Meemik/athenak_MR/inputs/hydro/fast_wind/capture_cool.athinput",
        data="/home/meemik/Meemik/athenak_MR/build_fast_wind_sp/src/capture_cool",
        out="/home/meemik/Meemik/athenak_MR/build_fast_wind_sp/src/capture_cool/plots",
        outputs=None,  # output ids, e.g. ["hydro_w"]; None = auto (see SimulationData)
        frames=None,   # frame numbers, e.g. range(0, 51, 5); None = all
    ),
    # As capture_cool with n_cgm = 10 and pdot = 3.5e34 dyn (same R_0 = 54 pc, same
    # mesh), density-matched heating below 3e6 K: is the shocked CGM radiative?
    "capture_cool_n10_nmin2": dict(
        athinput="/home/meemik/Meemik/athenak_MR/inputs/hydro/fast_wind/capture_cool_n10.athinput",
        data="/home/meemik/Meemik/athenak_MR/build_fast_wind_sp/src/capture_cool_n10_nmin2",
        out="/home/meemik/Meemik/athenak_MR/build_fast_wind_sp/src/capture_cool_n10_nmin2/plots",
        basename="capture_cool_n10_nmin2",  # run with job/basename=... on the command line;
                                            # optional, default = the athinput's basename
        outputs=None,
        frames=None,
    ),
    "capture_adiabatic": dict(
        athinput="/home/meemik/Meemik/athenak_MR/inputs/hydro/fast_wind/capture_adiabatic.athinput",
        data="/home/meemik/Meemik/athenak_MR/build_fast_wind_sp/src/capture_adiabatic",
        out="/home/meemik/Meemik/athenak_MR/build_fast_wind_sp/src/capture_adiabatic/plots",
        outputs=None,  # output ids, e.g. ["hydro_w"]; None = auto (see SimulationData)
        frames=None,   # frame numbers, e.g. range(0, 51, 5); None = all
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

# capture_cool box: x in [-1500, 500] pc (BH at 0, CGM flows in from +x at 1000 km/s),
# y, z in [-500, 500] pc.
PLOT = dict(
    axis="z",             # slice normal: x, y or z (z for 2D runs)
    position=None,        # slice position along axis, in length_units (degrees for an
                          # angle, e.g. phi with axis y in cylindrical); None = midplane
    length_units="pc",    # axis coordinates: code, cm, pc, kpc
    time_units="Myr",     # time in titles: code, s, yr, kyr, Myr
    time_decimals=2,      # decimal places of the time in titles (frames are 0.04 Myr apart)
    fig_size_single=(12.0, 6.5),  # full 2000 x 1000 pc slice
    panel_size=(10.0, 5.0),  # combined figure: width of each column (incl. its colorbar
                             # and label; with hspace set, the height follows the
                             # 2000 x 1000 pc box: ~6.2 x 3.1 in axes)
    dpi=200,
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
#   norm      "log", "symlog" (log in both signs, linear within +-linthresh) or None
#   linthresh symlog only: half-width of the linear band around 0, in those units
#             (default 1e-3 of the larger of |vmin|, |vmax|)
#   vmin/vmax colour limits in those units; None = autoscale each frame
#   ticks     optional list of colorbar ticks (single slices, combined and split
#             figures; COMBINED/SPLIT "styles" can override it per figure)
#
# Ranges for capture_cool: CGM n = 1 cm^-3 (n = rho/(mu m_u)), T = 1e6 K,
# P = 1.4e-10 dyne/cm^2, ram pressure at the head rho v_bh^2 = 1e-8 dyne/cm^2;
# free wind n ~ 0.4 (r_inj) to 0.01 (R_0), 9000 km/s, cooled to 1e4-1e5 K;
# shocked CGM ~1.4e7 K (n ~ 4), denser if it cools. v_x = -1000 km/s is the
# undisturbed inflow, v_x ~ 0 halted (captured) gas, v_x > 0 gas driven ahead;
# velocities on a symlog scale: +-10 to +-1e4 km/s in decades, linear within +-10.
# (fast_wind used n 1e-5..10, T 1e4..1e11, |v| up to 3e4 km/s.)
PLOT_VARS = {
    "dens": dict(
        label=r"Density [$\mathbf{cm^{-3}}$]",
        quantity="dens", units="cm^-3",
        cmap="viridis", norm="log", vmin=1.0e-3, vmax=1.0e3,
    ),
    "pres": dict(
        label=r"Pressure [$\mathbf{dyne/cm^2}$]",
        quantity="pres", units="dyne/cm^2",
        cmap="viridis", norm="log", vmin=1.0e-11, vmax=1.0e-7,
    ),
    "entropy": dict(
        label="Entropy [code]",
        quantity="entropy", units="code",
        cmap="magma", norm=None, vmin=None, vmax=None,
    ),
    "vmag": dict(
        label="|v| [km/s]",
        quantity="vmag", units="km/s",
        cmap="plasma", norm="log", vmin=1.0e1, vmax=1.0e4,
    ),
    "velx": dict(
        label="X Velocity [km/s]",
        quantity="velx", units="km/s",
        cmap="RdBu_r", norm="symlog", vmin=-1.0e4, vmax=1.0e4, linthresh=10.0,
        ticks=[-1.0e4, -1.0e2, 0.0, 1.0e2, 1.0e4],
    ),
    "vely": dict(
        label="Y Velocity [km/s]",
        quantity="vely", units="km/s",
        cmap="RdBu_r", norm="symlog", vmin=-1.0e4, vmax=1.0e4, linthresh=10.0,
        ticks=[-1.0e4, -1.0e2, 0.0, 1.0e2, 1.0e4],
    ),
    "temp": dict(
        label="Temperature [K]",
        quantity="temp", units="K",
        cmap="coolwarm", norm="log", vmin=1.0e4, vmax=1.0e8,
    ),
    "t_cool": dict(
        label="Cooling Time [Myr]",
        quantity="t_cool", units="Myr",
        cmap="turbo", norm="log", vmin=1.0e-3, vmax=1.0e3,
    ),
    "emissivity": dict(
        label=r"Emissivity $\mathbf{n^2\Lambda(T)}$ [$\mathbf{erg/cm^3/s}$]",
        quantity="emissivity", units="erg/cm^3/s",
        cmap="inferno", norm="log", vmin=1.0e-26, vmax=1.0e-18,
    ),
    # Line emissivities (any line_<name> works, see EMISSION; these set the style)
    "Ha": dict(
        label=r"H$\mathbf{\alpha}$ emissivity [$\mathbf{erg/cm^3/s}$]",
        quantity="line_Ha", units="erg/cm^3/s",
        cmap="inferno", norm="log", vmin=1.0e-32, vmax=1.0e-24,
    ),
    "OIII": dict(
        label=r"[OIII]5007 emissivity [$\mathbf{erg/cm^3/s}$]",
        quantity="line_OIII_5007", units="erg/cm^3/s",
        cmap="inferno", norm="log", vmin=1.0e-34, vmax=1.0e-26,
    ),
    "tracer": dict(
        label="Outflow Tracer",
        quantity="s_00", units="code",
        cmap="BuGn", norm="log", vmin=1.0e-3, vmax=1.0,
    ),
    # Impact parameter b = sqrt(y^2+z^2) the CGM gas had when it entered (pc; code
    # length). R_0 = 54 pc, R_cap = 2 R_0 = 108 pc is the middle of the map: captured
    # gas from b < R_cap should be blue, deflected gas from b > R_cap red.
    "impact": dict(
        label="Impact parameter b [pc]",
        quantity="s_01", units="code",
        cmap="RdBu_r", norm=None, vmin=0.0, vmax=216.0,
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
    "impact",
    "vmag",
    "velx",
    "vely",
    "t_cool",
    "emissivity",
]

# Combined figure (python main.py combined) and its video (python main.py video).
# layout is the grid of panels as a list of rows; each cell is
#   "temp"                    a PLOT_VARS key (or quantity), plotted for RUN
#   ("cbox_tabcool", "temp")  a (run, variable) pair, for comparing runs
#   ("split", "temp", "tracer")  a split panel: one slice cut in two, the first variable
#                             on one side and the second on the other, as set by SPLIT
#                             (cut, at, mirror, cut_line, styles, labels); also
#                             ("split", run, first, second)
#   None                      an empty panel
# With colorbars="auto", a row (or column) holding one variable throughout shares
# one colorbar.
COMBINED = dict(
    name="combined",  # images <name>.<axis>.<NNNNN>.<format>, video <name>.<axis>.mp4
    layout=[
        ["dens", "temp"],
        ["tracer", "t_cool"],
        ["velx", "vely"],
    ],
    colorbars="vertical",  # "right": every variable's colorbar side by side right of the
                         # figure, each as tall as the panels (needs hspace); "bottom":
                         # all in one row under the figure (needs hspace); "vertical":
                         # each panel's own on its
                         # right (a split panel: a short one beside each half);
                         # "horizontal": below each panel (a split panel's above and
                         # below; hspace=None); "auto": shared by a row/column of one
                         # variable, else one per panel on its right
    cbar_length=0.5,  # vertical/horizontal colorbar length, fraction of the panel's side
    cbar_room=1.8,    # inches right of each panel for its vertical colorbar and label
                      # (hspace set; also sets the gap between columns); None = 3.0
    # xlim=(-500.0, 250.0),  # (min, max) of the x axis in length_units, None = that end of
    #                        # the domain; None = full domain (head + near wake; the
    #                        # 1.95 pc region is x in [-375, 125], |y| < 125)
    # ylim=(-250.0, 250.0),  # (min, max) of the y axis
    sharex=True,      # share the x axis: tick labels on the bottom row only
    hspace=0.15,      # gap between rows, fraction of a panel's height: panels stacked by
                      # hand, the figure as tall as their aspect needs (vertical or auto
                      # colorbars); None = matplotlib's constrained layout
    title=True,       # time as the figure title (False: give it in the caption)
    title_text=None,  # written instead of the computed time, on every frame; None = the time
    panel_letters="upper left",  # (a), (b), ... in this corner of each panel, in reading
                                 # order ("upper/lower left/right"); None = no letters
    # Style of each variable in this figure only (PLOT_VARS stays as it is for the
    # other plots): any of cmap, norm, vmin, vmax, label, under, vcenter, ticks.
    # Labels on two lines (name, units) to keep the colorbar column narrow.
    styles={
        "dens": dict(label="Density\n[cm$^{-3}$]"),
        "temp": dict(label="Temperature\n[K]"),
        "tracer": dict(label="Outflow\ntracer"),
        "t_cool": dict(label="Cooling time\n[Myr]"),
        "velx": dict(label="$v_x$\n[km/s]"),
        "vely": dict(label="$v_y$\n[km/s]"),
    },
    fps=10,           # video frames per second (51 frames to 1.96 Myr)
    video_max_size=4096,  # video frames scaled down to at most this many pixels per side
                          # (the encoders fail above 4096); None/0 = full size
)
# Comparing runs, one row per variable and one column per run:
# COMBINED = dict(
#     name="compare_sigv_30_75",  # images <name>.<axis>.<NNNNN>.<format>, video <name>.<axis>.mp4
#     layout=[[(run, var) for run in ["turb_sigv_30", "turb_sigv_75"]]
#                       for var in ["dens", "temp","velx","vely"]],
#     fps=24,           # video frames per second
#     video_max_size=4096,  # video frames scaled down to at most this many pixels per side
#                           # (the encoders fail above 4096); None/0 = full size
# )


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
    styles={
        "dens": dict(cmap="viridis", vmin=1.0e-3, vmax=1.0e3),
        "temp": dict(cmap="coolwarm", vmin=1.0e4, vcenter=1.0e6, vmax=1.0e8),
    },
    cut="horizontal",      # "horizontal": cut along the plot's x axis (top/bottom halves);
                           # "vertical": cut along the plot's y axis (left/right halves)
    at=0.0,                # cut position along the plot's y (horizontal) or x (vertical)
                           # axis, in length_units; None = middle of the shown range
    mirror=False,          # False: each variable on its own true half;
                           # True: both show the first variable's half, the second reflected
    xlim=(-500.0, 250.0),  # (min, max) of the plot's x axis in length_units; None = full domain
    ylim=(-250.0, 250.0),  # (min, max) of the plot's y axis; None = full domain
    cut_line=dict(color="k", linewidth=0.5),  # line along the cut; None = no line
    labels=None,           # colorbar labels [first, second]; None = the PLOT_VARS labels
                           # (short ones read better on a paper figure, e.g. r"$n$ [cm$^{-3}$]")
    cbar_length=1.0,       # colorbar length, fraction of the plot's width (height for a
                           # vertical cut); horizontal cut: first's above, second's below
    title=True,            # time as the title
    fig_size=(7.0, 4.0),   # inches; ~7 in = two-column paper width, ~3.5 in = one column
    dpi=300,               # for raster formats
    format="png",          # pdf/svg for the paper (vector), png for quick looks
)



# Cold gas mass in slabs along x (python main.py cold_mass): cold mass in slabs
# of bin_width spanning the whole y and z extent, vs x, one line per run (and
# frame); the legend gives each run's total.  Cached per run in <out>/cold_mass/.
COLD_MASS = dict(
    name="cold_mass_vs_x",  # figure <outdir>/<name>.<format>; outdir = cold_mass/ of the first run
    runs=["ucgm", "tcgm", "turb_tcgm"],  # runs to plot; None = RUN
    frames=None,          # frame numbers for every run; None = each run's frames in SIMULATIONS
    labels={"ucgm": "uCGM_vw2", "tcgm": "tCGM_vw2", "turb_tcgm": "turb_tcgm"},  # run -> legend label
    T_cold=2e4,           # K; cold gas is T < T_cold
    bin_width=1.0,        # slab thickness along x, in length_units; None = no binning,
                          # dM/dx per grid cell drawn as a continuous line
    length_units="kpc",   # x axis and bin_width: code, pc, kpc
    mass_units="Msun",    # g or Msun
    logy=True,            # logarithmic mass axis
    legend_fontsize=12,   # points, or "small", "medium", ...
    legend_loc="upper right",  # matplotlib legend loc, e.g. "best", "upper right"
    figsize=(6.8, 4.8),   # inches; 2x the (3.4, 2.4) single-column size, so the
                          # fonts are half size when scaled down to one column
)


# Gas masses vs time (python main.py timeseries).  Values are cached per run
# in <out>/timeseries/, so adding a run to the plot only reads that run.
# fast_wind has no Mdot(t) = Mdot0 e^-t/dt_mdot in its <problem> block, so leave
# "analytic"/"entrained" out for it (e.g. runs=["fast_wind"],
# quantities=["wind", "wind_tracer"], match=[]).
TIMESERIES = dict(
    name="compare_mass_vs_time",  # figure <outdir>/<name>.<format>; outdir = timeseries/ of the first run
    runs=["ucgm", "tcgm", "turb_tcgm"],  # runs to plot; None = RUN
    quantities=["cold", "analytic"],  # any of: cold (T < T_cold),
                          # wind (s_00 > s_min), cold_wind (both), wind_tracer (sum of rho s_00 dV),
                          # analytic (injected wind mass Mdot0 dt_mdot (1 - e^-t/dt_mdot) for
                          # Mdot(t) = Mdot0 e^-t/dt_mdot, both from the athinput's <problem>),
                          # entrained (entrained CGM = cold - analytic, using the matched cold mass when cold is in match)
    match=["cold"],       # quantities rescaled to equal `analytic` at time match_at (a constant
                          # factor per run, shown in the legend); [] = no rescaling
    match_at=1.0,         # in time_units
    overlay=True,         # True: one axes (colour = run, line style = quantity);
                          # False: a panel per run, stacked
    labels={},            # run -> legend/panel label, e.g. {"turb_sigv_30": r"$\sigma_v$ = 30 km/s"}
    T_cold=2e4,           # K; cold gas is T < T_cold
    s_min=1e-3,           # wind-polluted gas is s_00 > s_min (as the run's cooling mask)
    mass_units="Msun",    # g or Msun
    time_units="Myr",     # None = PLOT["time_units"]
    logy=True,            # logarithmic mass axis
    figsize=(8.0, 4.5),   # inches; height is per panel when overlay is False
)


# Line emission from a table of eps/n_H^2 (n_H, T) made outside vis_athenak,
# e.g. by Cloudy (see emission/table.py for the format): the line_<name>
# quantities (slices of e.g. line_Ha) and projected sky maps
# (python main.py emission, see emission/projection.py).
EMISSION = dict(
    table="emis_KS19_z0.96_Z0.2.h5",  # relative to $EMISSION_TABLE_DIR (absolute = as is)
    sums={                # line name -> table lines added up for it
        "SII": ["SII_6716", "SII_6731"],
        "SIII": ["SIII_9069", "SIII_9531"],
        "OIII": ["OIII_5007", "OIII_4959"],
        "OIII_Ha": ["OIII_5007", "Ha"],  # JWST paper Fig. 3: [OIII]5007 + Halpha
    },
    # Projected maps, set up as JWST/NIRSpec Fig. 3 of van Dokkum et al. (2026)
    run="turb_sigv_30",   # None = RUN
    name="emission_i29_jwst_zoom",      # files <out>/emission/<name>.<NNNNN>.<format> (+ .npz cache)
    lines=["OIII_Ha", "OIII_5007", "Ha"],  # one panel each (table lines or sums)
    ratios=[("OIII_5007", "Ha")],  # one ratio panel each
    split_wind=False,     # also panels of the wind (s_00) and CGM (1 - s_00) parts of each line
    inclination=29.0,     # degrees; motion axis (x1) tilted toward the observer, about x2
                          # (line of sight (sin i, 0, cos i) in (x1, x2, x3); 0 = along x3)
    z=0.96,               # redshift for distance and angles (Planck18); None = no observer:
                          # maps of luminosity per area [erg/s/kpc^2], pixel and psf in kpc
    pixel=0.03,           # pixel size, arcsec (kpc if z is None); NIRSpec cube grid
    psf_fwhm=0.1,         # Gaussian PSF FWHM, arcsec (kpc if z is None); 0 = none;
                          # NIRSpec IFU ~0.06-0.15 (diffraction 0.03-0.04 at [OIII]/Halpha)
    subsample=1,          # each cell split n x n across the line of sight before binning
                          # (avoids moire when pixels are only a few cells); 1 = cell centres
    r_range=(-8.0, 2.0),  # zoom: projected distance from the black hole shown, kpc; None = all
                          # (JWST paper r = this + ~63 kpc; Fig. 3 spans r = 55..66 kpc)
    b_range=(-2.0, 2.0),  # zoom: b shown, kpc; None = all
    sb_range=(None, None),  # colour range of the maps (erg/s/cm^2/arcsec^2, or erg/s/kpc^2);
                          # None = top: brightest pixel shown, bottom: sb_decades below the top
    sb_decades=3.0,       # decades shown when the lower end of sb_range is None
    ratio_range=(1e-3, 10.0),  # colour range of the ratio maps
    ratio_min_sb=1e-21,   # ratio shown only where the denominator line is brighter than this
    cmap="inferno",
    ratio_cmap="RdBu_r",
    panel_size=(9.0, 3.2),  # inches per panel; panels are stacked vertically
)


def emission_table_path(table: str | None = None):
    """Absolute path of an emission table (default EMISSION["table"])."""
    return env_path("EMISSION_TABLE_DIR") / (table or EMISSION["table"])


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
