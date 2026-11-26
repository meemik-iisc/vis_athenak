"""
Projected emission-line maps of a run, as seen by an observer at redshift z.

    python main.py emission                          # config.EMISSION
    python main.py emission --frames 60 --lines Ha OIII_5007 --ratios OIII_5007/Ha
    python main.py emission --inclination 0 --z none --pixel 0.031 --psf-fwhm 0
    python main.py emission --split-wind -w 2

Each cell's line luminosity eps dV (eps from the emission table, see
emission/table.py, at n_H = X_H rho / m_H and the run's T) is put on the sky:
the line of sight is (sin i, 0, cos i) in (x1, x2, x3), so the motion axis x1
is tilted by the inclination i toward the observer and the sky coordinates are

    r = x1 cos i - x3 sin i     (projected distance along the motion; BH at 0)
    b = x2

(i = 0 is a plain projection along x3).  Luminosities are summed in square
pixels with edges at whole multiples of the pixel size.  With --subsample n each cell is split into
n x n x n parts first, which avoids moire patterns when a pixel is only a
few cells wide.  The sums are cached per frame in
``<out>/emission/cache/<table>_i<i>_p<pixel kpc>_n<n>.<NNNNN>.npz`` (erg/s per
pixel, before the PSF), so new colour ranges, PSFs or ratios need no new
read; --recompute ignores the cache.

With a redshift (Planck18 cosmology) the maps are surface brightness
[erg s^-1 cm^-2 arcsec^-2]: L / (4 pi D_L^2) per pixel solid angle, so the
(1+z)^4 dimming is included; pixel and PSF FWHM are in arcsec.  With
``--z none`` the maps are luminosity per projected area [erg s^-1 kpc^-2] and
pixel and PSF are in kpc.  The PSF is a Gaussian.  Ratio maps are the ratio
of the two (PSF-smoothed) maps where the denominator is above
--ratio-min-sb.  --split-wind adds the wind (s_00) and CGM (1 - s_00) parts
of each line (s_00 clipped to [0, 1]).  --r-range / --b-range zoom in (kpc from
the black hole; the PSF is applied to the whole map first, and with
sb_range (None, None) the colours span the zoomed region); with a redshift the
top and right axes give arcsec from the black hole.

Images go to ``<out>/emission/<name>.<NNNNN>.<format>``, one panel per line
and ratio, stacked; ``python main.py video --indir <out>/emission --name
<name>`` makes a video of them.  Needs Cartesian runs; every frame is read
whole (several GB for large runs), so use few workers.
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

if "." in __package__:  # imported as vis_athenak.emission
    from .. import config
    from ..plotting.quantities import available, get, hydrogen_density
    from ..plotting.slice2d import apply_fonts, n_workers, parse_frames, time_title
    from ..simulation_data import SimulationData
    from ..utils.units import PC_CGS, Units
else:  # run from the repo root
    import config
    from plotting.quantities import available, get, hydrogen_density
    from plotting.slice2d import apply_fonts, n_workers, parse_frames, time_title
    from simulation_data import SimulationData
    from utils.units import PC_CGS, Units

from . import table as emission_table  # noqa: E402

KPC_CGS = 1e3 * PC_CGS
C_KMS = 299792.458
PLANCK18 = dict(H0=67.66, Om=0.30966)  # flat; radiation left out
ARCSEC = math.pi / (180.0 * 3600.0)    # rad
FWHM_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

# Panel titles of the lines; any other line is shown by its name.
LINE_LABELS = {
    "Ha": r"H$\alpha$", "Hb": r"H$\beta$",
    "OIII_5007": "[OIII]5007", "OIII_4959": "[OIII]4959", "OIII": "[OIII]4959+5007",
    "NII_6584": "[NII]6584", "SII_6716": "[SII]6716", "SII_6731": "[SII]6731",
    "SII": "[SII]6716+6731", "SIII_9069": "[SIII]9069", "SIII_9531": "[SIII]9531",
    "SIII": "[SIII]9069+9531",
    "OIII_Ha": r"[OIII]5007 + H$\alpha$",
}


def line_label(line: str) -> str:
    return LINE_LABELS.get(line, line)


# ── Cosmology ────────────────────────────────────────────────────────────

def comoving_distance_kpc(z: float, H0: float = PLANCK18["H0"],
                          Om: float = PLANCK18["Om"]) -> float:
    """Line-of-sight comoving distance of a flat LambdaCDM universe, kpc."""
    integral, _ = quad(lambda x: 1.0 / math.sqrt(Om * (1.0 + x)**3 + 1.0 - Om), 0.0, z)
    return C_KMS / H0 * integral * 1e3


def luminosity_distance_cm(z: float) -> float:
    return (1.0 + z) * comoving_distance_kpc(z) * KPC_CGS


def kpc_per_arcsec(z: float) -> float:
    """Proper length per arcsec at redshift z (angular diameter distance)."""
    return comoving_distance_kpc(z) / (1.0 + z) * ARCSEC


# ── Projection ───────────────────────────────────────────────────────────

def _edges(lo: float, hi: float, step: float) -> np.ndarray:
    """Pixel edges of width ``step`` covering [lo, hi], at whole multiples of
    ``step`` (so pixels the size of the cells line up with them at i = 0)."""
    k0 = math.floor(lo / step + 1e-9)
    k1 = math.ceil(hi / step - 1e-9)
    return np.arange(k0, k1 + 1) * step


def sky_edges(frame, params: dict, inclination: float, pixel_kpc: float):
    """(r_edges, b_edges) in kpc of the pixels covering the projected box."""
    length = Units.from_params(params).length_cgs / KPC_CGS
    x1f, x2f, x3f = (np.asarray(frame.grid[k], dtype=float) * length
                     for k in ("x1f", "x2f", "x3f"))
    c, s = math.cos(math.radians(inclination)), math.sin(math.radians(inclination))
    r = [x1 * c - x3 * s for x1 in x1f[[0, -1]] for x3 in x3f[[0, -1]]]
    return _edges(min(r), max(r), pixel_kpc), _edges(x2f[0], x2f[-1], pixel_kpc)


def project(frame, params: dict, lines: list[str], table, inclination: float,
            pixel_kpc: float, subsample: int = 1, split_wind: bool = False) -> dict:
    """
    Luminosity [erg/s] per sky pixel of each line of ``lines`` for one frame:
    {"r_edges", "b_edges" (kpc), <line>: (n_r, n_b) array, and if split_wind
    "wind/<line>": the part from wind material (s_00)}.  See the module doc.
    """
    if frame.coordinates != "cartesian":
        raise NotImplementedError(f"projections need Cartesian runs, not {frame.coordinates!r}")
    u = Units.from_params(params)
    if split_wind and "s_00" not in available(frame):
        print(f"  frame {frame.number}: no s_00 tracer, no wind/CGM split")
        split_wind = False

    # Luminosity of each cell, erg/s
    nH = hydrogen_density(frame, params, table.X_H)
    T = np.asarray(get(frame, "temp", params, units="K"), dtype=float)
    eps = emission_table.line_emissivities(table, lines, nH, T)
    del nH, T
    v1, v2, v3 = frame.cell_volume_factors
    dV = (v3[:, None, None] * v2[None, :, None] * v1[None, None, :]) * u.length_cgs**3
    lum = {line: eps.pop(line) * dV for line in lines}
    del dV
    if split_wind:
        s = np.clip(np.asarray(get(frame, "s_00", params), dtype=float), 0.0, 1.0)
        lum.update({f"wind/{line}": lum[line] * s for line in lines})
        del s

    length = u.length_cgs / KPC_CGS
    x1v, x2v, x3v = (np.asarray(frame.grid[k], dtype=float) * length
                     for k in ("x1v", "x2v", "x3v"))
    dx1, dx2, dx3 = (np.diff(np.asarray(frame.grid[k], dtype=float)) * length
                     for k in ("x1f", "x2f", "x3f"))
    r_edges, b_edges = sky_edges(frame, params, inclination, pixel_kpc)
    n_r, n_b = r_edges.size - 1, b_edges.size - 1
    c, s = math.cos(math.radians(inclination)), math.sin(math.radians(inclination))

    maps = {name: np.zeros(n_r * n_b) for name in lum}
    offsets = (np.arange(subsample) + 0.5) / subsample - 0.5  # in cell widths
    weight = 1.0 / subsample**3
    for o1 in offsets:
        for o3 in offsets:
            r = (x1v + o1 * dx1)[None, :] * c - (x3v + o3 * dx3)[:, None] * s  # (k, i)
            ir = np.clip(((r - r_edges[0]) / pixel_kpc).astype(np.int64), 0, n_r - 1)
            for o2 in offsets:
                ib = np.clip(((x2v + o2 * dx2 - b_edges[0]) / pixel_kpc).astype(np.int64),
                             0, n_b - 1)
                index = (ir[:, None, :] * n_b + ib[None, :, None]).ravel()  # (k, j, i)
                for name, L in lum.items():
                    maps[name] += weight * np.bincount(index, weights=L.ravel(),
                                                       minlength=n_r * n_b)
    out = {name: m.reshape(n_r, n_b) for name, m in maps.items()}
    out.update(r_edges=r_edges, b_edges=b_edges)
    return out


# ── Cache ────────────────────────────────────────────────────────────────

def cache_path(outdir: Path, table_name: str, inclination: float, pixel_kpc: float,
               subsample: int, number: int) -> Path:
    tag = f"{Path(table_name).stem}_i{inclination:g}_p{pixel_kpc:.4g}_n{subsample}"
    return Path(outdir) / "emission" / "cache" / f"{tag}.{number:05d}.npz"


def read_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def write_cache(path: Path, maps: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **maps)
    tmp.replace(path)


def frame_maps(frame, params: dict, opts: dict, recompute: bool = False) -> dict:
    """`project` of the lines and ratio lines in opts, from the cache where possible."""
    table = emission_table.default_table()
    path = cache_path(opts["outdir"], table.path.name, opts["inclination"],
                      opts["pixel_kpc"], opts["subsample"], frame.number)
    needed = list(dict.fromkeys([*opts["lines"], *(x for pair in opts["ratios"] for x in pair)]))
    split = opts["split_wind"] and "s_00" in available(frame)
    maps = {} if recompute else read_cache(path)
    missing = [n for n in needed if n not in maps or (split and f"wind/{n}" not in maps)]
    if missing:
        maps.update(project(frame, params, missing, table, opts["inclination"],
                            opts["pixel_kpc"], opts["subsample"], split))
        maps["time"] = np.array(frame.time)
        write_cache(path, maps)
    return maps


def observed(lum: np.ndarray, opts: dict) -> np.ndarray:
    """Luminosity per pixel -> PSF-smoothed surface brightness (see module doc)."""
    if opts["psf_fwhm"] > 0:
        sigma = opts["psf_fwhm"] / FWHM_SIGMA / opts["pixel"]  # in pixels
        lum = gaussian_filter(lum, sigma, mode="constant")
    if opts["z"] is None:
        return lum / opts["pixel_kpc"]**2                       # erg/s/kpc^2
    return lum / (4.0 * math.pi * opts["D_L"]**2) / opts["pixel"]**2  # erg/s/cm^2/arcsec^2


# ── Plot ─────────────────────────────────────────────────────────────────

def _window(edges: np.ndarray, limits) -> tuple[tuple[int, int], np.ndarray]:
    """(index range, edges) of the pixels overlapping ``limits`` (lo, hi), either
    of which may be None; all pixels if ``limits`` is None."""
    lo, hi = limits if limits is not None else (None, None)
    i0 = 0 if lo is None else max(int(np.searchsorted(edges, lo, side="right")) - 1, 0)
    i1 = edges.size - 1 if hi is None else min(int(np.searchsorted(edges, hi, side="left")),
                                               edges.size - 1)
    if i1 <= i0:
        raise ValueError(f"range {limits} is outside the map ({edges[0]:.3g}..{edges[-1]:.3g} kpc)")
    return (i0, i1), edges[i0:i1 + 1]


def plot_maps(maps: dict, frame, params: dict, opts: dict, path: Path) -> Path:
    """One panel per line (and wind/CGM parts) and ratio, stacked vertically."""
    apply_fonts()
    split = opts["split_wind"] and all(f"wind/{n}" in maps for n in opts["lines"])
    sb = {n: observed(maps[n], opts) for n in
          dict.fromkeys([*opts["lines"], *(x for pair in opts["ratios"] for x in pair)])}
    panels = []  # (title, image, kind)
    for n in opts["lines"]:
        panels.append((line_label(n), sb[n], "sb"))
        if split:
            wind = observed(maps[f"wind/{n}"], opts)
            panels.append((f"{line_label(n)}, wind", wind, "sb"))
            panels.append((f"{line_label(n)}, CGM", sb[n] - wind, "sb"))
    for num, den in opts["ratios"]:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sb[den] > opts["ratio_min_sb"], sb[num] / sb[den], np.nan)
        panels.append((f"{line_label(num)} / {line_label(den)}", ratio, "ratio"))

    # Zoom: crop after the PSF (so the edges are right) and before the colour scaling
    (i0, i1), r = _window(maps["r_edges"], opts.get("r_range"))
    (j0, j1), b = _window(maps["b_edges"], opts.get("b_range"))
    panels = [(title, image[i0:i1, j0:j1], kind) for title, image, kind in panels]

    w, h = opts["panel_size"]
    fig, axes = plt.subplots(len(panels), 1, figsize=(w, h * len(panels)), squeeze=False,
                             sharex=True, layout="constrained")
    sb_units = (r"erg s$^{-1}$ kpc$^{-2}$" if opts["z"] is None
                else r"erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$")
    for n, (ax, (title, image, kind)) in enumerate(zip(axes[:, 0], panels)):
        if kind == "sb":
            image = np.where(image > 0, image, np.nan)
            lo, hi = opts["sb_range"]
            if hi is None:
                hi = np.nanmax(image) if np.isfinite(image).any() else 1.0
            if lo is None:  # sb_decades below the top
                lo = hi * 10.0 ** -opts["sb_decades"]
            norm, cmap, label = LogNorm(lo, hi, clip=True), opts["cmap"], None
        else:
            norm, cmap, label = LogNorm(*opts["ratio_range"], clip=True), opts["ratio_cmap"], "ratio"
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad(cmap(0.0) if kind == "sb" else "0.85")
        im = ax.imshow(image.T, origin="lower", extent=(r[0], r[-1], b[0], b[-1]),
                       norm=norm, cmap=cmap, interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize="medium")
        ax.set_ylabel("b [kpc]")
        if opts["z"] is not None:  # angles from the black hole, as in the JWST figures
            scale = opts["pixel_kpc"] / opts["pixel"]  # kpc per arcsec
            to_arcsec = (lambda x: x / scale, lambda x: x * scale)
            if n == 0:
                ax.secondary_xaxis("top", functions=to_arcsec).set_xlabel("r [arcsec]")
            ax.secondary_yaxis("right", functions=to_arcsec).set_ylabel("b [arcsec]")
        fig.colorbar(im, ax=ax, label=label, pad=0.01, fraction=0.03)
    axes[-1, 0].set_xlabel("projected distance along the motion [kpc]")
    if opts["z"] is None:
        sky = f"pixel {opts['pixel']:g} kpc, PSF {opts['psf_fwhm']:g} kpc"
    else:
        sky = (f"z = {opts['z']:g}, pixel {opts['pixel']:g}\" ({opts['pixel_kpc']:.3g} kpc), "
               f"PSF {opts['psf_fwhm']:g}\"")
    quantity = "luminosity per area" if opts["z"] is None else "surface brightness"
    fig.suptitle(f"{time_title(frame, params, opts)},  i = {opts['inclination']:g}°\n{sky}\n"
                 f"maps: {quantity} [{sb_units}]")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=opts["dpi"])
    plt.close(fig)
    return path


def _frame_job(job) -> Path:
    """Worker: maps (cached or computed) and figure of one frame."""
    frame, params, opts, table_path, sums, recompute = job
    emission_table.configure(table_path, sums)
    try:
        maps = frame_maps(frame, params, opts, recompute)
        path = Path(opts["outdir"]) / "emission" / f"{opts['name']}.{frame.number:05d}.{opts['format']}"
        return plot_maps(maps, frame, params, opts, path)
    finally:
        frame.unload()


# ── Command line ─────────────────────────────────────────────────────────

def _ratio(text: str) -> tuple[str, str]:
    num, sep, den = text.partition("/")
    if not sep or not num or not den:
        raise argparse.ArgumentTypeError(f"ratio must be LINE/LINE, got {text!r}")
    return num, den


def _redshift(text: str):
    return None if text.lower() in ("none", "0") else float(text)


def main(argv=None):
    missing = [n for n in ("EMISSION", "emission_table_path") if not hasattr(config, n)]
    if missing:
        raise SystemExit(f"main.py emission: config.py has no {', '.join(missing)} "
                         "(see config2.py for the emission settings)")
    em, plot = config.EMISSION, config.PLOT
    parser = argparse.ArgumentParser(
        prog="main.py emission",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run_default = em.get("run") or config.RUN
    parser.add_argument("--run", default=run_default, choices=list(config.SIMULATIONS),
                        help=f"simulation (default: {run_default})")
    parser.add_argument("--frames", nargs="+",
                        help="frame numbers and/or ranges like 0:61:5 (default: the run's)")
    parser.add_argument("--table", default=em["table"],
                        help=f"emission table, relative to $EMISSION_TABLE_DIR (default: {em['table']})")
    parser.add_argument("--lines", nargs="+", default=em["lines"],
                        help=f"lines to map: table lines or sums in EMISSION['sums'] (default: {em['lines']})")
    parser.add_argument("--ratios", nargs="*", type=_ratio, default=[tuple(r) for r in em["ratios"]],
                        help="ratio maps as LINE/LINE; no value = none (default: "
                             + " ".join(f"{a}/{b}" for a, b in em["ratios"]) + ")")
    parser.add_argument("--split-wind", action=argparse.BooleanOptionalAction,
                        default=em["split_wind"],
                        help=f"also the wind and CGM parts of each line (default: {em['split_wind']})")
    parser.add_argument("--inclination", type=float, default=em["inclination"],
                        help=f"degrees (default: {em['inclination']:g})")
    parser.add_argument("--z", type=_redshift, default=em["z"],
                        help=f"redshift, or none for maps per kpc^2 (default: {em['z']})")
    parser.add_argument("--pixel", type=float, default=em["pixel"],
                        help=f"pixel size, arcsec (kpc with --z none) (default: {em['pixel']:g})")
    parser.add_argument("--psf-fwhm", type=float, default=em["psf_fwhm"],
                        help=f"PSF FWHM, arcsec (kpc with --z none); 0 = none (default: {em['psf_fwhm']:g})")
    parser.add_argument("--subsample", type=int, default=em["subsample"],
                        help=f"split each cell n x n x n before binning (default: {em['subsample']})")
    parser.add_argument("--sb-range", type=float, nargs=2, default=em["sb_range"],
                        help=f"colour range of the maps (default: {em['sb_range']})")
    parser.add_argument("--sb-decades", type=float, default=em.get("sb_decades", 3.0),
                        help="with no lower colour limit in sb_range, show this many decades "
                             f"below the top (default: {em.get('sb_decades', 3.0):g})")
    parser.add_argument("--ratio-range", type=float, nargs=2, default=em["ratio_range"],
                        help=f"colour range of the ratios (default: {em['ratio_range']})")
    parser.add_argument("--ratio-min-sb", type=float, default=em["ratio_min_sb"],
                        help=f"ratio shown where the denominator is above this (default: {em['ratio_min_sb']:g})")
    parser.add_argument("--r-range", type=float, nargs=2, default=em.get("r_range"),
                        help="zoom: r range shown, kpc from the black hole (default: "
                             f"{em.get('r_range')}, None = all)")
    parser.add_argument("--b-range", type=float, nargs=2, default=em.get("b_range"),
                        help=f"zoom: b range shown, kpc (default: {em.get('b_range')}, None = all)")
    parser.add_argument("--name", default=em["name"], help=f"file name stem (default: {em['name']})")
    parser.add_argument("--outdir", help="folder holding emission/ (default: the run's out)")
    parser.add_argument("--dpi", type=int, default=plot["dpi"])
    parser.add_argument("--format", default=plot["format"], help="image format")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached maps")
    parser.add_argument("-w", "--workers", "-c", "--cores", dest="workers", type=int,
                        default=n_workers(),
                        help="parallel worker processes, 1 = serial; each holds a whole "
                             f"3D frame (default: config.N_WORKERS = {n_workers()})")
    args = parser.parse_args(argv)
    if args.subsample < 1:
        parser.error("--subsample must be >= 1")

    try:
        settings = config.resolve(args.run)
        table_path = config.emission_table_path(args.table)
        emission_table.configure(table_path, em.get("sums", {}))
        table = emission_table.default_table()
    except (RuntimeError, OSError, ValueError) as err:
        parser.error(str(err))
    wanted_lines = list(dict.fromkeys([*args.lines, *(x for p in args.ratios for x in p)]))
    for line in wanted_lines:
        unknown = [c for c in emission_table.components(line) if c not in table.lines]
        if unknown:
            parser.error(f"line {line!r}: {unknown} not in {table.path.name} "
                         f"(lines {table.lines}, sums {list(emission_table.sums())})")

    sim = SimulationData(settings["athinput"], settings["data"], outputs=settings["outputs"],
                         basename=settings.get("basename"))
    wanted = parse_frames(args.frames)
    if wanted is None and settings["frames"] is not None:
        wanted = set(settings["frames"])
    frames = [f for f in sim.frames if wanted is None or f.number in wanted]
    if not frames:
        parser.error(f"no frames selected; run has frames {sim.frame_numbers}")

    opts = {k: getattr(args, k) for k in
            ("lines", "ratios", "split_wind", "inclination", "z", "pixel", "psf_fwhm",
             "subsample", "sb_range", "sb_decades", "ratio_range", "ratio_min_sb", "r_range", "b_range",
             "name", "dpi", "format")}
    opts.update(outdir=Path(args.outdir) if args.outdir else settings["out"],
                cmap=em["cmap"], ratio_cmap=em["ratio_cmap"], panel_size=em["panel_size"],
                time_units=plot["time_units"], time_decimals=plot["time_decimals"])
    if args.z is None:
        opts["pixel_kpc"] = args.pixel
    else:
        scale = kpc_per_arcsec(args.z)
        opts.update(pixel_kpc=args.pixel * scale, D_L=luminosity_distance_cm(args.z))
        print(f"z = {args.z:g}: D_L = {opts['D_L'] / KPC_CGS / 1e3:.0f} Mpc, "
              f"{scale:.4g} kpc/arcsec, pixel {opts['pixel_kpc']:.4g} kpc")
    print(f"{args.run}: {len(frames)} frames, table {table}")

    jobs = [(f, sim.params, opts, table_path, emission_table.sums(), args.recompute)
            for f in frames]
    workers = min(args.workers, len(jobs))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for path in pool.map(_frame_job, jobs):
                print(f"  saved {path}", flush=True)
    else:
        for job in jobs:
            print(f"  saved {_frame_job(job)}", flush=True)


if __name__ == "__main__":
    main()
