# Adding Cloudy line-emission maps to vis_athenak

Notes from a Claude Code session (2026-11-18), started in `/data/data_meemik/RunawayBH/sim_data`.
Goal: make [OIII] and Hα emission maps (plus other optical lines) from AthenaK snapshots and
compare them to the runaway SMBH RBH-1 (van Dokkum et al. 2026, ApJL 998, L27: "JWST
Confirmation of a Runaway Supermassive Black Hole via Its Supersonic Bow Shock").

Working preference (updated 2026-11-18): Claude writes the code itself, one agreed step at a
time; the user runs the long jobs (e.g. the Cloudy grid) themselves.

---

## 0. Progress (2026-11-18)

- **Generator done**: `~/Meemik/cloudy_tables`, public repo
  https://github.com/meemik-iisc/cloudy_tables (branch `main`). Tested end to end on a 2×3 grid;
  the full 972-point grid is to be run by the user (`python run_grid.py` then `python collect.py`
  in `emtab/`, ~10 min with 8 workers).
- **vis_athenak side not started** (branch `feature/add_cloudy`). Next: `emission/table.py`.

---

## 1. Decided architecture

Two separate projects joined by a fixed table format:

1. **Generator project (new, separate repo, e.g. `emission_tables/`)**: runs Cloudy (or any other
   code) over a (n_H, T) grid and writes an emission table file in the format below.
2. **vis_athenak `emission/` subpackage**: only *reads* a table given by the user (it doesn't
   care how it was made), then makes emission quantities, sky maps, position–velocity (PV)
   diagrams and line-ratio profiles from simulation frames.

The table format is the shared interface. vis_athenak's `EmissionTable` reader is the reference
definition of the format, and the generator's `validate()` should open each new table with it.

---

## 2. Emission table format (version 1, implemented in the generator's `emtab/table_io.py`)

One HDF5 file per table, e.g. `emis_KS19_z0.96_Z0.2.h5`:

```
├── attrs:  format="emission_table", format_version=1,
│           X_H, log_floor=-50, source="Cloudy 25.00", radiation="KS19 z=0.96 + CMB + CR background",
│           metallicity_Zsun=0.2, abundances="Cloudy default solar, metals scaled by 0.2", created=...
├── axes/
│   ├── log_nH      (1D, cm^-3, increasing; spacing need not be uniform)
│   └── log_T       (1D, K, increasing)
│   attrs: axis_order = ["log_nH", "log_T"]
└── lines/
    ├── Ha          (n_nH × n_T)  log10(ε / n_H²)   [erg cm³ s⁻¹]
    ├── Hb, OIII_5007, OIII_4959, NII_6584, SII_6716, SII_6731, SIII_9069, SIII_9531
    attrs per line: wavelength_A, label (Cloudy label, e.g. "O  3 5006.84A"), units
```

Design choices:
- **Store ε/n_H², not ε.** It behaves like a cooling function Λ(T): smooth, and clamping it
  outside the density range gives ε ∝ n² automatically.
- **Store `X_H` (hydrogen mass fraction).** vis_athenak needs it to convert simulated density to
  n_H, consistently with the table's abundances. The generator works it out from its abundance
  set (Cloudy `save abundances`).
- **Individual lines only.** Doublet sums ([SII] 6716+6731, [SIII] 9069+9532) are formed in
  vis_athenak via config.
- **Two axes in v1.** Extra axes (Z, radiation strength) can be added in a later version.
- **`log_floor`** (added): values below it, including ε = 0 (e.g. [SII] in hot diffuse gas), are
  stored as `log_floor` = −50. The vis_athenak reader should treat cells at the floor as ε = 0.
- `X_H` for 0.2 Z☉ with Cloudy's default abundances (He/H = 0.1) is 0.714.

### Reader (vis_athenak `emission/table.py`)

```python
import h5py, numpy as np
from scipy.ndimage import map_coordinates

class EmissionTable:
    FORMAT, VERSION = "emission_table", 1

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            if f.attrs.get("format") != self.FORMAT:
                raise ValueError(f"{path}: not an emission_table file")
            if int(f.attrs["format_version"]) != self.VERSION:
                raise ValueError(f"{path}: format_version {f.attrs['format_version']} unsupported")
            if [str(a) for a in f["axes"].attrs["axis_order"]] != ["log_nH", "log_T"]:
                raise ValueError(f"{path}: v1 needs axes (log_nH, log_T)")
            self.log_nH = f["axes/log_nH"][:]
            self.log_T  = f["axes/log_T"][:]
            self.data   = {k: f["lines"][k][:] for k in f["lines"]}
            self.wavelength = {k: float(f["lines"][k].attrs["wavelength_A"]) for k in f["lines"]}
            self.meta   = {k: f.attrs[k] for k in f.attrs}
        self.X_H = float(self.meta["X_H"])

    @property
    def lines(self):
        return list(self.data)

    def emissivity(self, line, nH, T):
        """Volume emissivity [erg s^-1 cm^-3]; nH [cm^-3], T [K]; any array shape."""
        ln, lt = np.log10(nH), np.log10(T)
        # fractional indices (valid for non-uniform axes); np.interp clamps at the edges
        fi = np.interp(ln, self.log_nH, np.arange(self.log_nH.size))
        fj = np.interp(lt, self.log_T,  np.arange(self.log_T.size))
        q = map_coordinates(self.data[line], [fi.ravel(), fj.ravel()], order=1, mode="nearest")
        return 10**q.reshape(np.shape(nH)) * nH**2
```

---

## 3. Planned layouts

### vis_athenak additions

```
vis_athenak/
├── emission/
│   ├── __init__.py
│   ├── table.py           # EmissionTable (above): load + validate + interpolate
│   └── projection.py      # sky maps, PV diagrams, line-ratio profiles; `python main.py emission`
├── plotting/quantities.py # + DERIVED "line_Ha", "line_OIII", ... (NOT "emissivity": that name
│                          #   is already the n^2 Λ(T) cooling emissivity)
├── config.py              # + EMISSION dict: table file, z, inclination, pixel, PSF, sigma_v, line sums
└── .env.example           # + EMISSION_TABLE_DIR
```

Notes:
- Use `SimulationData` / `Frame` (not a standalone reader) and `utils/units.py` for code→cgs
  (no hard-coded T_unit). Read `frame.py` and `units.py` before writing the quantity functions.
- `line_*` derived quantities must be elementwise numpy so they also work on `Plane` slices
  (CLAUDE.md convention). Then `python main.py slices` can draw line-emissivity slices for free.
- Projections need full 3D fields (`Frame` loads them): ~3–4 GB per snapshot for the run below.
  Use few workers (~4) with `ProcessPoolExecutor`.
- Tables are data: keep them outside the repo (path via `.env`), gitignored.

### Generator project (separate repo, built: `~/Meemik/cloudy_tables`)

```
cloudy_tables/
├── config.py            # grid, radiation field, metallicity, lines, N_WORKERS (8), HEATMAPS;
│                        #   TABLE_NAME is built from the settings (emis_KS19_z0.96_Z0.2)
├── .env                 # CLOUDY_EXE, OUTPUT_DIR (gitignored; template .env.example)
├── emtab/               # each module runs as `python emtab/x.py` or `python -m emtab.x`
│   ├── cloudy_input.py  # config + (nH, T) -> Cloudy input text
│   ├── run_grid.py      # write inputs, run in parallel (-w), resume, rerun points whose input
│   │                    #   changed, list failures in failed.txt; --point I J, --force
│   ├── collect.py       # parse .ems/.abn -> log ε/n_H², X_H; write + validate .h5; draw heatmaps
│   ├── table_io.py      # write_table(), validate(), summary() for format v1
│   └── heatmaps.py      # PNGs next to the table: log ε/n_H² (Hα, [OIII]) with log ε contours,
│                        #   and log [OIII]/Hα with contours at the observed 0.6 and 1.5
└── tables/              # <TABLE_NAME>.h5 and PNGs are committed; run folders tables/*/ are not
```

Reuse: change `METALLICITY_ZSUN` (or the radiation field) in `config.py` and rerun; the new
settings give a new `TABLE_NAME`, so earlier tables are kept.

Sizes per table: run folder ~230 MB (almost all `.out` logs, gitignored), `.h5` ~80 KB, PNGs
~50–200 KB each.

---

## 4. Verified facts about the environment

- Cloudy 25.00 is at `~/opt/cloudy/c25.00`, executable `~/bin/cloudy` → `.../source/cloudy.exe`.
  Run with `cloudy -r name` (reads `name.in`). Earlier test files are in `~/opt/cloudy/test/`.
- Line labels in c25 must have only the species quoted, wavelength after: `"H  1" 6562.81A`.
  The fully quoted form `"H  1 6562.81A"` fails with "failed to read wavelength". Lines used:
  H 1 6562.81, 4861.32; O 3 5006.84, 4958.91; N 2 6583.45; S 2 6716.44, 6730.82;
  S 3 9068.62, 9530.62 (all Å).
- A single pass gives cautions ("H Lyman continuum is thin, and I assumed that it was thick");
  `iterate` removes them and changes the lines by < 0.02 %, at no extra run time (~5–7 s/point).
- `save abundances "x.abn" last` gives log10 **number densities** n_X [cm⁻³] (header `#abund H
  HELI ...`, 30 elements), not values relative to H.
- `save line emissivity "x.ems" last` + label list + `end of lines` gives **linear** volume
  emissivity [erg cm⁻³ s⁻¹] with a `#depth` column first.
- Test at log n_H = 0, T = 10⁴ K, KS19 z=0.96, metals 0.2: Hα = 2.42e-25, [OIII]5007 = 1.46e-28
  (i.e. [OIII]/Hα ≈ 6e-4). Use this as a regression check for the generator.
- Planck18 (H0=67.66, Ωm=0.30966) at z=0.96: D_L = 6464 Mpc, 8.158 kpc/arcsec.
- `astropy` is not installed in the base env (scipy is), so compute cosmology with scipy `quad`.

### Cloudy input used for one grid point

```
title lognH=<n> logT=<t>
table KS19 redshift 0.96
cmb redshift 0.96
cosmic rays background
hden <log nH>
constant temperature <log T>
metals -0.699 log          # 0.2 Zsun
set dr 0                   # 1 cm thick zone -> optically thin
stop zone 1
iterate                    # 2nd pass uses the thin optical depths
save line emissivity "<name>.ems" last
"H  1" 6562.81A
<other line labels...>
end of lines
save abundances "<name>.abn" last
```

Grid: log n_H = −4.5…2 (step 0.25), log T = 4…7.5 (step 0.1) → 27 × 36 = 972 runs, ~10 min
with 8 workers (the machine is an i9-14900K; 8 leaves room for other work).

---

## 5. Target simulation: `turb_tcgm/athenak_3d_runs/turb_tcgm_single`

Full path: `/data/data_meemik/RunawayBH/sim_data/turb_tcgm/athenak_3d_runs/turb_tcgm_single/`
(`bin/turb_tcgm.hydro_w.000NN.bin`, NN = 00–60, ~805 MB each).

- 3D Cartesian, black hole rest frame, BH at origin, moving toward +x1 (background CGM flows in from +x1).
  x1 ∈ [−60, 4] kpc (2048), x2, x3 ∈ [−2, 2] kpc (128 each); cell = 31 pc; 8 meshblocks of 256×128×128.
- Variables: `dens velx vely velz eint s_00` (s_00 = wind mass fraction tracer).
- Units: 1 kpc, 1000 km/s, density 1 m_p/cc; time unit 0.978 Myr; μ = 0.6;
  T = (γ−1)·eint/dens × 7.216e7 K (header: tfloor 1.3857e-4 ↔ 10⁴ K).
- Problem: M_bh = 2e7 M☉, v_bh = 1000 km/s, n_cgm(t) = 1·exp(−t/8), T_cgm = 1e6 K,
  Mdot(t) = 0.5·exp(−t/15) M☉/yr, v_w = 100 km/s, turbulence σ_v = 30 km/s, tlim = 60.
- Cooling: ISM table with T floor 10⁴ K, **masked where s_00 < 1e-3** (pure CGM never cools).

Snapshot 60 (t = 60 Myr) statistics:
- n_H (with X=0.7): min 7e-5, median 1.5e-2, max 52 cm⁻³; upstream CGM ≈ 4e-4.
- T: ~93,000 cells at the 10⁴ K floor (93,005 of 93,022 are wind gas), ~15,000 at 10⁴–10⁶ K,
  ~33 million at 10⁶–10⁷ K; max 1.7e7 K.
- Cold wind gas (T < 10^5.5) has log n_H ≈ −3…1.7.
- `s_00` overshoots up to 3.5 (numerical): clip to [0, 1].

Binary layout (confirmed): header size = filesize − nmb·block; per meshblock: 6 int32 index,
4 int32 logical (lx1, lx2, lx3, level), 6 float32 geometry, then nvar·nz·ny·nx float32.
(vis_athenak's own readers already handle this.)

---

## 6. Observation settings to model (from the paper)

| Quantity | Value |
|---|---|
| Redshift | z = 0.96 |
| Metallicity | 0.2 Z☉ (SMC-like) for CGM and wind |
| Radiation field (baseline) | KS19 UV background at z = 0.96 + CMB |
| Inclination | motion axis tilted i = 29° toward observer |
| Black hole speed | v• = 954 km/s (sim uses 1000) |
| Pixel | 0.05″ (NIRSpec IFU cube) ≈ 0.41 kpc |
| PSF | ~0.08″ FWHM (adjustable) |
| Spectral resolution | G140M, σ ≈ 130 km/s; channels 193 km/s at [OIII], 147 at Hα |
| Lines | [OIII]5007, Hα, (Hβ), [NII]6584, [SII]6716+6731, [SIII]9069+9532 |
| Comparisons | maps (Figs 2, 3), PV diagram (Figs 5, 7), line-ratio profiles (Fig 8) |
| Observed values | [OIII]/Hα ≈ 0.6 → 1.5 toward tip; [NII]/Hα ≈ 0.02; wake plateau ≈ −300 km/s |

### Projection geometry (BH frame → sky)

```python
# arrays are [x3, x2, x1]; observer direction n = (sin i, 0, cos i)
r_sky = x1*cos(i) - x3*sin(i)       # projected distance from BH along the wake [kpc]
b_sky = x2
v_r   = -((velx + v_bh)*sin(i) + velz*cos(i)) * 1000   # km/s, host-galaxy frame, + = receding
# expected wake plateau (paper eq. 3, chi=3): -v_bh*(1-1/3)*sin(29°) ≈ -323 km/s
```

Map recipe: L_cell = ε·dV → `np.histogram2d(r_sky, b_sky, weights=L)` on 0.05″ pixels →
flux = L/(4π D_L²) → Gaussian PSF → divide by pixel area (arcsec²).
PV: histogram2d(r_sky, v_r) for |b| < 1 kpc, 50 km/s bins, smooth along v by σ_instr.
Ratios: 1D histograms of L along r_sky per line, then divide.

---

## 7. Known caveats / expectations

1. **[OIII] will likely be strongly underpredicted** (expect [OIII]/Hα ≲ 1e-2 vs observed 0.6–1.5).
   Because of the cooling mask, shocked CGM at the tip stays at 10⁶–10⁷ K. The cold gas is wind
   material held at the 10⁴ K floor, where with only the UV background O⁺⁺ is rare. The paper's
   MAPPINGS fits needed strong preshock (precursor) ionization, which a per-cell table built with
   only the UV background leaves out.
   Diagnostic: split maps into wind (`L*fw`) vs CGM (`L*(1-fw)`).
   Follow-up: a second table with an extra ionizing field (shock-precursor-like, or a point source
   at the BH), or estimate line emission from the shock's energy flux using shock-model line ratios.
2. **Box width**: ±2 kpc vs observed bow ±3 kpc, so the bow edges may be truncated.
3. **Wake length**: at t = 60 Myr the projected wake is ~52 kpc (60·cos 29°); observed 62 kpc
   projected ≈ 70 kpc real (paper t_wake ≈ 73 Myr). Compare the tip region, not total length.
4. **Equilibrium + optically thin**: Cloudy tables assume equilibrium ionization and no
   self-shielding/dust; non-equilibrium cooling behind shocks is not captured.
5. **μ**: the sim uses fixed μ = 0.6. Near 10⁴ K the true μ ≈ 1.2, so T there is approximate.
   Kept for consistency with the simulation's own cooling.

---

## 8. Open questions

- ~~Confirm table format v1~~: implemented as in §2 (plus `log_floor`).
- Wind metallicity: baseline 0.2 Z☉ for both. Normal AGN outflows are metal-rich (BLR ~1–5+ Z☉),
  but a runaway BH's wind is mostly reprocessed CGM it accreted, so ~CGM metallicity; the low
  observed [NII]/Hα ≈ 0.02 also favours low Z. Optional check: a 1 Z☉ table applied to the wind
  part only (`s_00`), which avoids a metallicity axis in v1.
- Which extra ionizing source for the second table.
- Single snapshot (60) or full time series / movie.
