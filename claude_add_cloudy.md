# Adding Cloudy line-emission maps to vis_athenak

Notes from a Claude Code session (2026-11-18), started in `/data/data_meemik/RunawayBH/sim_data`.
Goal: make [OIII] and Hα emission maps (plus other optical lines) from AthenaK snapshots and
compare them to the runaway SMBH RBH-1 (van Dokkum et al. 2026, ApJL 998, L27: "JWST
Confirmation of a Runaway Supermassive Black Hole via Its Supersonic Bow Shock").

Working preference: Claude does **not** write code files in this project. It gives snippets in
chat and the user writes the code.

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

## 2. Emission table format (version 1, proposed; not yet confirmed)

One HDF5 file per table, e.g. `emis_KS19_z0.96_Z0.2.h5`:

```
├── attrs:  format="emission_table", format_version=1,
│           X_H, source="Cloudy 25.00", radiation="KS19 z=0.96 + CMB + CR background",
│           metallicity_Zsun=0.2, abundances="Cloudy default solar, scaled", created=...
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

### Generator project (separate repo)

```
emission_tables/
├── config.py            # grid (log_nH, log_T), radiation field, z, metallicity, lines, n_workers
├── .env                 # CLOUDY_EXE, OUTPUT_DIR
├── emtab/
│   ├── cloudy_input.py  # config + (nH, T) -> Cloudy input text
│   ├── run_grid.py      # write inputs, run in parallel, list failures, resume unfinished points
│   ├── collect.py       # parse .ems files -> arrays; compute X_H from `save abundances`
│   ├── table_io.py      # write_table() in format v1 + validate()
│   └── inspect.py       # sanity plots: ε/n² vs T, comparison with known CIE curves
└── tables/              # output .h5 (gitignored)
```

Next step agreed in the session: start with the generator's `config.py` + `cloudy_input.py`,
then vis_athenak's `emission/` module.

---

## 4. Verified facts about the environment

- Cloudy 25.00 is at `~/opt/cloudy/c25.00`, executable `~/bin/cloudy` → `.../source/cloudy.exe`.
  Run with `cloudy -r name` (reads `name.in`). Earlier test files are in `~/opt/cloudy/test/`.
- One-zone test run (~5 s each) confirmed these line labels work in c25:
  `"H  1 6562.81A" "H  1 4861.32A" "O  3 5006.84A" "O  3 4958.91A" "N  2 6583.45A"`
  `"S  2 6716.44A" "S  2 6730.82A" "S  3 9068.62A" "S  3 9530.62A"`
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
save line emissivity "<name>.ems" last
<line labels...>
end of lines
```

Suggested grid: log n_H = −4.5…2 (step 0.25), log T = 4…7.5 (step 0.1) → 972 runs, ~6 min on
14 cores.

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

- Confirm table format v1 (HDF5, ε/n_H², line key names).
- Wind metallicity: same 0.2 Z☉ as CGM, or different? (Would need a metallicity axis later.)
- Which extra ionizing source for the second table.
- Single snapshot (60) or full time series / movie.
