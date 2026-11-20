# vis_athenak

Visualization and analysis tools for AthenaK (and Athena++ `.athdf`) simulation outputs.
Developed by the IISc Computational Astrophysics group.

## Project structure
- `main.py` — entry point (also the `vis_athenak` command after `pip install -e .`); with no command runs the tasks switched on in `config.MAIN`, or one task: `slices`, `combined`, `video`, `timeseries`, `emission`, `vtk`
- `config.py` — all per-run settings: `RUN` / `SIMULATIONS` (paths relative to `.env` roots, resolved by `config.resolve()`), `PLOT`, `FONTS`, `PLOT_VARS`, `PLOT_ORDER`, `COMBINED` (panel grid + video), `TIMESERIES` (gas masses vs time) + `TIMESERIES_VARS` (label/colour/line style per time-series quantity), `EMISSION` (emission table, line sums, projected-map settings), `MAIN` (task switches), `N_WORKERS`
- `.env` (gitignored; template `.env.example`) — machine-specific `ATHENAK_DIR`, `ATHINPUT_DIR`, optional `DATA_DIR` (root of run outputs, defaults to `ATHENAK_DIR`), `CONDA_ENV`; read by `utils/env.py` on every run (`.envrc` takes only `CONDA_ENV` from it)
- `simulation_data/` — `SimulationData` / `Frame` / `Field`: lazy, frame-by-frame access to a run (athinput + output folder); plotting and utils build on these classes. `Frame.cell_volume` / `cell_volume_factors` give cell volumes (Cartesian, cylindrical, spherical polar). Finds AthenaK `bin` outputs (`<basename>.<id>.NNNNN.bin/.athdf`) and Athena++ `hdf5` outputs (`<problem_id>.outN.NNNNN.athdf`)
  - `plane.py` — `FramePlane`: lazy 2D slice of a frame reading only the crossing meshblocks (`BinLayout` scans .bin block headers; .athdf uses `AthdfLayout` + h5py selections); result equals `np.take` of the full field; rank-split .bin, ghost zones and AthenaK slice/sum outputs fall back to full reads
- `plotting/`
  - `quantities.py` — `get(frame, name, params, units)`; raw variables plus derived ones in `DERIVED`, each with a dimension for unit conversion; `line_<name>` = line emissivity from the emission table (n_H = X_H ρ/m_H); AthenaK names (`dens`, `velx`, `s_00`) alias Athena++'s (`rho`, `vel1`, `r0`) via `ALIASES`
  - `slices.py` — `Plane` (= `simulation_data.plane.FramePlane`), `take_slice` (slices an already computed array), `draw_slice` (imshow for uniform grids, pcolormesh otherwise; axis labels from `frame.coordinates`, e.g. R/z for a cylindrical φ slice); no config. Slice `--position` is in degrees when the normal is an angle (φ, θ)
  - `slice2d.py` — single-variable images; also the shared helpers (`var_settings`, `slice_variable`, `time_title`, `apply_fonts`, `n_workers`, `parse_frames`)
  - `combined.py` — multi-panel image per frame from `COMBINED["layout"]`
  - `video.py` — ffmpeg mp4 from numbered images (libx264, else h264_nvenc, else mpeg4)
  - `timeseries.py` — gas masses vs time (cold `T < T_cold`, wind-polluted `s_00 > s_min`, both, wind material `ρ s_00`, plus `analytic`, the injected `∫Ṁ dt` for Ṁ = Ṁ₀ e^(−t/dt_mdot) from the athinput, and `entrained` = (matched) cold − analytic; any of them optionally rescaled to match `analytic` at `match_at`) for one or more runs, overlaid on one axes or stacked panels; reads whole 3D frames in parallel, sums with `Frame.cell_volume_factors`, caches per run in `<out>/timeseries/masses_T<T>_s<s>.npz`
- `emission/` — line emission from external (e.g. Cloudy) tables: `table.py` (`EmissionTable` reads format v1, log ε/n_H² on (log n_H, log T); `load()` caches per process; `configure`/`default_table`/`components`/`line_emissivities` give the default table and line sums such as SII = 6716+6731); table file from `config.EMISSION["table"]` under `.env` `EMISSION_TABLE_DIR` (`config.emission_table_path()`)
  - `projection.py` — `python main.py emission`: projected line maps on the sky (line of sight (sin i, 0, cos i), r = x1 cos i − x3 sin i, b = x2; Cartesian only), cell luminosities binned with `np.bincount` (optional n³ subsampling), Planck18 D_L / kpc-per-arcsec via scipy, Gaussian PSF, surface brightness (or erg/s/kpc² with `--z none`), ratio maps, optional wind/CGM split; per-frame cache `<out>/emission/cache/<table>_i<i>_p<pixel>_n<n>.NNNNN.npz` (erg/s per pixel, pre-PSF); images `<out>/emission/<name>.NNNNN.png`
- `utils/` — `units.py` (code → cgs/display units, `UNIT_NAMES`, incl. mass in g/Msun), `env.py` (`.env` loading via python-dotenv), `ism_cooling.py` (`ISMCoolFn`)
- `data_processing/` — AthenaK's raw readers/converters (`athena_read.py`, `bin_convert.py`), `make_athdf_fast.py`, `read_prtcl_bin.py` (particle `.prtclbin` reader), `export_vtk.py` (`python main.py vtk`: one frame's fields as a VTK ImageData `.vti`, float32 cell data, log10 for PLOT_VARS with norm="log" (e.g. `log_temp`), optional `--downsample`; Cartesian uniform grids; for ParaView volume rendering, to `<out>/vtk/`); `plot_slice.py` is AthenaK's legacy plotter, superseded by `plotting/`
- `test/test_simulation_data.py` — walkthrough of the SimulationData API on a real run

## Related work
- Cooling function implementations live in the separate `Athenak_cooling` repo

## Conventions
- Python 3.10+; dependencies in both `requirements.txt` and `pyproject.toml`
- Run scripts from the repo root (`python main.py ...` or `python -m plotting.<module>`), never as `python plotting/x.py`
- Modules support both import styles: `if "." in __package__:` relative imports (as `vis_athenak.*`), else top-level imports (run from the root); `config.py` uses `if __package__:`
- Machine-specific paths go only in `.env`; `config.py` holds paths relative to those roots and is the same on every machine
- New settings go in `config.py` with a command-line flag overriding them; plotting functions take settings as arguments rather than reading config globals (except the CLI entry points and `slice2d` helpers)
- New plottable quantities: add to `DERIVED` in `plotting/quantities.py` (code units, with a dimension) and a style entry in `PLOT_VARS`; derived functions must use only `frame[...]` / `_raw(frame, ...)` (AthenaK names, resolved for Athena++ too) and elementwise numpy so they work on a `Plane`
- `Frame` itself loads whole 3D variables (lazy = deferred, per variable); for 2D work go through `FramePlane` / `plotting.slices.Plane`, which reads only the slice
- Frames are plotted in parallel with `ProcessPoolExecutor`; worker functions must be module-level and call `apply_fonts()` themselves
- Raw simulation data is never committed (`*.bin`, `*.athdf`, `*.h5`, ... are gitignored); it lives outside the repo, located through `.env`
- Processed arrays may be cached as `.npy` files (also gitignored)
- Each script should be runnable standalone with a clear argparse interface

## Common tasks Claude should help with
- Reading AthenaK HDF5/binary output formats
- Writing efficient numpy-based reduction pipelines
- Plotting 2D slices and radial profiles of MHD quantities (density, temperature, B-field)
- Comparing outputs across runs with different cooling prescriptions
