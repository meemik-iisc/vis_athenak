# vis_athenak

Visualization and analysis tools for AthenaK simulation outputs.
Developed by the IISc Computational Astrophysics group.

## Project structure
- `main.py` — entry point (also the `vis_athenak` command after `pip install -e .`); with no command runs the tasks switched on in `config.MAIN`, or one task: `slices`, `combined`, `video`
- `config.py` — all per-run settings: `RUN` / `SIMULATIONS` (paths relative to `.env` roots, resolved by `config.resolve()`), `PLOT`, `FONTS`, `PLOT_VARS`, `PLOT_ORDER`, `COMBINED` (panel grid + video), `MAIN` (task switches), `N_WORKERS`
- `.env` (gitignored; template `.env.example`) — machine-specific `ATHENAK_DIR`, `ATHINPUT_DIR`, optional `DATA_DIR` (root of run outputs, defaults to `ATHENAK_DIR`), `CONDA_ENV`; read by `utils/env.py` on every run (`.envrc` takes only `CONDA_ENV` from it)
- `simulation_data/` — `SimulationData` / `Frame` / `Field`: lazy, frame-by-frame access to a run (athinput + output folder); plotting and utils build on these classes
  - `plane.py` — `FramePlane`: lazy 2D slice of a frame reading only the crossing meshblocks (`BinLayout` scans .bin block headers; .athdf uses `AthdfLayout` + h5py selections); result equals `np.take` of the full field; rank-split .bin, ghost zones and AthenaK slice/sum outputs fall back to full reads
- `plotting/`
  - `quantities.py` — `get(frame, name, params, units)`; raw variables plus derived ones in `DERIVED`, each with a dimension for unit conversion
  - `slices.py` — `Plane` (= `simulation_data.plane.FramePlane`), `take_slice` (slices an already computed array), `draw_slice` (imshow for uniform grids, pcolormesh otherwise); no config
  - `slice2d.py` — single-variable images; also the shared helpers (`var_settings`, `slice_variable`, `time_title`, `apply_fonts`, `n_workers`, `parse_frames`)
  - `combined.py` — multi-panel image per frame from `COMBINED["layout"]`
  - `video.py` — ffmpeg mp4 from numbered images (libx264, else h264_nvenc, else mpeg4)
- `utils/` — `units.py` (code → cgs/display units, `UNIT_NAMES`), `env.py` (`.env` loading via python-dotenv), `ism_cooling.py` (`ISMCoolFn`)
- `data_processing/` — AthenaK's raw readers/converters (`athena_read.py`, `bin_convert.py`), `make_athdf_fast.py`, `read_prtcl_bin.py` (particle `.prtclbin` reader); `plot_slice.py` is AthenaK's legacy plotter, superseded by `plotting/`
- `test/test_simulation_data.py` — walkthrough of the SimulationData API on a real run

## Related work
- Cooling function implementations live in the separate `Athenak_cooling` repo

## Conventions
- Python 3.10+; dependencies in both `requirements.txt` and `pyproject.toml`
- Run scripts from the repo root (`python main.py ...` or `python -m plotting.<module>`), never as `python plotting/x.py`
- Modules support both import styles: `if "." in __package__:` relative imports (as `vis_athenak.*`), else top-level imports (run from the root); `config.py` uses `if __package__:`
- Machine-specific paths go only in `.env`; `config.py` holds paths relative to those roots and is the same on every machine
- New settings go in `config.py` with a command-line flag overriding them; plotting functions take settings as arguments rather than reading config globals (except the CLI entry points and `slice2d` helpers)
- New plottable quantities: add to `DERIVED` in `plotting/quantities.py` (code units, with a dimension) and a style entry in `PLOT_VARS`; derived functions must use only `frame[...]` and elementwise numpy so they work on a `Plane`
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
