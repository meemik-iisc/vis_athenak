# vis_athenak

Visualization and analysis tools for [AthenaK](https://github.com/IAS-Astrophysics/athenak) simulation outputs.

Developed by the IISc Computational Astrophysics group.

## Structure

```
vis_athenak/
├── main.py              # Entry point: python main.py makes the plots config.py asks for
├── config.py            # What to plot: simulations, variables, styles, layout, switches
├── .env.example         # Template for .env: machine-specific root folders, conda env name
├── .envrc               # direnv: activates the conda env (CONDA_ENV in .env) on cd into the repo
├── simulation_data/     # SimulationData / Frame: lazy, frame-by-frame access to a run
├── plotting/            # 2D slices, combined multi-panel figures, videos
├── utils/               # Unit conversions, .env loading, cooling function
├── data_processing/     # Low-level readers / converters for raw outputs (.bin, .athdf, .prtclbin)
└── test/                # Walkthrough of SimulationData on a real run
```

### What each file does

**Top level**

| File | Purpose |
|---|---|
| `main.py` | Entry point. With no command it runs the tasks switched on in `config.MAIN` (slices, combined figure, video); `slices`, `combined` and `video` run a single task with all its options. |
| `config.py` | Everything you change between runs: `RUN` / `SIMULATIONS` (which runs, with paths relative to the `.env` roots), `PLOT` (slice axis, units, figure size), `FONTS`, `PLOT_VARS` (quantity, units, label, colormap, limits per variable), `PLOT_ORDER`, `COMBINED` (panel grid and video), `MAIN` (task switches) and `N_WORKERS`. |
| `.env.example` | Template for `.env` (gitignored): `ATHENAK_DIR`, `ATHINPUT_DIR`, optional `DATA_DIR`, `CONDA_ENV`. |
| `.envrc` | [direnv](https://direnv.net) config: on entering the repo, activates `CONDA_ENV` from `.env` (the root folders are read by Python, so edits to `.env` apply on the next run). |
| `__init__.py` | Makes the repo importable as the `vis_athenak` package. |
| `pyproject.toml`, `requirements.txt` | Package metadata and dependencies. |

**`simulation_data/`** — reading a run (see [its README](simulation_data/README.md))

| File | Purpose |
|---|---|
| `simulation_data.py` | `SimulationData`: finds a run's output frames from its athinput `<output>` blocks; indexable and iterable. |
| `frame.py` | `Frame`: one snapshot, reading fields and the grid from disk only when used. |
| `field.py` | `Field`: one lazily loaded variable of a frame; works directly with numpy. |
| `athinput.py` | `parse_athinput`: athinput file → `{section: {key: value}}`. |
| `readers.py` | Reads headers (time, variable names) and data from single `.bin` / `.athdf` files. |
| `athdf.py` | Fast single-variable reads from `.athdf` files. |
| `plane.py` | `FramePlane`: a lazy 2D slice of a frame that reads only the meshblocks crossing it (`.bin` via seeks, `.athdf` via h5py selections); falls back to full reads for files it can't handle. |
| `__main__.py` | `python -m simulation_data <athinput> <datafolder>` prints a summary of a run. |

**`plotting/`** — figures

| File | Purpose |
|---|---|
| `quantities.py` | `get(frame, name, params, units)`: raw variables and derived ones (`pres`, `temp`, `vmag`, `bmag`, `beta`, `entropy`, `t_cool`), converted to the units asked for. |
| `slices.py` | `Plane` (a lazy 2D slice of a frame, from `simulation_data/plane.py`: each field's slice is read on first use from only the meshblocks crossing it, and derived quantities are computed on the slice), `take_slice` (slice an already computed 3D array) and `draw_slice` (draw on a matplotlib axes; `imshow` for uniform grids, `pcolormesh` otherwise); plain functions with no config, for your own scripts. |
| `slice2d.py` | `python main.py slices`: one image per frame and variable. Also holds the helpers the other scripts share (variable settings, fonts, worker count). |
| `combined.py` | `python main.py combined`: one multi-panel image per frame, laid out by `config.COMBINED["layout"]`; can compare several runs side by side. |
| `video.py` | `python main.py video`: stitches numbered images into an mp4 with ffmpeg; frames larger than `COMBINED["video_max_size"]` (default 4096 px per side, the encoders' limit) are scaled down to fit. |

**`utils/`** — shared helpers

| File | Purpose |
|---|---|
| `units.py` | `Units`: AthenaK code units → cgs and display units (pc, Myr, km/s, K, cm⁻³, ...), from the athinput `<units>` block. |
| `env.py` | Loads `.env` and reads the root folders (`env_path`); shell variables take precedence. |
| `ism_cooling.py` | `ISMCoolFn(T)`: the ISM cooling function Λ(T), used for `t_cool`. |

**`data_processing/`** — raw file handling

| File | Purpose |
|---|---|
| `athena_read.py` | AthenaK's reader for `.athdf`, `.tab` and other outputs. |
| `bin_convert.py` | AthenaK's `.bin` reader and `.bin` → `.athdf`/`.xdmf` converter. |
| `make_athdf_fast.py` | Converts a whole run's `.bin` files to `.athdf` in parallel. |
| `read_prtcl_bin.py` | `read_particle_binary(file)`: reads AthenaK particle outputs (`.prtclbin`) into a dict of positions, velocities, tags/ids, and grid quantities (e.g. `dens`, `temp`) at the particle locations. |
| `plot_slice.py` | AthenaK's original standalone slice plotter (kept for reference; `plotting/` replaces it). |

**`test/`**

| File | Purpose |
|---|---|
| `test_simulation_data.py` | Walkthrough of every `SimulationData` / `Frame` / `Field` feature on a real run; edit the paths at the top and run `python test/test_simulation_data.py` (needs `pip install -e .`). |

## Setup

Requires Python 3.10+, and [ffmpeg](https://ffmpeg.org) for videos.

```bash
git clone git@github.com:meemik-iisc/vis_athenak.git
cd vis_athenak
pip install -r requirements.txt
cp .env.example .env          # then edit the paths in .env
```

Optionally, with [direnv](https://direnv.net) installed and hooked into your
shell (`eval "$(direnv hook bash)"` at the end of `~/.bashrc`), run
`direnv allow` once in the repo.  From then on, entering it activates
the conda environment named by `CONDA_ENV` in `.env`, and leaving it
deactivates it.

## Making plots

1. **`.env`** — root folders on your machine (`ATHENAK_DIR`, `ATHINPUT_DIR`,
   and `DATA_DIR` if the outputs are not under `ATHENAK_DIR`) and `CONDA_ENV`.
   Variables exported in your shell take precedence.
2. **`config.py`** — add your run to `SIMULATIONS` (`athinput` relative to
   `ATHINPUT_DIR`; `data` and `out` relative to `DATA_DIR`, else
   `ATHENAK_DIR`), set `RUN`, choose the variables (`PLOT_VARS`,
   `PLOT_ORDER`), the panel grid (`COMBINED["layout"]`) and what to make
   (`MAIN`).
3. **Run** from the repo root:

```bash
python main.py                              # what config.MAIN switches on
python main.py --no-make-slices --frames 0:21:5 -w 4
python main.py slices dens temp t_cool      # one task, with all its options
python main.py combined --layout "dens,temp;pres,t_cool" --video
python main.py --no-save-png                # video only: no slices, combined images deleted
python main.py combined --video --no-save-png   # the same, as a single task
python main.py video --fps 5                # video of existing combined images
python main.py [slices|combined|video] --help
```

After `pip install -e .`, `vis_athenak` does the same as `python main.py`
from any directory (e.g. `vis_athenak combined --video`).

Images go to the run's `out` folder from `config.SIMULATIONS`: one folder per
variable for the slices, and `combined/` for the combined images and video.

### config.py switches

`MAIN` decides what `python main.py` makes; each switch can be flipped for one
run with a flag:

| Switch | Flag | Effect |
|---|---|---|
| `make_slices` | `--[no-]make-slices` | One image per frame and variable in `PLOT_ORDER`. |
| `make_combined` | `--[no-]make-combined` | One image per frame of `COMBINED["layout"]`. |
| `make_video` | `--[no-]make-video` | An mp4 of the combined images. |
| `save_png` | `--[no-]save-png` | When off: no slices, and the combined images are deleted once the video is made (only those written in that run; without a video they are kept). |

Frames are plotted in parallel by `N_WORKERS` worker processes (`1` runs
serially); `-w N` / `--workers N` overrides it for one run.

### Combined layout

`COMBINED["layout"]` is a list of rows; each cell is a variable (plotted for
`RUN`), a `(run, variable)` pair, or `None` for an empty panel:

```python
COMBINED["layout"] = [["dens", "temp"],
                      ["pres", "t_cool"]]                       # 2x2, one run
COMBINED["layout"] = [[("run_a", "temp"), ("run_b", "temp")]]   # two runs side by side
```

A row (or column) holding one variable throughout shares one colorbar.
On the command line the same grid is `--layout "dens,temp;pres,t_cool"`
(rows by `;`, cells by `,`, `run:var` for another run, `-` for empty).

## Using SimulationData from Python

```python
from simulation_data import SimulationData

sim = SimulationData("path/to/athinput.kh2d", "path/to/outputs")
len(sim)                    # number of output frames
for frame in sim:
    rho = frame["dens"]     # data is read from disk only here
```

[`test/test_simulation_data.py`](test/test_simulation_data.py) walks through
every feature.  For your own figures, combine it with `plotting.quantities.get`
and `plotting.slices.Plane` / `draw_slice`:

```python
import config
from plotting.quantities import get
from plotting.slices import Plane, draw_slice

run = config.resolve()                    # paths of config.RUN from .env
sim = SimulationData(run["athinput"], run["data"])
plane = Plane(sim[-1], axis="z")          # midplane; position= picks another
T = get(plane, "temp", sim.params, units="K")   # computed on the slice only
draw_slice(ax, plane.x, plane.y, T, cmap="inferno", norm="log")
```

### Using it from another project

Install it into any virtual environment straight from GitHub:

```bash
pip install "git+https://github.com/meemik-iisc/vis_athenak.git"
pip install "vis_athenak[plot] @ git+https://github.com/meemik-iisc/vis_athenak.git"   # + matplotlib, scipy
```

```python
from vis_athenak import SimulationData
```

To work on the code itself, clone it and install it in editable mode with
`pip install -e .`, so edits take effect without reinstalling.

Without installing, you can also clone the repo into your project folder and
import it from there:

```bash
cd my_project
git clone git@github.com:meemik-iisc/vis_athenak.git
```

```python
# my_project/analysis.py, run from my_project/
from vis_athenak import SimulationData
```

The clone has to be named `vis_athenak` (the default), since a folder name
with a hyphen can't be imported.  Install the dependencies with
`pip install -r vis_athenak/requirements.txt`.

## Acknowledgements

`data_processing/athena_read.py`, `bin_convert.py`, and `plot_slice.py` are
taken from [AthenaK's `vis/python`](https://github.com/IAS-Astrophysics/athenak/tree/main/vis/python)
and are distributed under its [BSD-3-Clause license](https://github.com/IAS-Astrophysics/athenak/blob/main/LICENSE).

## Contributors

- Meemik Roy (meemikroy@iisc.ac.in)
- Abhiram K  (abhiram1@iisc.ac.in)
- Behara Sasi Mitra ()
