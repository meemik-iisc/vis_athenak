# simulation_data

Lazy, frame-by-frame access to the outputs of an AthenaK or Athena++ run. Give it the
athinput file and the output folder, and you get a sequence of frames. Field
data is read from disk only when a frame is used. Plotting and analysis code
in this repo is meant to build on these classes.

```python
from simulation_data import SimulationData

sim = SimulationData("test_data/kh2d/athinput.kh2d", "test_data/kh2d/bin")
len(sim)                    # 31
for frame in sim:
    rho = frame["dens"]     # only the file holding dens is read here
sim[-1].time                # 3.0, read from the file header only
```

[`test/test_simulation_data.py`](../test/test_simulation_data.py) walks through
every feature. Edit the paths at its top and run it with
`python test/test_simulation_data.py` (it imports the installed `vis_athenak`,
so `pip install -e .` first).

## Layout

| File | Contents |
|---|---|
| `athinput.py` | `parse_athinput(path)`: athinput file → `{section: {key: value}}`, values as strings |
| `readers.py` | `read_time(path)` and `read_variable_names(path)` (header only), and `read_file(path, dtype, quantities)`, built on `data_processing/bin_convert.py` and `athena_read.py` |
| `frame.py` | `Frame`: one snapshot, lazily loaded |
| `field.py` | `Field`: one variable of a frame, lazily loaded |
| `athdf.py` | fast reads of single variables from `.athdf` files (block layout read once per file) |
| `simulation_data.py` | `SimulationData`: finds the frames of a run |
| `__main__.py` | `python -m simulation_data <athinput> <datafolder>`: prints the frames and their times |

Import from the package (`from simulation_data import SimulationData, Frame,
Field, parse_athinput`) and run from the repo root, since `readers.py` imports
`data_processing`.

## SimulationData

```python
SimulationData(athinput, datafolder, outputs=None, dtype=None)
```

| | |
|---|---|
| `len(sim)`, `for frame in sim` | number of frames; iterate in order |
| `sim[i]`, `sim[-1]`, `sim[a:b]` | positional indexing, like a list; slices return a list of `Frame` |
| `sim.frames`, `sim.frame_numbers` | the `Frame` list; file numbers (`NNNNN` in the filename) |
| `sim.times` | time of every frame, from file headers only (cached) |
| `sim.params`, `sim.basename`, `sim.outputs` | parsed athinput, job basename, output ids merged into each frame |

### Finding the frames

* AthenaK writes each `<output>` block with `file_type = bin` as
  `{basename}.{id}.{NNNNN}.bin`, where `id` defaults to `variable`.
  `SimulationData` reads these blocks from the athinput file. It then
  searches `datafolder` recursively, so the run directory or its `bin/`
  subdirectory both work.
* `.athdf` files converted with `bin_convert` are found the same way. If
  both formats are present, `.bin` is used.
* Athena++ runs work too. Athena++ writes each `<outputN>` block with
  `file_type = hdf5` as `{problem_id}.{id}.{NNNNN}.athdf`, where `id`
  defaults to `outN` (e.g. `RunawayBH.out2.00001.athdf`). `sim.basename` is
  then the `problem_id`.
* One file per rank (`bin/rank_XXXXXXXX/`) is supported. Only rank 0's file
  is listed, and the reader collects the other ranks.
* **Output blocks with the same cadence are merged into one frame.** For
  example, `frame["dens"]` (from `hydro_w`) and `frame["mom1"]` (from
  `hydro_u`) come from the same snapshot. Frame numbers only line up between
  outputs written at the same `dt`/`dcycle`.
  * By default, the first bin/hdf5 output and every other one with the
    same cadence are used. Outputs with a different cadence are skipped with
    a warning.
  * Pass `outputs=["mhd_w", "mhd_bcc"]` to choose. Asking for outputs with
    different cadences raises `ValueError`.
* A frame is kept only if every selected output has a file for it.
* Indexing is positional. `frame.number` is the file number. The two differ
  after a restart, or when a frame is missing for one of the outputs.

### Memory

`for frame in sim` frees each frame's data once the loop moves on, so a long
run doesn't fill up memory. Arrays you keep a reference to are not affected.
`sim[a:b]` and `sim.frames` are plain lists, so looping over them keeps data
cached until you call `frame.unload()`.

## Frame

| | |
|---|---|
| `frame.fields` | `{name: Field}` for every variable in the frame's files, from the file headers only |
| `frame["dens"]` | shorthand for `frame.fields["dens"].data`: the array, shape `(nx3, nx2, nx1)` |
| `frame.load(["dens", "mom1"])` | read several fields with one read per file (faster than one at a time, see below) |
| `frame["x1v"]`, `frame.grid` | coordinates (`x1f`, `x1v`, …) and file attributes (`Time`, `NumCycles`, …) |
| `frame.keys()`, `"bcc1" in frame` | field and grid names (reads the grid if a name isn't a field) |
| `frame.time` | simulation time; reads only the header if the grid isn't loaded |
| `frame.number`, `frame.paths` | file number; `{output id: path}` |
| `frame.data` | every field and the grid in one dict (reads everything) |
| `frame.is_loaded`, `frame.unload()` | whether any field is in memory; drop all fields and the grid |

### Fields and what gets read

| | |
|---|---|
| `field.data` | the array, read on first access |
| `field.is_loaded`, `field.unload()` | whether it's in memory; drop it |
| `field.name`, `field.output`, `field.path` | variable name, output id, file it's read from |
| `np.log10(field)` | a `Field` works directly with numpy |

* **`.athdf`**: only the requested fields are read, by `athdf.py`. The
  file's block layout is read once per frame (about 0.25 ms on the 64x128
  test run), and each field then costs about 0.1 ms. The result is
  identical to `athena_read.athdf`, mesh refinement included. Files with
  ghost zones, or slices/sums along an extended dimension, fall back to
  `athena_read.athdf`, which costs about 1 ms per call. Coordinates are only
  read when `frame.grid` or `frame["x1v"]` is used.
* **`.bin`**: the reader can only read a file whole, so asking for one field
  loads every field of that file. Files of the frame's other outputs are
  still not read (`dens` from `hydro_w` doesn't read `hydro_u`).
* If two outputs have a variable with the same name, the first output's is
  `fields["dens"]` and the others are `fields["hydro_u/dens"]`.

Variable names are as stored: AthenaK's (`dens`, `velx`, `eint`, `mom1`,
`ener`, `bcc1`, `s_00`, …) or Athena++'s (`rho`, `press`, `vel1`, `Bcc1`,
`r0`, …). No renaming happens here; `plotting.quantities` accepts the
AthenaK names for both codes. Arrays are
float32 unless you pass `dtype=np.float64`. Frames loaded from `.athdf` also
contain the file's HDF5 attributes (`RootGridSize`, `VariableNames`, …).

## Not supported yet

* Physical units (everything is in code units)
* Derived fields such as temperature
* Coarsened binary output (`cbin`), VTK, and tab outputs

## TODO

1. **Per-field `.bin` reads.** The `.bin` reader still reads whole files.
   Reading one variable at a time from a file would need a reader that
   seeks over the other variables of each meshblock. Fields could also
   carry units (`Field` exists now; `frame["dens"]` still returns a plain
   array).
2. **Derived fields.** These work like fields, but are defined by a function
   that combines other fields, for example temperature from `eint` and `dens`.
   The result is computed when the frame is loaded and stored with the
   frame's other fields.