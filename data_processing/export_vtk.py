"""
Export 3D fields of one frame as a VTK image (.vti) for ParaView / PyVista.

    python main.py vtk                                # temp of the last frame of config.RUN
    python main.py vtk --run turb_sigv_30 --frame 60 --vars temp s_00 dens
    python -m data_processing.export_vtk --vars temp --downsample 2

Each variable is a PLOT_VARS entry (or a quantity name), written in that
entry's units as float32 cell data; entries with norm="log" are written as
log10 and named log_<var> (temp -> log_temp, log10 of T in K).  Values <= 0
of a log variable become NaN.  Coordinates are in --length-units (kpc).
--downsample n averages n x n x n cells (of the written values) to cut the
file size by n^3.

The file goes to ``<run out>/vtk/<basename>.<NNNNN>.vti`` (one frame of
2048 x 128 x 128 cells is ~135 MB per variable).  Needs a Cartesian run with
uniform cells (no mesh refinement).  In ParaView: open the file, Apply, set
the representation to Volume and colour by log_temp.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

if "." in __package__:  # imported as vis_athenak.data_processing
    from .. import config
    from ..plotting.quantities import get
    from ..plotting.slice2d import unit_factor
    from ..simulation_data import SimulationData
else:  # run from the repo root
    import config
    from plotting.quantities import get
    from plotting.slice2d import unit_factor
    from simulation_data import SimulationData


def frame_arrays(frame, params: dict, variables: list[str], downsample: int = 1
                 ) -> dict[str, np.ndarray]:
    """{name in the file: float32 (nx3, nx2, nx1) array} of each variable."""
    out = {}
    for var in variables:
        settings = config.plot_var(var)
        data = np.asarray(get(frame, settings["quantity"], params, units=settings["units"]),
                          dtype=np.float64)
        name = var
        if settings.get("norm") == "log":
            with np.errstate(divide="ignore", invalid="ignore"):
                data = np.where(data > 0, np.log10(data), np.nan)
            name = f"log_{var}"
        out[name] = _block_mean(data, downsample).astype(np.float32)
    return out


def _block_mean(a: np.ndarray, n: int) -> np.ndarray:
    """Mean over n x n x n blocks (trailing cells that do not fill a block are dropped)."""
    if n == 1:
        return a
    k, j, i = (s // n for s in a.shape)
    a = a[:k * n, :j * n, :i * n]
    return np.nanmean(a.reshape(k, n, j, n, i, n), axis=(1, 3, 5))


def write_vti(path: Path, arrays: dict[str, np.ndarray], origin, spacing) -> Path:
    """
    VTK XML ImageData file with ``arrays`` (each (nz, ny, nx), x fastest) as
    cell data, raw appended binary.  ``origin`` is the corner of the first
    cell and ``spacing`` the cell size, both (x, y, z).
    """
    nz, ny, nx = next(iter(arrays.values())).shape
    extent = f"0 {nx} 0 {ny} 0 {nz}"
    blocks, offset, entries = [], 0, []
    for name, a in arrays.items():
        raw = np.ascontiguousarray(a, dtype="<f4").tobytes()
        entries.append(f'        <DataArray type="Float32" Name="{name}" format="appended" '
                       f'offset="{offset}"/>')
        blocks.append((raw, offset))
        offset += 8 + len(raw)
    header = "\n".join([
        '<?xml version="1.0"?>',
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
        f'  <ImageData WholeExtent="{extent}" Origin="{origin[0]:.9g} {origin[1]:.9g} '
        f'{origin[2]:.9g}" Spacing="{spacing[0]:.9g} {spacing[1]:.9g} {spacing[2]:.9g}">',
        f'    <Piece Extent="{extent}">',
        f'      <CellData Scalars="{next(iter(arrays))}">',
        *entries,
        '      </CellData>',
        '    </Piece>',
        '  </ImageData>',
        '  <AppendedData encoding="raw">',
        '_',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(header.encode())
        for raw, _ in blocks:
            f.write(np.uint64(len(raw)).tobytes())
            f.write(raw)
        f.write(b"\n  </AppendedData>\n</VTKFile>\n")
    tmp.replace(path)
    return path


def export_frame(frame, params: dict, variables: list[str], path: Path,
                 length_units: str = "kpc", downsample: int = 1) -> Path:
    if frame.coordinates != "cartesian":
        raise NotImplementedError(f"VTK image export needs Cartesian runs, not {frame.coordinates!r}")
    length = unit_factor(params, "length", length_units)
    faces = [np.asarray(frame.grid[k], dtype=float) * length for k in ("x1f", "x2f", "x3f")]
    widths = [np.diff(f) for f in faces]
    if any(not np.allclose(w, w[0], rtol=1e-6) for w in widths):
        raise NotImplementedError("cells are not uniform (mesh refinement?); .vti needs a uniform grid")
    arrays = frame_arrays(frame, params, variables, downsample)
    origin = [f[0] for f in faces]
    spacing = [w[0] * downsample for w in widths]
    return write_vti(path, arrays, origin, spacing)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="main.py vtk", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=config.RUN, choices=list(config.SIMULATIONS),
                        help=f"simulation (default: {config.RUN})")
    parser.add_argument("--frame", type=int, help="frame number (default: the last)")
    parser.add_argument("--vars", nargs="+", default=["temp"],
                        help="PLOT_VARS entries or quantities to write (default: temp)")
    parser.add_argument("--downsample", type=int, default=1,
                        help="average n x n x n cells (default: 1 = full resolution)")
    parser.add_argument("--length-units", default="kpc", help="coordinates: code, pc, kpc")
    parser.add_argument("--output", help="file to write (default: <run out>/vtk/<basename>.<NNNNN>.vti)")
    args = parser.parse_args(argv)
    if args.downsample < 1:
        parser.error("--downsample must be >= 1")

    settings = config.resolve(args.run)
    sim = SimulationData(settings["athinput"], settings["data"], outputs=settings["outputs"])
    number = args.frame if args.frame is not None else sim.frame_numbers[-1]
    if number not in sim.frame_numbers:
        parser.error(f"no frame {number}; run has frames {sim.frame_numbers}")
    frame = sim.frames[sim.frame_numbers.index(number)]
    basename = sim.params.get("job", {}).get("basename", args.run)
    path = Path(args.output) if args.output else (
        settings["out"] / "vtk" / f"{basename}.{number:05d}.vti")
    print(f"{args.run} frame {number}: writing {args.vars} ...", flush=True)
    path = export_frame(frame, sim.params, args.vars, path, args.length_units, args.downsample)
    print(f"saved {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
