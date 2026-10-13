"""
Reading single AthenaK output files (.bin, or .athdf from bin_convert),
built on the repo's ``bin_convert`` and ``athena_read`` modules.
"""

from __future__ import annotations

from pathlib import Path

if "." in __package__:  # imported as vis_athenak.simulation_data
    from ..data_processing import athena_read, bin_convert
else:  # imported as simulation_data, from the repo root
    from data_processing import athena_read, bin_convert

# Output file types that can be read, in order of preference.
EXTENSIONS = ("bin", "athdf")

# With one file per rank, rank 0's directory holds the file handed to the
# reader, which collects the matching files of the other ranks itself.
RANK0_DIR = "rank_00000000"


def read_time(path: Path) -> float:
    """Read only the simulation time from an output file's header."""
    if path.suffix == ".athdf":
        import h5py

        with h5py.File(path, "r") as f:
            return float(f.attrs["Time"])

    with open(path, "rb") as fp:
        fp.readline()  # "Athena ... version=1.1"
        pheader_count = int(fp.readline().split(b"=")[-1])
        for _ in range(pheader_count - 1):
            key, val = fp.readline().decode("utf-8").split("=", 1)
            if key.strip() == "time":
                return float(val)
    raise ValueError(f"no time found in header of {path}")


def read_coordinates(path: Path) -> str:
    """Coordinate system of an output file: "cartesian", "cylindrical",
    "spherical_polar", ...  AthenaK .bin files are always on a Cartesian grid."""
    if path.suffix == ".athdf":
        import h5py

        with h5py.File(path, "r") as f:
            coords = f.attrs.get("Coordinates", b"cartesian")
        return coords.decode("ascii", "replace") if isinstance(coords, bytes) else str(coords)
    return "cartesian"


def read_variable_names(path: Path) -> list[str]:
    """Read only the names of the variables stored in an output file."""
    if path.suffix == ".athdf":
        import h5py

        with h5py.File(path, "r") as f:
            return [v.decode("ascii", "replace") for v in f.attrs["VariableNames"]]

    with open(path, "rb") as fp:
        fp.readline()  # "Athena ... version=1.1"
        pheader_count = int(fp.readline().split(b"=")[-1])
        for _ in range(pheader_count - 1):
            fp.readline()
        fp.readline()  # "number of variables=N"
        line = fp.readline().decode("utf-8")  # "variables:  dens  velx ..."
    if not line.strip().startswith("variables:"):
        raise ValueError(f"no variable list found in header of {path}")
    return line.split()[1:]


def can_read_single_variable(path: Path) -> bool:
    """Whether `read_file` can read some variables of `path` without the rest.
    The .bin reader always reads every variable of a file."""
    return path.suffix == ".athdf"


def read_file(path: Path, dtype=None, quantities: list[str] | None = None) -> dict:
    """
    Read one output file into an athdf-style dict of merged arrays.

    `quantities` limits which variables are read (coordinates and file
    attributes are always included); it is ignored for .bin files, which are
    always read whole.
    """
    if path.suffix == ".athdf":
        return athena_read.athdf(str(path), quantities=quantities, dtype=dtype)
    if path.parent.name == RANK0_DIR:
        return bin_convert.read_all_ranks_binary_as_athdf(str(path), dtype=dtype)
    return bin_convert.read_binary_as_athdf(str(path), dtype=dtype)
