"""
Reading one 2D slice of a frame without reading the rest of the box.

    plane = FramePlane(frame, axis="z", position=0.0)   # nothing is read yet
    plane.x, plane.y         # cell faces of the slice (file headers only)
    plane["dens"]            # reads only the meshblocks crossing the slice
    plane.load(["velx", "vely"])   # several fields in one pass per file

A slice at finest-level index ``i`` along the normal axis is crossed, at each
refinement level, by one layer of meshblocks, and every block of that layer
holds the slice at the same index inside the block.  Only those blocks are
read, and only the variables asked for:

- ``.bin``: the block records are scanned once (`BinLayout`), reading only the
  small header of each, then the needed blocks are read with seeks.
- ``.athdf``: one h5py read per level and variable selects just the planes.

The planes are merged as the full readers merge blocks (each block repeated
``2**(max_level - level)`` times per axis and placed at its logical location),
so ``plane[name]`` equals ``np.take(frame[name], plane.index, axis)``.

Files these readers do not handle (rank-split .bin outputs, ghost zones,
slice or sum outputs of AthenaK) fall back to reading the whole field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import athdf
from .readers import RANK0_DIR

# Slice normal -> array axis of (nx3, nx2, nx1) arrays, and the in-plane
# (horizontal, vertical) coordinates.
AXES = {"z": (0, "x1", "x2"), "y": (1, "x1", "x3"), "x": (2, "x2", "x3")}
NORMAL = {0: "x3", 1: "x2", 2: "x1"}


# ── .bin layout ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BinLayout:
    """Header and block table of one .bin file, read without its data."""

    var_names: list[str]
    var_dtype: np.dtype           # dtype of the stored variables
    levels: np.ndarray            # (nmb,) refinement level of each block
    locations: np.ndarray         # (nmb, 3) logical location (x1, x2, x3)
    data_offsets: np.ndarray      # (nmb,) byte offset of each block's data
    block_size: tuple[int, int, int]   # (nx1, nx2, nx3) output cells per block
    shape: tuple[int, int, int]        # (nx3, nx2, nx1) of the merged array
    max_level: int
    extent: tuple[tuple[float, float], ...]  # ((x1min, x1max), (x2..), (x3..))
    time: float
    cycle: int
    supported: bool               # False: read the whole file instead

    @classmethod
    def read(cls, path: Path) -> BinLayout:
        with open(path, "rb") as fp:
            fp.seek(0, 2)
            filesize = fp.tell()
            fp.seek(0)

            code_header = fp.readline().split()
            if not code_header or code_header[0] != b"Athena":
                raise TypeError(f"{path} is not an AthenaK .bin file")
            pheader_count = int(fp.readline().split(b"=")[-1])
            pheader = {}
            for _ in range(pheader_count - 1):
                key, val = fp.readline().decode("utf-8").split("=", 1)
                pheader[key.strip()] = val.strip()
            nvars = int(fp.readline().split(b"=")[-1])
            var_names = [v.decode("utf-8") for v in fp.readline().split()[1:]]
            header_size = int(fp.readline().split(b"=")[-1])
            params = _parse_input_text(fp.read(header_size).decode("utf-8"))

            locsize = int(pheader["size of location"])
            varsize = int(pheader["size of variable"])
            var_dtype = np.dtype(np.float64 if varsize == 8 else np.float32)
            nghost = int(params["mesh"]["nghost"])

            # Scan the block records: 24 B index range, 16 B logical location
            # and level, 6 locations of geometry, then nvars * ncells values.
            levels, locations, offsets, sizes = [], [], [], set()
            while fp.tell() < filesize:
                index = np.frombuffer(fp.read(24), dtype=np.int32).astype(np.int64) - nghost
                logical = np.frombuffer(fp.read(16), dtype=np.int32)
                size = (int(index[1] - index[0] + 1), int(index[3] - index[2] + 1),
                        int(index[5] - index[4] + 1))
                sizes.add(size)
                fp.seek(6 * locsize, 1)
                offsets.append(fp.tell())
                locations.append(logical[:3].astype(np.int64))
                levels.append(int(logical[3]))
                fp.seek(nvars * size[0] * size[1] * size[2] * varsize, 1)

        root = [int(params["mesh"][f"nx{d}"]) for d in (1, 2, 3)]
        meshblock = [int(params["meshblock"][f"nx{d}"]) for d in (1, 2, 3)]
        block_size = next(iter(sizes))
        max_level = max(levels)
        nx = [r * 2**max_level if b > 1 else 1 for b, r in zip(block_size, root)]
        # Same-size blocks holding whole meshblocks (no ghost zones, no slice
        # or sum along an extended dimension), as the merge below assumes.
        supported = len(sizes) == 1 and all(
            b == m if r > 1 else b == 1
            for b, m, r in zip(block_size, meshblock, root))
        extent = tuple((float(params["mesh"][f"x{d}min"]), float(params["mesh"][f"x{d}max"]))
                       for d in (1, 2, 3))
        return cls(
            var_names=var_names,
            var_dtype=var_dtype,
            levels=np.array(levels),
            locations=np.array(locations),
            data_offsets=np.array(offsets),
            block_size=block_size,
            shape=(nx[2], nx[1], nx[0]),
            max_level=max_level,
            extent=extent,
            time=float(pheader["time"]),
            cycle=int(pheader["cycle"]),
            supported=supported,
        )

    def grid(self, dtype=None) -> dict:
        """Coordinates and attributes as ``read_binary_as_athdf`` builds them."""
        dtype = np.float32 if dtype is None else dtype
        grid = {}
        for d, n in zip((1, 2, 3), self.shape[::-1]):
            lo, hi = self.extent[d - 1]
            faces = (np.array([lo, hi], dtype=dtype) if n == 1
                     else np.linspace(lo, hi, n + 1, dtype=dtype))
            grid[f"x{d}f"] = faces
            grid[f"x{d}v"] = (0.5 * (faces[:-1] + faces[1:])).astype(dtype)
        grid.update(Time=self.time, NumCycles=self.cycle, MaxLevel=self.max_level)
        return grid


def _parse_input_text(text: str) -> dict[str, dict[str, str]]:
    """The athinput copy in a .bin header -> {block: {key: value}}."""
    params, block = {}, None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            block = line[1:-1].strip().lower()
            params.setdefault(block, {})
        elif block is not None and "=" in line:
            key, value = line.split("=", 1)
            params[block][key.strip()] = value.strip()
    return params


# ── Which blocks cross a plane ───────────────────────────────────────────

def _crossing_blocks(levels, locations, block_size, shape, max_level, cut, index):
    """
    Blocks crossing finest-level index ``index`` along array axis ``cut``.

    Yields ``(block, local index in the block, scale, (rows, cols))``, where
    rows and cols are the slices of the 2D plane the block fills.
    """
    bs = block_size[::-1]                 # (nx3, nx2, nx1), matching the array axes
    extended = [n > 1 for n in shape]
    inplane = [a for a in range(3) if a != cut]
    for b, (level, loc) in enumerate(zip(levels, locations)):
        s = 2 ** (max_level - int(level))
        loc = loc[::-1]                   # (x3, x2, x1)
        if extended[cut]:
            start = loc[cut] * bs[cut] * s
            if not start <= index < start + bs[cut] * s:
                continue
            local = (index - start) // s
        else:
            local = 0
        dest = tuple(
            slice(loc[a] * bs[a] * s, (loc[a] + 1) * bs[a] * s) if extended[a]
            else slice(0, 1)
            for a in inplane
        )
        yield b, int(local), s, dest


def _place(out, block_plane, s, dest, extended_inplane):
    """Prolongate a block's plane by ``s`` along extended axes and place it."""
    for ax, ext in enumerate(extended_inplane):
        if s > 1 and ext:
            block_plane = np.repeat(block_plane, s, axis=ax)
    out[dest] = block_plane


def _plane_shape(shape, cut):
    return tuple(n for a, n in enumerate(shape) if a != cut)


# ── Reading planes ───────────────────────────────────────────────────────

def read_bin_plane(path: Path, layout: BinLayout, names: list[str], cut: int,
                   index: int, dtype=None) -> dict[str, np.ndarray]:
    """Planes of ``names`` from a .bin file, reading only the crossing blocks."""
    unknown = [n for n in names if n not in layout.var_names]
    if unknown:
        raise KeyError(f"{unknown} not in {path}; it has {layout.var_names}")
    dtype = np.float32 if dtype is None else dtype
    out = {n: np.zeros(_plane_shape(layout.shape, cut), dtype=dtype) for n in names}
    bs = layout.block_size[::-1]          # (nz, ny, nx) of each block's data
    ncells = bs[0] * bs[1] * bs[2]
    itemsize = layout.var_dtype.itemsize
    extended_inplane = [layout.shape[a] > 1 for a in range(3) if a != cut]

    with open(path, "rb") as fp:
        for b, local, s, dest in _crossing_blocks(
                layout.levels, layout.locations, layout.block_size, layout.shape,
                layout.max_level, cut, index):
            for n in names:
                start = layout.data_offsets[b] + layout.var_names.index(n) * ncells * itemsize
                if cut == 0:  # a z-plane is contiguous within the block
                    fp.seek(start + local * bs[1] * bs[2] * itemsize)
                    block_plane = np.fromfile(fp, layout.var_dtype, bs[1] * bs[2])
                    block_plane = block_plane.reshape(bs[1], bs[2])
                else:
                    fp.seek(start)
                    block = np.fromfile(fp, layout.var_dtype, ncells).reshape(bs)
                    block_plane = np.take(block, local, axis=cut)
                _place(out[n], block_plane, s, dest, extended_inplane)
    return out


def read_athdf_plane(path: Path, layout: athdf.AthdfLayout, names: list[str], cut: int,
                     index: int, dtype=None) -> dict[str, np.ndarray]:
    """
    Planes of ``names`` from a .athdf file; one h5py read per variable and
    distinct in-block index (usually one per refinement level).
    """
    import h5py

    unknown = [n for n in names if n not in layout.variables]
    if unknown:
        raise KeyError(f"{unknown} not in {path}; it has {list(layout.variables)}")
    dtype = layout.dtype if dtype is None else np.dtype(dtype)
    out = {n: np.zeros(_plane_shape(layout.shape, cut), dtype=dtype) for n in names}
    extended_inplane = [layout.shape[a] > 1 for a in range(3) if a != cut]

    # Blocks of one level crossing the plane share the same in-block index,
    # so each group is read with a single selection.
    by_local: dict[int, list] = {}
    for b, local, s, dest in _crossing_blocks(
            layout.levels, layout.locations, layout.block_size, layout.shape,
            layout.max_level, cut, index):
        by_local.setdefault(local, []).append((b, s, dest))

    with h5py.File(path, "r") as f:
        for n in names:
            dataset, var = layout.variables[n]
            for local, blocks in by_local.items():
                ids = [b for b, _, _ in blocks]           # ascending, as h5py needs
                selection = [var, ids, slice(None), slice(None), slice(None)]
                selection[2 + cut] = local                # axes after (var, block)
                planes = f[dataset][tuple(selection)]     # (len(ids), rows, cols)
                for plane, (_, s, dest) in zip(planes, blocks):
                    _place(out[n], plane, s, dest, extended_inplane)
    return out


# ── Lazy plane of a frame ────────────────────────────────────────────────

class FramePlane:
    """
    One 2D slice of a `Frame`, read lazily and only where the slice is.

    Nothing is read when it is created.  ``index``, ``x`` and ``y`` need only
    the file headers; ``plane[name]`` reads that field's slice on first use
    and caches it.  Fields the frame already holds in memory are sliced from
    there instead of being read again.
    """

    def __init__(self, frame, axis: str = "z", position: float | None = None):
        if axis not in AXES:
            raise ValueError(f"axis must be one of {list(AXES)}, got {axis!r}")
        self.frame, self.axis, self.position = frame, axis, position
        self._cut, self._h, self._v = AXES[axis]
        self._planes: dict[str, np.ndarray] = {}
        self._bin_layouts: dict[Path, BinLayout] = {}
        self._grid: dict | None = None
        self._index: int | None = None

    # ── Geometry (headers only) ──────────────────────────────────────────

    def _first_path(self) -> Path:
        return next(iter(self.frame.paths.values()))

    def _bin_layout(self, path: Path) -> BinLayout | None:
        if path.suffix != ".bin" or path.parent.name == RANK0_DIR:
            return None
        if path not in self._bin_layouts:
            self._bin_layouts[path] = BinLayout.read(path)
        return self._bin_layouts[path]

    @property
    def grid(self) -> dict:
        """Coordinates of the frame, without reading field data where possible."""
        if self._grid is None:
            layout = self._bin_layout(self._first_path())
            if self.frame._grid is None and layout is not None and layout.supported:
                self._grid = layout.grid(self.frame._dtype)
            else:  # .athdf grids are header reads already
                self._grid = self.frame.grid
        return self._grid

    @property
    def index(self) -> int:
        """Finest-level cell index of the slice along the normal axis."""
        if self._index is None:
            faces = self.grid[f"{NORMAL[self._cut]}f"]
            ncells = len(faces) - 1
            if self.position is None:
                self._index = ncells // 2
            else:
                if not faces[0] <= self.position <= faces[-1]:
                    raise ValueError(f"{NORMAL[self._cut]} = {self.position} outside "
                                     f"[{faces[0]}, {faces[-1]}]")
                i = int(np.searchsorted(faces, self.position, side="right")) - 1
                self._index = min(i, ncells - 1)
        return self._index

    @property
    def in_plane(self) -> tuple[str, str]:
        """Coordinates along the (horizontal, vertical) axes, e.g. ("x1", "x3")."""
        return self._h, self._v

    @property
    def x(self) -> np.ndarray:
        """Cell faces along the horizontal axis of the slice."""
        return self.grid[f"{self._h}f"]

    @property
    def y(self) -> np.ndarray:
        """Cell faces along the vertical axis of the slice."""
        return self.grid[f"{self._v}f"]

    @property
    def time(self) -> float:
        return self.frame.time

    @property
    def number(self) -> int:
        return self.frame.number

    @property
    def coordinates(self) -> str:
        return self.frame.coordinates

    @property
    def fields(self):
        return self.frame.fields

    # ── Fields ───────────────────────────────────────────────────────────

    def load(self, names) -> None:
        """Read the slices of several fields, one pass per file."""
        wanted = [n for n in names if n not in self._planes]
        unknown = [n for n in wanted if n not in self.frame.fields]
        if unknown:
            raise KeyError(f"{unknown} not in frame {self.frame.number}; "
                           f"fields: {sorted(self.frame.fields)}")

        by_path: dict[Path, list[str]] = {}
        for key in wanted:
            field = self.frame.fields[key]
            if field.is_loaded:  # already in memory: just slice it
                self._planes[key] = np.take(field.data, self.index, axis=self._cut)
            else:
                by_path.setdefault(field.path, []).append(key)

        for path, keys in by_path.items():
            names_in_file = [self.frame.fields[k].name for k in keys]
            planes = self._read(path, names_in_file)
            for key, name in zip(keys, names_in_file):
                self._planes[key] = planes[name]

    def _read(self, path: Path, names: list[str]) -> dict[str, np.ndarray]:
        dtype = self.frame._dtype
        layout = self._bin_layout(path)
        if layout is not None and layout.supported:
            return read_bin_plane(path, layout, names, self._cut, self.index, dtype)
        athdf_layout = self.frame._athdf_layout(path)
        if athdf_layout is not None and athdf_layout.supported:
            return read_athdf_plane(path, athdf_layout, names, self._cut, self.index, dtype)
        # Anything else: read the whole fields and slice them.
        keys = {f.name: k for k, f in self.frame.fields.items() if f.path == path}
        return {n: np.take(self.frame[keys[n]], self.index, axis=self._cut) for n in names}

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self.frame.fields:
            if name not in self._planes:
                self.load([name])
            return self._planes[name]
        return self.grid[name]  # coordinates and attributes as they are

    def __contains__(self, name: str) -> bool:
        return name in self.frame.fields or name in self.grid

    def unload(self) -> None:
        """Drop the cached slices; block layouts and the grid are kept."""
        self._planes.clear()

    def __repr__(self) -> str:
        where = "midplane" if self.position is None else f"{self.position}"
        return (f"<FramePlane {self.axis} = {where} of frame {self.frame.number}  "
                f"{len(self._planes)} fields read>")
