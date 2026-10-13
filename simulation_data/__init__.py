"""
Lazy, frame-by-frame access to the outputs of an AthenaK or Athena++ run.

    from simulation_data import SimulationData

    sim = SimulationData("runs/blast/athinput.blast", "runs/blast")
    len(sim)                    # number of output frames
    for frame in sim:
        rho = frame["dens"]     # only the file holding dens is read here
    sim[-1].time                # header-only read, no field data loaded
    sim[-1].fields              # {name: Field}, from the file headers only

Frames are discovered from the <output> blocks of the athinput file: an
output block with ``file_type = bin`` writes ``{basename}.{id}.{NNNNN}.bin``
(``id`` defaults to ``variable``).  Output blocks that share the same cadence
(e.g. ``mhd_w`` and ``mhd_bcc``) are merged, so a single Frame exposes the
variables of all of them.  ``.athdf`` files produced by ``bin_convert`` are
picked up as well.  Athena++ runs work the same way: an output block with
``file_type = hdf5`` writes ``{problem_id}.{id}.{NNNNN}.athdf``, where ``id``
defaults to ``outN`` for ``<outputN>``.

Modules
-------
athinput         parse_athinput: athinput file -> {section: {key: value}}
readers          read_time / read_variable_names / read_file: one output file ->
                 header time / variable names / arrays
frame            Frame: one lazily loaded snapshot
field            Field: one lazily loaded variable of a Frame
athdf            fast reads of single variables from .athdf files
simulation_data  SimulationData: discovers the frames of a run

Run as a module to print a summary of a run:

    python -m simulation_data athinput.blast runs/blast
"""

from .athinput import parse_athinput
from .field import Field
from .frame import Frame
from .simulation_data import SimulationData

__all__ = ["Field", "Frame", "SimulationData", "parse_athinput"]
