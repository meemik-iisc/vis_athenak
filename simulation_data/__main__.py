"""
Print a summary of the outputs of an AthenaK run.

    python -m simulation_data athinput.blast runs/blast
"""

import argparse

from .simulation_data import SimulationData


def main():
    parser = argparse.ArgumentParser(
        prog="python -m simulation_data",
        description="Summarise the outputs of an AthenaK run.",
    )
    parser.add_argument("athinput", help="athinput file of the run")
    parser.add_argument("datafolder", help="directory containing the output files")
    parser.add_argument(
        "--outputs", nargs="+", default=None,
        help="output ids to include (default: bin outputs sharing the first one's dt)",
    )
    parser.add_argument("--basename", help="output file basename, if not the athinput's")
    args = parser.parse_args()

    sim = SimulationData(args.athinput, args.datafolder, outputs=args.outputs,
                         basename=args.basename)
    print(sim)
    for frame, t in zip(sim, sim.times):
        print(f"  frame {frame.number:5d}  t = {t:.6g}")


if __name__ == "__main__":
    main()
