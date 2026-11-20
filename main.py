"""
Entry point: plot the runs set up in config.py.

    python main.py                       # whatever config.MAIN switches on
    python main.py --no-make-slices --frames 0:21:5 -w 4
    python main.py slices dens t_cool    # one task, with all its options
    python main.py combined --layout "dens,temp;pres,t_cool" --video
    python main.py video --fps 5
    python main.py timeseries --runs turb_sigv_30 turb_tcgm_8192 --overlay
    python main.py emission --frames 60 --lines Ha OIII_5007
    python main.py vtk --run turb_sigv_30 --vars temp s_00   # 3D VTK file for ParaView

Without a command, config.MAIN decides what is made: the single-variable
slices (make_slices), the combined figure (make_combined), its video
(make_video), whether PNGs are kept at all (save_png: if False, the
slices are skipped and the combined images are deleted once in the video),
the gas masses vs time plot (make_timeseries) and the projected emission-line
maps (make_emission).  Each switch has a flag
(--make-video / --no-make-video, ...); --frames and --workers are passed on to
every task, --run to all but the time series (whose runs are
config.TIMESERIES["runs"]) and the emission maps (config.EMISSION["run"]).  A
command runs just that task:
``python main.py <slices|combined|video|timeseries|emission|vtk> --help`` lists its options.
"""

import argparse
import sys

if __package__:  # imported as vis_athenak.main
    from . import config
    from .plotting import combined, slice2d, timeseries, video
    from .plotting.slice2d import n_workers
    from .emission import projection
    from .data_processing import export_vtk
else:            # python main.py
    import config
    from plotting import combined, slice2d, timeseries, video
    from plotting.slice2d import n_workers
    from emission import projection
    from data_processing import export_vtk

COMMANDS = {"slices": slice2d.main, "combined": combined.main, "video": video.main,
            "timeseries": timeseries.main, "emission": projection.main,
            "vtk": export_vtk.main}


def run_tasks(argv) -> None:
    """Make what config.MAIN (or the flags overriding it) switches on."""
    switches = config.MAIN
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Make the plots switched on in config.MAIN.  Commands "
                    "slices, combined and video run a single task.",
    )
    for name, help_text in [
        ("make_slices", "one image per frame and variable in PLOT_ORDER"),
        ("make_combined", "one image per frame of COMBINED['layout']"),
        ("make_video", "a video of the combined images"),
        ("save_png", "keep PNGs; if off, skip the slices and delete the combined "
                     "images once the video is made"),
        ("make_timeseries", "gas masses vs time of TIMESERIES['runs']"),
        ("make_emission", "projected emission-line maps of EMISSION['run']"),
    ]:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name,
                            action=argparse.BooleanOptionalAction, default=switches[name],
                            help=f"{help_text} (default: {switches.get(name, False)})")
    parser.add_argument("--run", default=config.RUN, choices=list(config.SIMULATIONS),
                        help=f"simulation (default: {config.RUN})")
    parser.add_argument("--frames", nargs="+",
                        help="frame numbers and/or ranges like 0:21:5 (default: all)")
    parser.add_argument("-w", "--workers", "-c", "--cores", dest="workers", type=int,
                        default=n_workers(),
                        help="parallel worker processes, 1 = serial "
                             f"(default: config.N_WORKERS = {n_workers()})")
    args, extra = parser.parse_known_args(argv)
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}; to pass options to "
                     "one task use a command, e.g. python main.py slices dens temp")
    if args.make_slices and not args.save_png:
        # The slices are PNGs only, so with save_png off there is nothing to keep.
        print("save_png is off: skipping the single-variable slices")
        args.make_slices = False
    if not (args.make_slices or args.make_combined or args.make_video
            or args.make_timeseries or args.make_emission):
        parser.error("nothing to do: make_slices (or save_png), make_combined, "
                     "make_video, make_timeseries and make_emission are all off")

    common = ["--run", args.run, "--workers", str(args.workers)]
    if args.frames:
        common += ["--frames", *args.frames]

    if args.make_slices:
        print("── slices")
        slice2d.main(common)
    if args.make_combined:
        print("── combined" + (" + video" if args.make_video else ""))
        combined.main(common + (["--video"] if args.make_video else [])
                      + ["--save-png" if args.save_png else "--no-save-png"])
    elif args.make_video:
        # Video of combined images made earlier; save_png only deletes images
        # made in the same run, so these are kept.
        print("── video")
        video.main(["--run", args.run])
    if args.make_timeseries:
        print("── timeseries")
        timeseries.main(["--workers", str(args.workers)]
                        + (["--frames", *args.frames] if args.frames else []))
    if args.make_emission:
        print("── emission")
        projection.main(["--workers", str(args.workers)]
                        + (["--frames", *args.frames] if args.frames else []))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in COMMANDS:
        COMMANDS[argv[0]](argv[1:])
    else:
        run_tasks(argv)


if __name__ == "__main__":
    main()
