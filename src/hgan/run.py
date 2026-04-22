import argparse
import glob
import logging
import os
import os.path
import re
import torch
import torch.distributed as dist
from hgan.configuration import load_config
from hgan.experiment import Experiment


logger = logging.getLogger("hgan")


def _in_ddp():
    ws = os.environ.get("WORLD_SIZE", "1")
    lr = os.environ.get("LOCAL_RANK", "")
    try:
        return int(ws) > 1 and lr != ""
    except ValueError:
        return False


def _broadcast_run_dir(run_dir):
    """Broadcast rank-0's `run_dir` string to all ranks so they share one folder."""
    if not (_in_ddp() and dist.is_initialized()):
        return run_dir
    # Pack string through a simple object list broadcast.
    obj_list = [run_dir] if dist.get_rank() == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def _system_tag(config):
    """Short label describing which system/data source this run trains on."""
    gen = getattr(config.experiment, "rt_data_generator", None)
    if gen in (None, "", "hgn", "dm"):
        return config.experiment.system_name or "unknown"
    return gen  # e.g. "real_pendulum"


def _next_run_dir(base_output, tag):
    """Return `<base_output>/<tag>_v<N>` where N is the next available integer."""
    pattern = re.compile(rf"^{re.escape(tag)}_v(\d+)$")
    existing = 0
    for path in glob.glob(os.path.join(base_output, f"{tag}_v*")):
        m = pattern.match(os.path.basename(path))
        if m:
            existing = max(existing, int(m.group(1)))
    return os.path.join(base_output, f"{tag}_v{existing + 1}")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to configuration.ini specifying experiment parameters",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subdirectory name under paths.output for this run "
        "(default: <system>_v<N> auto-incremented).",
    )
    return parser


def main(*args):
    args = get_parser().parse_args(args)

    config = load_config(args.config_path)

    # When launched via torchrun, init the process group *before* deciding the
    # run directory so rank 0 can broadcast the chosen path to other ranks.
    if _in_ddp() and not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    base_output = config.paths.output
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        os.makedirs(base_output, exist_ok=True)
        if args.run_name:
            run_dir = os.path.join(base_output, args.run_name)
        else:
            run_dir = _next_run_dir(base_output, _system_tag(config))
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = None
    run_dir = _broadcast_run_dir(run_dir)

    # Update both the ConfigSection and the underlying ConfigParser so that
    # the saved configuration.ini reflects the actual per-run output path.
    config.paths.output = run_dir
    config.config["paths"]["output"] = run_dir

    # Only rank 0 writes to the log file; other ranks use stdout only.
    if rank == 0:
        logging_file_handler = logging.FileHandler(
            os.path.join(config.paths.output, "hgan.log")
        )
        logging_file_handler.setLevel(logging.NOTSET)
        logger.addHandler(logging_file_handler)

    experiment = Experiment(config)
    experiment.train()


if __name__ == "__main__":
    # Default stream handler so per-rank logs go to stdout (torchrun aggregates).
    import sys

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main(*sys.argv[1:])
