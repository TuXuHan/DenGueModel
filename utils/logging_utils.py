import logging
from pathlib import Path


def setup_logging(log_dir: Path, timestr: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"{timestr}.log"
    logging.basicConfig(
        level=logging.INFO,
        filename=log_filename,
        filemode="w",
        format="[%(asctime)s %(levelname)-8s] %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )
    return log_filename
