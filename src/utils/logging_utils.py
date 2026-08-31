import logging
import sys

from src.utils.error_handling import suppress_known_runtime_warnings


def setup_logging(level: int = logging.INFO) -> None:
    suppress_known_runtime_warnings()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
