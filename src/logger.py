import logging
import logging.config
import yaml
from pathlib import Path

_LOGGING_CONFIGURED = False


def setup_logging(config_path: str = "config/logging.yaml"):
    """Load YAML logging config once."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    # ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    cfg = Path(config_path)
    if cfg.exists():
        with cfg.open("rt") as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        # console only fallback (NO FILE HANDLER)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    _LOGGING_CONFIGURED = True


def get_logger(module_file: str):
    """
    Per-module logger:
    logs/<module>.log
    """
    if not _LOGGING_CONFIGURED:
        setup_logging()

    module_name = Path(module_file).stem
    logger = logging.getLogger(module_name)

    if not logger.handlers:
        file_path = Path("logs") / f"{module_name}.log"

        handler = logging.FileHandler(file_path, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # critical: stops root console handler from duplicating
        logger.propagate = False

    return logger
