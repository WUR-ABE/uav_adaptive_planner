from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from logging import WARN, root, warning
from typing import TYPE_CHECKING, no_type_check
import warnings

from colorlog import ColoredFormatter, StreamHandler, getLogger
from tqdm import TqdmExperimentalWarning
from tqdm.std import tqdm as std_tqdm

if TYPE_CHECKING:
    from logging import Handler, Logger
    from typing import Any, Generator


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

__version__ = "1.0.0"


@no_type_check
def require_module(module_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                __import__(module_name)
            except ImportError:
                warning(f"The package '{module_name}' is not installed!")
                return None
            return func(*args, **kwargs)

        return wrapper

    return decorator


def setup_logging(name: str, level: int | str = "INFO") -> Logger:
    handler = StreamHandler()
    handler.setFormatter(ColoredFormatter("%(log_color)s%(message)s"))

    log = getLogger(name)
    log.addHandler(handler)
    log.setLevel(level)

    # RasterIO keeps complaining about virtual file systems, just shut it up
    getLogger("rasterio._filepath").setLevel(WARN)

    return log


def _is_console_logging_handler(handler: Handler) -> bool:
    from logging import StreamHandler as StdStreamHandler
    import sys

    if isinstance(handler, StdStreamHandler):
        return handler.stream in {sys.stdout, sys.stderr}

    if isinstance(handler, StreamHandler):
        return True

    return False


@contextmanager
def logging_redirect_tqdm(
    loggers: list[Logger] | None = None,
    tqdm_class: type[std_tqdm] = std_tqdm,  # type: ignore[type-arg]
) -> Generator[Any, Any, Any]:
    from tqdm.contrib.logging import _TqdmLoggingHandler  # type: ignore[attr-defined]

    if loggers is None:
        loggers = [root]
    original_handlers_list = [logger.handlers for logger in loggers]
    try:
        for logger in loggers:
            tqdm_handler = _TqdmLoggingHandler(tqdm_class)

            orig_handler = None
            for h in logger.handlers:
                if isinstance(h, StreamHandler):
                    orig_handler = h
                    break

            if orig_handler is not None:
                tqdm_handler.setFormatter(orig_handler.formatter)
                tqdm_handler.stream = orig_handler.stream
            logger.handlers = [handler for handler in logger.handlers if not _is_console_logging_handler(handler)] + [tqdm_handler]
        yield
    finally:
        for logger, original_handlers in zip(loggers, original_handlers_list):
            logger.handlers = original_handlers
