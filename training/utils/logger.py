"""MedicalAI Project Logging Utility.

This module provides a reusable :class:`~Logger` class for consistent console and
file logging across the MedicalAI training/evaluation/prediction pipeline.

Key features
- Console logging
- File logging (one log file per session)
- Automatic log directory creation
- Timestamped session log filename: YYYY-MM-DD_HH-MM-SS.log
- Colored console output (best-effort; falls back gracefully)

Only standard library modules are used.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    """Reusable logger wrapper for the MedicalAI project.

    The class internally uses Python's :mod:`logging` module, which is already
    thread-safe.
    """

    _LEVEL_TO_COLOR = {
        logging.DEBUG: "\033[90m",  # bright black / gray
        logging.INFO: "\033[94m",  # bright blue
        logging.WARNING: "\033[93m",  # bright yellow
        logging.ERROR: "\033[91m",  # bright red
        logging.CRITICAL: "\033[91m",
    }

    _RESET = "\033[0m"

    def __init__(
        self,
        name: str = "MedicalAI",
        log_dir: str | Path = Path("MedicalAI") / "training" / "logs",
        level: int = logging.INFO,
        enable_console: bool = True,
        enable_file: bool = True,
        use_color: bool = True,
        session_time: Optional[datetime] = None,
    ) -> None:
        """Create a logger.

        Args:
            name: Logger name.
            log_dir: Directory where log files will be written.
            level: Minimum log level.
            enable_console: Whether to log to stdout.
            enable_file: Whether to log to a session file.
            use_color: Whether to attempt colored console output.
            session_time: Optional datetime used to form the session filename.
                Mainly for deterministic self-testing.
        """

        self.name = name
        self.level = int(level)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        session_dt = session_time or datetime.now()
        filename = session_dt.strftime("%Y-%m-%d_%H-%M-%S.log")
        self.log_path = self.log_dir / filename

        self._logger = logging.getLogger(name)
        self._logger.setLevel(self.level)
        self._logger.propagate = False

        # Avoid duplicate handlers if Logger is instantiated multiple times.
        # We only consider our own handlers as duplicates.
        handlers_already_present = any(
            getattr(h, "_medicalai_logger_file", False) or getattr(h, "_medicalai_logger_console", False)
            for h in self._logger.handlers
        )

        if not handlers_already_present:
            self._configure_handlers(
                enable_console=enable_console,
                enable_file=enable_file,
                use_color=use_color,
            )

    def _configure_handlers(self, enable_console: bool, enable_file: bool, use_color: bool) -> None:
        fmt = "%(asctime)s [%(levelname)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        if enable_file:
            file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
            file_handler.setLevel(self.level)
            file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            setattr(file_handler, "_medicalai_logger_file", True)
            self._logger.addHandler(file_handler)

        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)

            if use_color:
                # Best-effort colored formatter. If unsupported, it will still work
                # because ANSI codes are only embedded into the message.
                console_handler.setFormatter(_ColorizingFormatter(fmt=fmt, datefmt=datefmt, level_to_color=self._LEVEL_TO_COLOR))
            else:
                console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

            setattr(console_handler, "_medicalai_logger_console", True)
            self._logger.addHandler(console_handler)

    def info(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log an INFO level message."""

        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log a WARNING level message."""

        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log an ERROR level message."""

        self._logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log a DEBUG level message."""

        self._logger.debug(msg, *args, **kwargs)

    def exception(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log an ERROR level message with exception traceback.

        Intended to be called from an exception handler.
        """

        self._logger.exception(msg, *args, **kwargs)

    def close(self) -> None:
        """Flush and close all handlers owned by this logger."""

        for handler in list(self._logger.handlers):
            try:
                handler.flush()
            except Exception:
                # Best-effort; closing still proceeds.
                pass
            try:
                handler.close()
            except Exception:
                pass
            try:
                self._logger.removeHandler(handler)
            except Exception:
                pass


class _ColorizingFormatter(logging.Formatter):
    """Formatter that colorizes log level names/messages for console output."""

    def __init__(
        self,
        fmt: str,
        datefmt: Optional[str],
        level_to_color: dict[int, str],
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._level_to_color = level_to_color

    def format(self, record: logging.LogRecord) -> str:
        # Build the base message first.
        base = super().format(record)

        color_prefix = self._level_to_color.get(record.levelno)
        if not color_prefix:
            return base

        # Insert ANSI color codes without trying to parse the format string.
        # We color the entire line for simplicity.
        return f"{color_prefix}{base}{Logger._RESET}"


if __name__ == "__main__":
    # Self-test: write one message for each level and verify log file creation.
    test_log_dir = Path("MedicalAI") / "training" / "logs" / "_self_test"

    logger = Logger(
        name="MedicalAI.self_test",
        log_dir=test_log_dir,
        level=logging.DEBUG,
        enable_console=True,
        enable_file=True,
        use_color=True,
    )

    try:
        logger.debug("Self-test DEBUG message")
        logger.info("Self-test INFO message")
        logger.warning("Self-test WARNING message")
        logger.error("Self-test ERROR message")
        try:
            raise RuntimeError("Self-test exception")
        except RuntimeError:
            logger.exception("Self-test EXCEPTION message")

        # Verify log file exists.
        log_path = logger.log_path
        exists = log_path.exists() and log_path.is_file()
        print(f"Self-test log file: {log_path}")
        print(f"Self-test status: {'OK' if exists else 'FAILED'}")
    finally:
        logger.close()

