import datetime
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from app.core.config import settings


class ColoredFormatter(logging.Formatter):
    """Terminal formatter adding ANSI color escapes to log levels."""

    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        orig_levelname = record.levelname
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{orig_levelname:<8}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = orig_levelname


class JSONFormatter(logging.Formatter):
    """Formatter that outputs structured logs as JSON objects (JSON Lines)."""

    STANDARD_LOG_ATTRS = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.UTC
            ).isoformat(),
            "level": logging.getLevelName(record.levelno),
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }

        # Include exception traceback if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Include stack info if present
        if record.stack_info:
            log_record["stack_info"] = self.formatStack(record.stack_info)

        # Capture any custom fields passed via extra={...}
        extra_fields = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self.STANDARD_LOG_ATTRS and not k.startswith("_")
        }
        if extra_fields:
            log_record["extra"] = extra_fields

        return json.dumps(log_record, default=str, ensure_ascii=False)


def setup_logger(
    name: str = "research_agent",
    log_level: str | int | None = None,
    log_dir: str | Path | None = None,
    log_filename: str = "app.log",
    json_filename: str = "app.json",
    enable_file_logging: bool = True,
    enable_json_logging: bool = True,
) -> logging.Logger:
    """Configure and return a structured logger with console, plain text, and JSON file handlers.

    Args:
        name: Logger name (e.g. 'research_agent' or __name__).
        log_level: Logging level (e.g. 'DEBUG', 'INFO'). Defaults to settings.log_level or INFO.
        log_dir: Directory where log files are stored. Defaults to settings.log_dir.
        log_filename: Standard plain-text log filename.
        json_filename: JSON formatted log filename.
        enable_file_logging: Whether to attach rotating plain-text file handler.
        enable_json_logging: Whether to attach rotating JSON file handler.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Determine log level from parameters or settings
    if log_level is None:
        level_str = (settings.log_level or "INFO").upper()
        log_level = getattr(logging, level_str, logging.INFO)

    logger.setLevel(log_level)
    logger.propagate = False

    # Prevent duplicate handlers if setup_logger is called repeatedly
    if logger.handlers:
        return logger

    # 1. Console Handler with color formatting (directed to stderr to keep stdout clean for MCP stdio)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_format = ColoredFormatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    log_path = Path(log_dir or settings.log_dir)

    # 2. Rotating Plain-text File Handler
    if enable_file_logging:
        try:
            log_path.mkdir(parents=True, exist_ok=True)
            file_target = log_path / log_filename

            file_handler = RotatingFileHandler(
                file_target,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(log_level)
            file_format = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        except Exception:
            pass

    # 3. Rotating JSON File Handler
    if enable_json_logging:
        try:
            log_path.mkdir(parents=True, exist_ok=True)
            json_target = log_path / json_filename

            json_handler = RotatingFileHandler(
                json_target,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
            )
            json_handler.setLevel(log_level)
            json_handler.setFormatter(JSONFormatter())
            logger.addHandler(json_handler)
        except Exception:
            pass

    return logger


def get_logger(name: str = "research_agent") -> logging.Logger:
    """Retrieve an existing logger or create one with default configuration."""
    existing_logger = logging.getLogger(name)
    if not existing_logger.handlers:
        return setup_logger(name=name)
    return existing_logger


# Default application logger instance
logger = get_logger("research_agent")
