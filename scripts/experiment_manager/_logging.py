"""Thread-safe coloured logger for the experiment manager."""

from __future__ import annotations

import sys
import threading
from datetime import datetime
from pathlib import Path


class Logger:
    """Thread-safe logger with coloured console output and file persistence."""

    _LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARN": 30,
        "ERROR": 40,
    }

    _COLORS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",   # green
        "WARN": "\033[33m",   # yellow
        "ERROR": "\033[31m",  # red
    }
    _RESET = "\033[0m"

    def __init__(self, path: Path, level: str = "INFO") -> None:
        self.path = path
        self._lock = threading.Lock()
        level_upper = str(level).upper()
        if level_upper not in self._LEVELS:
            level_upper = "INFO"
        self.level = level_upper

    def _enabled(self, level: str) -> bool:
        return self._LEVELS[level] >= self._LEVELS[self.level]

    def _console_line(self, level: str, plain_line: str) -> str:
        if not sys.stdout.isatty():
            return plain_line
        color = self._COLORS[level]
        return f"{color}{plain_line}{self._RESET}"

    def _log(self, level: str, msg: str) -> None:
        if not self._enabled(level):
            return
        plain_line = f"[{self._ts()}] [{level}] {msg}"
        console_line = self._console_line(level, plain_line)
        with self._lock:
            print(console_line)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(plain_line + "\n")

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def info(self, msg: str) -> None:
        self._log("INFO", msg)

    def warn(self, msg: str) -> None:
        self._log("WARN", msg)

    def error(self, msg: str) -> None:
        self._log("ERROR", msg)

    def debug(self, msg: str) -> None:
        self._log("DEBUG", msg)
