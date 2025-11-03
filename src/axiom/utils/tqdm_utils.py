from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger


class TqdmToLogger(io.StringIO):
    """
    File-like object that redirects tqdm's output to a logger.
    """

    def __init__(self, logger: Logger, level: int = logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level
        self.buf = ""

    def write(self, buf: str) -> int:
        self.buf = buf.strip("\r\n\t ")
        return len(buf)

    def flush(self) -> None:
        if self.buf:
            self.logger.log(self.level, self.buf)
        self.buf = ""
