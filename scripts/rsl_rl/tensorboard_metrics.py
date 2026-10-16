"""Filter scalar writes without touching rollout data, curricula or checkpoints."""

import re


class ScalarFilterWriter:
    def __init__(self, writer, pattern):
        self._writer = writer
        self._pattern = re.compile(pattern)

    def add_scalar(self, tag, *args, **kwargs):
        if self._pattern.fullmatch(tag):
            return self._writer.add_scalar(tag, *args, **kwargs)
        return None

    def __getattr__(self, name):
        return getattr(self._writer, name)
