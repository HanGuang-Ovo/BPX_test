"""部署日志和调试录制共用的 JSONL 写入器；不依赖 SDK、窗口或回放。"""

from contextlib import AbstractContextManager
import json
from pathlib import Path


class JsonlWriter(AbstractContextManager):
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.metadata = metadata
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        self.write({"type": "metadata", "schema": 1, **self.metadata})
        return self

    def write(self, value):
        self.stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def __exit__(self, *args):
        if self.stream:
            try:
                exc = args[1] if len(args) > 1 else None
                self.write(
                    {"type": "end", "status": "error" if exc else "completed", "error": str(exc) if exc else None}
                )
            finally:
                self.stream.close()
