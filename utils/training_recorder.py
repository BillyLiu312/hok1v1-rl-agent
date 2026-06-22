#!/usr/bin/env python3
"""
Append-only JSONL recorder for training diagnostics.

The recorder is intentionally best-effort: failures are reported through the
framework logger when available, but never interrupt training.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from agent_ppo.conf.runtime_config import runtime_value


DEFAULT_RECORD_DIR = "logs/run_records"
DISABLE_VALUES = {"0", "false", "False", "no", "NO", "off", "OFF"}


class TrainingRecorder:
    def __init__(self, logger=None, record_dir: str | None = None, run_id: str | None = None):
        self.logger = logger
        self.enabled = runtime_value("HOK_TRAINING_RECORDER", "1") not in DISABLE_VALUES
        self.record_dir = Path(record_dir or runtime_value("HOK_TRAINING_RECORD_DIR", DEFAULT_RECORD_DIR))
        self.run_id = run_id or runtime_value("HOK_TRAINING_RUN_ID") or self._make_run_id()
        os.environ.setdefault("HOK_TRAINING_RUN_ID", self.run_id)
        self.pid = os.getpid()
        self._warned = False

    def record_config_snapshot(self, name: str, paths: list[str], extra: dict[str, Any] | None = None):
        files = []
        for raw_path in paths:
            path = Path(raw_path)
            item = {"path": raw_path, "exists": path.exists()}
            if path.exists() and path.is_file():
                try:
                    data = path.read_bytes()
                    item.update(
                        {
                            "size": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "text": data.decode("utf-8", errors="replace"),
                        }
                    )
                except Exception as exc:
                    item["read_error"] = repr(exc)
            files.append(item)

        self.record(
            "config",
            {
                "name": name,
                "files": files,
                "extra": extra or {},
            },
        )

    def record(self, stream: str, payload: dict[str, Any]):
        if not self.enabled:
            return

        event = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "time_unix": round(time.time(), 3),
            "pid": self.pid,
            "run_id": self.run_id,
            "stream": stream,
            "payload": self._to_jsonable(payload),
        }

        try:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            path = self.record_dir / f"{stream}-{self.run_id}-{self.pid}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        except Exception as exc:  # pragma: no cover - keep training alive on platform IO issues
            if self.logger and not self._warned:
                self.logger.warning(f"training recorder disabled after write failure: {exc}")
            self._warned = True
            self.enabled = False

    def _make_run_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S")

    def _to_jsonable(self, value: Any):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if hasattr(value, "tolist"):
            try:
                return self._to_jsonable(value.tolist())
            except Exception:
                pass
        return repr(value)
