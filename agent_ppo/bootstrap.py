#!/usr/bin/env python3
"""
Bootstrap project-local imports when Kaiwu loads agent modules by dotted path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_project_root_on_path() -> Path:
    bootstrap_path = Path(__file__)
    candidates = [
        bootstrap_path.absolute().parents[1],
        Path.cwd().absolute(),
        Path(os.environ.get("PWD", "")).absolute() if os.environ.get("PWD") else None,
        bootstrap_path.resolve().parents[1],
    ]

    for base in (Path.cwd().absolute(), Path(os.environ.get("PWD", ".")).absolute()):
        candidates.extend(base.parents)

    project_root = next(
        (
            candidate
            for candidate in candidates
            if candidate is not None
            and (candidate / "utils" / "training_recorder.py").is_file()
            and (candidate / "agent_ppo").exists()
        ),
        bootstrap_path.absolute().parents[1],
    )

    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)
    return project_root
