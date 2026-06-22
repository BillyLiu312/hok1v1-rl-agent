#!/usr/bin/env python3
"""Compatibility wrapper for the training recorder.

The runtime implementation lives in agent_ppo.conf.training_recorder so Kaiwu
training can import it without relying on project-level utils being visible.
"""

from agent_ppo.conf.training_recorder import DEFAULT_RECORD_DIR, DISABLE_VALUES, TrainingRecorder


__all__ = ["DEFAULT_RECORD_DIR", "DISABLE_VALUES", "TrainingRecorder"]
