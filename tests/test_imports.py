"""Smoke tests: every daemon module must import (catches stale paths)."""

import importlib

MODULES = [
    "oi_agent.cli",
    "oi_agent.config",
    "oi_agent.store",
    "oi_agent.poster",
    "oi_agent.manifest.generator",
    "oi_agent.watch.gate",
    "oi_agent.watch.discord_client",
    "oi_agent.agent.llm",
    "oi_agent.agent.tools",
    "oi_agent.agent.responder",
]


def test_all_modules_import():
    for name in MODULES:
        assert importlib.import_module(name) is not None
