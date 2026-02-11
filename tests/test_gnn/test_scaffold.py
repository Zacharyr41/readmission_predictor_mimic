"""Scaffold tests for the src.gnn package.

Verifies that all submodules are importable, exist on disk, and have docstrings.
"""

import importlib
from pathlib import Path

import pytest

SUBMODULES = [
    "graph_export",
    "embeddings",
    "view_adapter",
    "sampling",
    "transformer",
    "diffusion",
    "model",
    "losses",
    "train",
    "evaluate",
    "experiments",
]


def test_gnn_package_importable():
    """The top-level src.gnn package is importable."""
    mod = importlib.import_module("src.gnn")
    assert mod is not None


def test_gnn_package_has_docstring():
    """The top-level src.gnn package has a non-empty docstring."""
    mod = importlib.import_module("src.gnn")
    assert mod.__doc__ and mod.__doc__.strip()


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_importable(name):
    """Each submodule under src.gnn is importable."""
    mod = importlib.import_module(f"src.gnn.{name}")
    assert mod is not None


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_file_exists(name):
    """Each submodule has a corresponding .py file on disk."""
    gnn_dir = Path(__file__).resolve().parents[2] / "src" / "gnn"
    assert (gnn_dir / f"{name}.py").is_file()


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_has_docstring(name):
    """Each submodule has a non-empty module-level docstring."""
    mod = importlib.import_module(f"src.gnn.{name}")
    assert mod.__doc__ and mod.__doc__.strip()
