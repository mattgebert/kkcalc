"""
Tests to check the modification and loading of ASF files into the GUI.
"""

import os

import pytest

# Force the offscreen platform plugin before any Qt import, so tests can run without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PyQt6 = pytest.importorskip("PyQt6", reason="PyQt6 is required for GUI tests.")


# First test the database dialog and the addition.
