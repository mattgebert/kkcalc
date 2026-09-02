"""
Shared pytest fixtures for GUI tests.

GUI tests require the optional `PyQt6` dependency (the `gui` dependency group); tests in this
package are skipped entirely if it is not installed. Tests run headlessly using Qt's "offscreen"
platform plugin, so no display server is required.
"""

import os

import pytest

# Force the offscreen platform plugin before any Qt import, so tests can run without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PyQt6 = pytest.importorskip("PyQt6", reason="PyQt6 is required for GUI tests.")
from PyQt6 import QtWidgets  # noqa: E402 RUF100 - Import after setting QT_QPA_PLATFORM.


@pytest.fixture(scope="session")
def qapp() -> QtWidgets.QApplication:
    """
    Session-scoped `QApplication` instance, required to construct any Qt widgets.

    Returns
    -------
    QtWidgets.QApplication
        The singleton `QApplication` instance for the test session.
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app
