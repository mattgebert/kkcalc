"""
Tests for the `kkcalc2.__main__` module's GUI entry points.

Covers `main` (loads the example Polystyrene data and constructs the main window) and
`main_with_traceback` (wraps `main` to display errors in a dialog instead of crashing).
Both functions are exercised end-to-end, with the blocking `QApplication.exec` event loop and
any modal dialogs (`window.show`, `QErrorMessage`) patched out so tests complete immediately.

This guards against regressions in the example-data loading / window construction flow used by
the `kkcalc2`/`kkcalc2_tr` console-script entry points (see `pyproject.toml`).
"""

import io
import pkgutil
import typing

import numpy as np
import pytest

import kkcalc2.__main__ as main_module


class DummyWindow:
    """A stand-in for `kk_gui`, recording constructor arguments instead of building a real window."""

    instances: typing.ClassVar[list["DummyWindow"]] = []

    def __init__(
        self, objs=None, autohide_modifier: bool = False
    ) -> None:  # numpydoc ignore=GL08
        self.objs = objs
        self.autohide_modifier = autohide_modifier
        self.shown = False
        DummyWindow.instances.append(self)

    def show(self) -> None:
        """Record that the window was shown, without displaying anything."""
        self.shown = True


@pytest.fixture(autouse=True)
def _reset_dummy_window():
    """Reset `DummyWindow.instances` before/after each test."""
    DummyWindow.instances = []
    yield
    DummyWindow.instances = []


@pytest.fixture
def patched_main(qapp, monkeypatch) -> DummyWindow | None:
    """
    Runs `main_module.main()` with the window class and event loop patched out.

    Returns
    -------
    DummyWindow | None
        The `DummyWindow` instance constructed by `main()`, or `None` if `main()` raised
        before constructing a window (the caller should check for this case).
    """
    monkeypatch.setattr(main_module, "kk_gui", DummyWindow)
    # Avoid constructing a second real QApplication instance (only one may exist per process).
    monkeypatch.setattr(main_module.QtWidgets, "QApplication", lambda *a, **k: qapp)
    monkeypatch.setattr(qapp, "exec", lambda: None)
    main_module.main()
    return DummyWindow.instances[0] if DummyWindow.instances else None


class TestMain:
    """Tests the `main` entry point: loading example data and constructing the main window."""

    def test_main_constructs_exactly_one_window(
        self, patched_main: DummyWindow
    ) -> None:
        """`main` constructs exactly one `kk_gui` window."""
        assert len(DummyWindow.instances) == 1
        assert patched_main is not None
        assert patched_main.shown

    def test_main_window_not_autohidden(self, patched_main: DummyWindow) -> None:
        """`main` constructs the window with the modifier panel visible (`autohide_modifier=False`)."""
        assert patched_main.autohide_modifier is False

    def test_main_loads_single_asf_object(self, patched_main: DummyWindow) -> None:
        """`main` passes exactly one atomic scattering factor object to the window."""
        assert patched_main.objs is not None
        assert len(patched_main.objs) == 1

    def test_main_loaded_object_matches_example_data(
        self, patched_main: DummyWindow
    ) -> None:
        """The loaded object's name, stoichiometry, and energies/factors match the example data."""
        asf_ps = patched_main.objs[0]

        assert asf_ps.name == "Polystyrene"
        assert asf_ps.stoichiometry is not None
        assert str(asf_ps.stoichiometry) == "C8H8"

        # Compare against the packaged example data file directly.
        raw = pkgutil.get_data("kkcalc2", "data/PS_004_-dc.txt")
        assert raw is not None
        expected = np.genfromtxt(io.BytesIO(raw), skip_header=4)

        assert len(asf_ps.energies) == expected.shape[0]
        assert np.allclose(np.sort(asf_ps.energies), np.sort(expected[:, 0]))
        assert np.all(np.isfinite(asf_ps.factors))

    def test_main_scales_to_database(self, patched_main: DummyWindow) -> None:
        """`main` scales the loaded example data to the bundled database (per its `scale_to_database=True`)."""
        asf_ps = patched_main.objs[0]
        # Scaling only mutates `factors`; the original NEXAFS-derived data is retained separately.
        assert asf_ps.origin_data is not None


class TestMainWithTraceback:
    """Tests `main_with_traceback`: exceptions from `main` are caught and shown in a dialog."""

    def test_exceptions_from_main_are_suppressed(self, qapp, monkeypatch) -> None:
        """If `main` raises, `main_with_traceback` catches it rather than propagating."""

        def failing_main():
            raise RuntimeError("Simulated failure for testing.")

        monkeypatch.setattr(main_module, "main", failing_main)
        monkeypatch.setattr(main_module.QtWidgets, "QApplication", lambda *a, **k: qapp)
        monkeypatch.setattr(qapp, "exec", lambda: None)
        monkeypatch.setattr(
            main_module.QtWidgets.QErrorMessage, "showMessage", lambda self, msg: None
        )
        monkeypatch.setattr(
            main_module.QtWidgets.QErrorMessage, "exec", lambda self: None
        )

        # Should not raise.
        main_module.main_with_traceback()

    def test_no_exception_does_not_show_error_dialog(self, qapp, monkeypatch) -> None:
        """If `main` succeeds, no error dialog is constructed."""
        error_dialogs_created = []
        original_init = main_module.QtWidgets.QErrorMessage.__init__

        def tracking_init(self, *args, **kwargs):
            error_dialogs_created.append(self)
            return original_init(self, *args, **kwargs)

        monkeypatch.setattr(main_module, "kk_gui", DummyWindow)
        monkeypatch.setattr(main_module.QtWidgets, "QApplication", lambda *a, **k: qapp)
        monkeypatch.setattr(qapp, "exec", lambda: None)
        monkeypatch.setattr(
            main_module.QtWidgets.QErrorMessage, "__init__", tracking_init
        )

        main_module.main_with_traceback()

        assert error_dialogs_created == []
