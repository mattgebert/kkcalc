"""
Tests for drag-and-drop file import in `kk_object_list` (the "Import Data" equivalent).

Covers:
- `dragEnterEvent`/`dropEvent` correctly extract local file paths from a drop, ignoring
  non-local URLs and paths that don't exist.
- `import_data_files` imports multiple files sequentially, and if the user cancels a file's
  import, prompts whether to continue with the remaining files or cancel the rest.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PyQt6 import QtCore, QtWidgets

from kkcalc2.gui.asf_loader import kk_object_list


class FakeDropEvent:
    """A minimal stand-in for `QtGui.QDropEvent`, carrying real `QMimeData` with file URLs."""

    def __init__(
        self, paths: list[str], has_urls: bool = True
    ) -> None:  # numpydoc ignore=GL08
        self._mime = QtCore.QMimeData()
        if has_urls:
            self._mime.setUrls([QtCore.QUrl.fromLocalFile(p) for p in paths])
        self.accepted = False

    def mimeData(self) -> QtCore.QMimeData:
        """Return the (real) `QMimeData` carried by this fake event."""
        return self._mime

    def acceptProposedAction(self) -> None:
        """Record that the drop/drag action was accepted."""
        self.accepted = True


@pytest.fixture
def object_list(qapp):
    """A bare `kk_object_list` widget, with no example data pre-loaded."""
    widget = kk_object_list()
    yield widget
    widget.deleteLater()


@pytest.fixture
def temp_files(tmp_path: Path) -> list[str]:
    """Two temporary (empty) files, standing in for real data files to be dropped."""
    paths = []
    for name in ("data1.txt", "data2.txt"):
        p = tmp_path / name
        p.write_text("280.0 0.1\n285.0 0.2\n")
        paths.append(str(p))
    return paths


class TestDragAndDropEvents:
    """Tests that drag/drop events correctly identify importable local files."""

    def test_drag_enter_accepts_local_file_urls(
        self, object_list: kk_object_list, temp_files: list[str]
    ) -> None:
        """`dragEnterEvent` accepts drags carrying local file URLs."""
        event = FakeDropEvent(temp_files)
        object_list.dragEnterEvent(event)
        assert event.accepted

    def test_drag_enter_ignores_non_local_urls(
        self, object_list: kk_object_list
    ) -> None:
        """`dragEnterEvent` does not accept drags with only non-local (e.g. web) URLs."""
        event = FakeDropEvent([])
        event._mime.setUrls([QtCore.QUrl("https://example.com/data.txt")])
        object_list.dragEnterEvent(event)
        assert not event.accepted

    def test_drop_event_imports_only_existing_local_files(
        self, object_list: kk_object_list, temp_files: list[str], monkeypatch
    ) -> None:
        """`dropEvent` filters to existing local files, and delegates to `import_data_files`."""
        missing_file = str(Path(tempfile.gettempdir()) / "does_not_exist_kkcalc.txt")
        event = FakeDropEvent([*temp_files, missing_file])

        recorded_paths = []
        monkeypatch.setattr(
            object_list, "import_data_files", lambda paths: recorded_paths.extend(paths)
        )

        object_list.dropEvent(event)

        assert event.accepted
        assert [str(Path(p)) for p in recorded_paths] == [
            str(Path(p)) for p in temp_files
        ]

    def test_drop_event_with_no_files_does_not_call_import(
        self, object_list: kk_object_list, monkeypatch
    ) -> None:
        """`dropEvent` does nothing further if no valid local files were dropped."""
        event = FakeDropEvent([])
        called = MagicMock()
        monkeypatch.setattr(object_list, "import_data_files", called)

        object_list.dropEvent(event)

        called.assert_not_called()


class TestImportDataFiles:
    """Tests sequential import of multiple files via `import_data_files`."""

    def test_all_files_imported_when_none_cancelled(
        self, object_list: kk_object_list, temp_files: list[str], monkeypatch
    ) -> None:
        """If every file imports successfully, all are processed with no prompts shown."""
        import_calls = []
        monkeypatch.setattr(
            object_list,
            "import_data",
            lambda path=None: import_calls.append(path) or True,
        )
        question = MagicMock()
        monkeypatch.setattr(QtWidgets.QMessageBox, "question", question)

        object_list.import_data_files(temp_files)

        assert import_calls == temp_files
        question.assert_not_called()

    def test_cancelling_prompts_to_continue_and_continues_on_yes(
        self, object_list: kk_object_list, temp_files: list[str], monkeypatch
    ) -> None:
        """Cancelling the first file's import prompts the user; choosing 'Yes' continues."""
        import_calls = []
        # First file is "cancelled" (returns False), second succeeds.
        monkeypatch.setattr(
            object_list,
            "import_data",
            lambda path=None: import_calls.append(path) or path == temp_files[1],
        )
        question = MagicMock(return_value=QtWidgets.QMessageBox.StandardButton.Yes)
        monkeypatch.setattr(QtWidgets.QMessageBox, "question", question)

        object_list.import_data_files(temp_files)

        assert import_calls == temp_files  # Both files were attempted.
        question.assert_called_once()

    def test_cancelling_prompts_to_continue_and_stops_on_no(
        self, object_list: kk_object_list, temp_files: list[str], monkeypatch
    ) -> None:
        """Cancelling the first file's import prompts the user; choosing 'No' stops the rest."""
        import_calls = []
        monkeypatch.setattr(
            object_list,
            "import_data",
            lambda path=None: import_calls.append(path) or False,
        )
        question = MagicMock(return_value=QtWidgets.QMessageBox.StandardButton.No)
        monkeypatch.setattr(QtWidgets.QMessageBox, "question", question)

        object_list.import_data_files(temp_files)

        assert import_calls == [temp_files[0]]  # Only the first file was attempted.
        question.assert_called_once()

    def test_no_prompt_when_last_file_is_cancelled(
        self, object_list: kk_object_list, temp_files: list[str], monkeypatch
    ) -> None:
        """No prompt is shown if the cancelled file is the last one (nothing left to continue)."""
        monkeypatch.setattr(object_list, "import_data", lambda path=None: False)
        question = MagicMock()
        monkeypatch.setattr(QtWidgets.QMessageBox, "question", question)

        object_list.import_data_files([temp_files[0]])

        question.assert_not_called()
