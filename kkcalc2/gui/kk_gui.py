"""
GUI interface for the Kramers-Kronig Calculator.

A simple interface that allows the user to input data, assign to real
or imaginary components, and perform a Kramers-Kronig transform on it.
The interface is built using the PyQt6 library.
"""

import importlib.util
import io
import os
import pkgutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector
from PyQt6 import QtCore, QtGui, QtWidgets

from kkcalc2 import asf_database
from kkcalc2.gui import session
from kkcalc2.gui.asf_loader import kk_object_list
from kkcalc2.gui.asf_modifier import kk_object_modifier
from kkcalc2.gui.asf_viewer import GraphType, asf_viewer
from kkcalc2.models import (
    asf_abstract,
    asf_im,
    asf_re,
    asp_abstract,
    asp_im,
    asp_re,
)
from kkcalc2.stoich import stoichiometry


class kk_gui(QtWidgets.QWidget):
    """
    Widget for the Kramers-Kronig Calculator.

    The widget contains a viewer, a list of objects, and a modifier for the objects.

    Parameters
    ----------
    parent : QtWidgets.QWidget | None
        The parent widget.
    objs : list[asf_abstract | asp_abstract] | None
        A list of initial objects to load into the GUI.
    autohide_modifier : bool
        Whether to autohide the modifier when no object is selected.

    Attributes
    ----------
    obj_list : kk_object_list
        The list and selector of objects.
    obj_modifier : kk_object_modifier
        The modifier (i.e. of density information or stochiometry) for the objects.
    viewer : asf_viewer
        The graphical viewer for the objects.
    """

    def __init__(
        self,
        parent=None,
        objs: list[asf_abstract | asp_abstract] | None = None,
        autohide_modifier: bool = False,
    ) -> None:  # numpydoc ignore=GL08
        super().__init__(parent=parent)
        self.setWindowTitle("Kramers-Kronig Calculator")

        windowIconPath = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "docs",
            "source",
            "_static",
            "logo2.png",
        )
        if os.path.exists(windowIconPath):
            self.setWindowIcon(QtGui.QIcon(windowIconPath))
        else:
            try:
                data = pkgutil.get_data("kkcalc", "logo2.png")
                if data is not None:
                    self.setWindowIcon(QtGui.QIcon(io.BytesIO(data)))
            except FileNotFoundError:  # as e:
                pass

        self._layout = QtWidgets.QHBoxLayout()
        self.setLayout(self._layout)
        self._draggable = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._layout.addWidget(self._draggable)

        # Setup margins if parent is provided.
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._layout.setContentsMargins(0, 0, 0, 0)

        # Create the viewer and the UI elements
        self.obj_list = obj_list = kk_object_list(objs=objs)
        self.obj_modifier = obj_modifier = kk_object_modifier()
        self.viewer = viewer = asf_viewer()

        # Add elements to the layout.
        # self._layout.addWidget(obj_list)
        # self._layout.addWidget(object_modifier)
        # self._layout.addWidget(viewer)
        self._draggable.addWidget(obj_list)
        self._draggable.addWidget(obj_modifier)
        self._draggable.addWidget(viewer)
        # self.obj_modifier.setFixedWidth(250)
        self._draggable.setStretchFactor(0, 10)
        self._draggable.setStretchFactor(1, 1)
        self._draggable.setStretchFactor(2, 10)

        # Modify element properties
        obj_list.setMinimumWidth(240)
        self._autohide_modifier = autohide_modifier
        if autohide_modifier:
            obj_modifier.hide()
        self._draggable.setHandleWidth(10)

        # Initialize an internal var for plotting extension handles
        self._has_handle = False
        """Tracks if the viewer has a handle for extending the domain."""
        self._handle: SpanSelector | None = None
        """The handle for extending the domain."""

        # Connect the viewer to the object list
        obj_list.viewSelectionChanged.connect(self.on_view_change)
        obj_list.selectedObjectChanged.connect(self.on_object_select)
        obj_modifier.objectModified.connect(self.on_object_modify)
        obj_modifier.objectCreated.connect(self.on_object_create)
        obj_modifier.hasHandle.connect(self.on_has_handle)

        # Setup the viewer
        self.on_view_change()

        self._setup_menu()

    def _setup_menu(self) -> None:
        """Create the File and Database menus."""
        self.menu_bar = QtWidgets.QMenuBar(self)
        self._layout.setMenuBar(self.menu_bar)

        file_menu = self.menu_bar.addMenu("&File")
        self.open_action = file_menu.addAction("&Open...")
        self.open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_files)
        file_menu.addSeparator()
        self.save_session_action = file_menu.addAction("&Save Session...")
        self.save_session_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_session_action.triggered.connect(self.save_session)
        self.open_session_action = file_menu.addAction("Open &Previous Session...")
        self.open_session_action.triggered.connect(self.open_session)
        file_menu.addSeparator()
        self.export_csv_action = file_menu.addAction("Save Data to &CSV...")
        self.export_csv_action.triggered.connect(self.export_csv)
        file_menu.addSeparator()
        self.exit_action = file_menu.addAction("E&xit")
        self.exit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)

        db_menu = self.menu_bar.addMenu("&Database")
        self.db_group = QtGui.QActionGroup(self)
        self.db_group.setExclusive(True)
        self.db_henke_action = db_menu.addAction("&Henke (default)")
        self.db_periodictable_action = db_menu.addAction("&periodictable")
        for action, backend in (
            (self.db_henke_action, "kkcalc"),
            (self.db_periodictable_action, "periodictable"),
        ):
            action.setCheckable(True)
            action.setData(backend)
            self.db_group.addAction(action)
        if importlib.util.find_spec("periodictable") is None:
            self.db_periodictable_action.setEnabled(False)
            self.db_periodictable_action.setToolTip(
                "The optional 'periodictable' package is not installed."
            )
        current = asf_database.get_db_backend()
        self.db_henke_action.setChecked(current == "kkcalc")
        self.db_periodictable_action.setChecked(current == "periodictable")
        self.db_group.triggered.connect(self.on_database_choice)
        db_menu.addSeparator()
        self.add_db_action = db_menu.addAction("&Add Database Object...")
        self.add_db_action.triggered.connect(self.obj_list.add_database)

    def open_files(self) -> None:
        """Select one or more data files and import them."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open Data Files", "", "All Files (*)"
        )
        if paths:
            self.obj_list.import_data_files(paths)

    def save_session(self, path: str | None = None) -> bool:
        """
        Save all loaded objects to a session file.

        Parameters
        ----------
        path : str | None, optional
            The file to write. If not given (or when triggered by a menu action), the user is prompted.

        Returns
        -------
        bool
            Whether a session was saved.
        """
        if not isinstance(path, str):
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Session", "session.json", "Session Files (*.json)"
            )
        if not path:
            return False
        try:
            session.save_session(self.obj_list.all_objects, path)
        except (OSError, TypeError, ValueError) as e:
            QtWidgets.QMessageBox.critical(self, "Save Session Failed", str(e))
            return False
        return True

    def open_session(self, path: str | None = None) -> bool:
        """
        Load objects from a session file, replacing any currently loaded objects.

        Parameters
        ----------
        path : str | None, optional
            The file to read. If not given (or when triggered by a menu action), the user is prompted.

        Returns
        -------
        bool
            Whether a session was loaded.
        """
        if not isinstance(path, str):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Session", "", "Session Files (*.json)"
            )
        if not path:
            return False
        try:
            objs = session.load_session(path)
        except (OSError, KeyError, TypeError, ValueError) as e:
            QtWidgets.QMessageBox.critical(self, "Open Session Failed", str(e))
            return False
        if self.obj_list.all_objects:
            response = QtWidgets.QMessageBox.question(
                self,
                "Replace Current Objects?",
                "Opening a session replaces the currently loaded objects. Continue?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if response != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        self.obj_list.clear_objects()
        self.obj_list.add_kk_objs(objs)
        return True

    def export_csv(self, directory: str | None = None) -> list[str]:
        """
        Export each loaded object to its own CSV file.

        Parameters
        ----------
        directory : str | None, optional
            The output directory. If not given (or when triggered by a menu action), the user is prompted.

        Returns
        -------
        list[str]
            The paths written (empty if cancelled or there is nothing to export).
        """
        if not isinstance(directory, str):
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select CSV Output Folder"
            )
        if not directory or not self.obj_list.all_objects:
            return []
        try:
            return session.export_objects_to_csv(self.obj_list.all_objects, directory)
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Export Failed", str(e))
            return []

    def on_database_choice(self, action: QtGui.QAction) -> None:
        """
        Switch the active atomic scattering database backend.

        Parameters
        ----------
        action : QtGui.QAction
            The chosen database action, whose data holds the backend name.
        """
        backend = action.data()
        if backend == asf_database.get_db_backend():
            return
        try:
            asf_database.set_db_backend(backend)
        except (ImportError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Database Unavailable", str(e))
            current = asf_database.get_db_backend()
            self.db_henke_action.setChecked(current == "kkcalc")
            self.db_periodictable_action.setChecked(current == "periodictable")

    def on_has_handle(self, has_handle: bool):
        """
        Catch the signal from the modifier, for when an extended domain is to be created.

        Parameters
        ----------
        has_handle : bool
            Whether to create the handle or not.
        """
        if has_handle:
            # Get the current obj
            obj = self.obj_list.selected_object
            if self._has_handle:
                # Update the existing handle.
                self._handle.extents = obj.energies.min(), obj.energies.max()
            else:
                # Get the current graph style
                graph_type = self.viewer.graph_type
                # Find the appropriate axes
                handle_ax = None
                if obj is None:
                    # No object selected, do nothing.
                    return
                # Only allow database extensions for re/im objects.
                if (
                    graph_type is GraphType.RE_IM_OVERLAY
                    or graph_type is GraphType.RE_IM_SEPARATE
                ):
                    if isinstance(obj, (asf_re, asp_re)):
                        handle_ax = self.viewer.ax1
                    elif isinstance(obj, (asf_im, asp_im)):
                        handle_ax = self.viewer.ax2
                # Create the handle
                if handle_ax is not None:
                    # Define functions to convert the pixel coordinates to data coordinates
                    def pix_to_data(
                        ax: plt.Axes, x: float, y: float
                    ) -> tuple[float, float]:
                        # Convert pixel coordinates to data coordinates
                        return ax.transData.transform([x, y])
                        # return inv.transform((x, y))

                    def handle_update(x, y):
                        min_x, max_x = pix_to_data(handle_ax, x, y)
                        self.on_handle_update(min_x, max_x)

                    # Create the handle
                    self._handle = SpanSelector(
                        ax=handle_ax,
                        onselect=handle_update,
                        direction="horizontal",
                        useblit=True,
                        interactive=True,
                        drag_from_anywhere=True,
                    )
                    self._handle.extents = (obj.energies.min(), obj.energies.max())
        else:
            if self._has_handle:
                # Remove the handle
                self._handle.visible = False
                self._handle.clear()
                self._handle.ax.artists.remove(self._handle)
                # Lose track of the handle
                self._handle = None
        self.viewer.reset_graph()

    def on_handle_update(self, min_x: float, max_x: float):
        """
        Catch updates to the handle, and update the modifier values.

        Parameters
        ----------
        min_x : float
            The minimum x value of the handle.
        max_x : float
            The maximum x value of the handle.
        """
        # Update the lb and ub values
        self.obj_modifier.merge_dom_lb_edit.setText(f"{min_x:.2f}")
        self.obj_modifier.merge_dom_ub_edit.setText(f"{max_x:.2f}")

    def on_view_change(self):
        """
        Catch a change of toggling viewed objects.
        """
        objs = self.obj_list.checked_objects
        if self._has_handle:
            # TODO: Create a temporary asf object to pass to the viewer using the handle
            pass
        # Update the viewer
        self.viewer.snap_objects = self.obj_list.all_objects
        self.viewer.scattering_objects = objs

    def on_object_select(self):
        """
        Catch an object selection change, and updates the modifier view.
        """
        selected_obj = self.obj_list.selected_object
        if selected_obj is not None:
            self.obj_modifier.show()
            self.obj_modifier.object = selected_obj
        else:
            if self._autohide_modifier:
                self.obj_modifier.hide()
            else:
                # Clear the modifier QTextEdits
                self.obj_modifier.clear()

    def on_object_modify(self):
        """
        Catch an object update, and updates the table view of the object.
        """
        self.obj_list.update_kk_obj(self.obj_modifier.object)

    def on_object_create(self, new_obj: type[asf_abstract | asp_abstract]):
        """
        Catch signals that generate new objects, and adds them to the object list.

        Parameters
        ----------
        new_obj : type[asf_abstract | asp_abstract]
            The new object to add.
        """
        if new_obj is not None:
            self.obj_list.add_kk_obj(new_obj)


def demo_app():
    """
    A demo application for the kk_gui widget.
    """
    # Create the Application
    app = QtWidgets.QApplication([])
    app.setApplicationName("kkcalc: Kramers-Kronig Calculator")

    PS_NAME = "Polystyrene"
    PS_STOICHIOMETRY = "CH"
    ps_stoich = stoichiometry(PS_STOICHIOMETRY)

    # Import Data
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    data_file = os.path.normpath(os.path.join(data_dir, "PS_004_-dc.txt"))
    data_PS = np.genfromtxt(data_file, skip_header=4)
    assert data_PS.shape[1] == 2, "Data file must have two columns"

    # Create the atomic scattering factors from NEXAFS data
    asf_PS = asf_im.from_NEXAFS(
        energies=data_PS[:, 0],
        NEXAFS=data_PS[:, 1],
        name=PS_NAME,
        stoichiometry=ps_stoich,
        scale_to_database=True,
    )

    # db_poly = asp_db(ps_stoich, name = PS_NAME)

    P3MEEET_NAME = "P3MEEET"
    P3MEEET_STOICHIOMETRY = "C11H16O3S"
    P3MEEET_file_O = os.path.normpath(os.path.join(data_dir, "P3MEEET_Oxygen_K.csv"))
    P3MEEET_file_S = os.path.normpath(os.path.join(data_dir, "P3MEEET_Sulfur_K.csv"))
    P3MEEET_data_O = np.genfromtxt(P3MEEET_file_O, skip_header=0, delimiter=",")
    P3MEEET_data_S = np.genfromtxt(P3MEEET_file_S, skip_header=0, delimiter=",")
    PEDOT_NAME = "PEDOT-C6C8"
    PEDOT_STOICHIOMETRY = "C21H36O2S"
    PEDOT_file_O = os.path.normpath(os.path.join(data_dir, "PEDOTC6C8_Oxygen_K.csv"))
    PEDOT_file_S = os.path.normpath(os.path.join(data_dir, "PEDOTC6C8_Sulfur_K.csv"))
    PEDOT_data_O = np.genfromtxt(PEDOT_file_O, skip_header=0, delimiter=",")
    PEDOT_data_S = np.genfromtxt(PEDOT_file_S, skip_header=0, delimiter=",")
    asf_P3MEEET_O = asf_im.from_NEXAFS(
        energies=P3MEEET_data_O[:, 0],
        NEXAFS=P3MEEET_data_O[:, 1],
        name=P3MEEET_NAME + " O",
        stoichiometry=P3MEEET_STOICHIOMETRY,
        scale_to_database=True,
    )
    asf_P3MEEET_S = asf_im.from_NEXAFS(
        energies=P3MEEET_data_S[:, 0],
        NEXAFS=P3MEEET_data_S[:, 1],
        name=P3MEEET_NAME + " S",
        stoichiometry=P3MEEET_STOICHIOMETRY,
        scale_to_database=True,
    )
    asf_PEDOT_O = asf_im.from_NEXAFS(
        energies=PEDOT_data_O[:, 0],
        NEXAFS=PEDOT_data_O[:, 1],
        name=PEDOT_NAME + " O",
        stoichiometry=PEDOT_STOICHIOMETRY,
        scale_to_database=True,
    )
    asf_PEDOT_S = asf_im.from_NEXAFS(
        energies=PEDOT_data_S[:, 0],
        NEXAFS=PEDOT_data_S[:, 1],
        name=PEDOT_NAME + " S",
        stoichiometry=PEDOT_STOICHIOMETRY,
        scale_to_database=True,
    )

    # Create the main window
    window = kk_gui(
        objs=[asf_PS, asf_P3MEEET_O, asf_P3MEEET_S, asf_PEDOT_O, asf_PEDOT_S],
        autohide_modifier=False,
    )

    # Run the application
    window.show()
    app.exec()


if __name__ == "__main__":
    demo_app()
