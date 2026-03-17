"""
GUI interface for the Kramers-Kronig Calculator.

A simple interface that allows the user to input data, assign to real
or imaginary components, and perform a Kramers-Kronig transform on it.
The interface is built using the PyQt6 library.
"""

from PyQt6 import QtWidgets, QtCore, QtGui
import os
import io
import numpy as np
from matplotlib.widgets import SpanSelector
import pkgutil

from kkcalc.gui.asf_viewer import asf_viewer, GraphType
from kkcalc.gui.asf_modifier import kk_object_modifier
from kkcalc.gui.asf_loader import kk_object_list
from kkcalc.models import (
    asf_abstract,
    asp_abstract,
    asp_im,
    asp_re,
    asf_im,
    asf_re,
)
from kkcalc.stoich import stoichiometry


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
        obj_modifier.merge_dom_lb_edit.textChanged.connect(
            lambda text: self.on_modifier_update(
                float(text) if text != "" else 0.0,
                float(obj_modifier.merge_dom_ub_edit.text())
                if obj_modifier.merge_dom_ub_edit.text() != ""
                else 0.0,
            )
        )
        obj_modifier.merge_dom_ub_edit.textChanged.connect(
            lambda text: self.on_modifier_update(
                float(obj_modifier.merge_dom_lb_edit.text())
                if obj_modifier.merge_dom_lb_edit.text() != ""
                else 0.0,
                float(text) if text != "" else 0.0,
            )
        )
        # Connect the merge handle checkbox to the handle display
        obj_modifier.merge_handle_checkbox.stateChanged.connect(
            self.on_merge_handle_update
        )
        # Connect the viewer graph x-snapping to handle redrawing

        viewer.graphUpdated.connect(self.on_snap)
        # viewer.snap_x_combo.currentIndexChanged.connect(self.on_snap)

        # Setup the viewer
        self.on_view_change()

    def on_snap(self):
        """
        Catch a snap event from the viewer.
        """
        if self._has_handle and self._handle is not None:
            # This is a hack to create a new handle, as updating the existing
            # one does not redraw it properly (positioning, log scale etc).
            self.show_handle(False)  # Force the removal of the handle.
            self.show_handle(
                True
            )  # Force an update of the handle onto the current axis.

    def show_handle(self, has_handle: bool):
        """
        Create a handle when an extended domain is desired.

        Parameters
        ----------
        has_handle : bool
            Whether to create the handle or not (or remove existing).
        """
        # Check if a handle is needed
        if has_handle:
            # Get the current obj
            obj = self.obj_list.selected_object
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
            # Create or update the handle
            if not self._has_handle:
                # Create the handle
                if handle_ax is not None:
                    # Create the handle
                    self._handle = SpanSelector(
                        ax=handle_ax,
                        onselect=self.on_handle_update,
                        direction="horizontal",
                        useblit=True,
                        interactive=True,
                        drag_from_anywhere=True,
                    )
                    self._handle.set_props(color="red", alpha=0.1)
                    self._handle.set_handle_props(color="red", alpha=0.2)
                    # A handle now exists
                    self._has_handle = True
            else:
                # Ensure the axes match
                if (
                    self._handle is not None
                    and handle_ax is not None
                    and self._handle.ax != handle_ax
                ):
                    # Get the old ax
                    old_ax = self._handle.ax
                    # Remove old artists
                    for artist in self._handle.artists + old_ax.artists:
                        try:
                            artist.remove()
                        except ValueError:
                            pass
                    # Remove the old ax from the figure if it's still there
                    try:
                        self.viewer.figure.axes.remove(old_ax)
                    except ValueError:
                        pass
                    # Update the handle to the new ax
                    # self._handle._setup_edge_handles(self._handle._handle_props)
                    # self._handle.new_axes(handle_ax)
                    # self._handle._draw_shape(*self._handle.extents)
            try:
                txt_min, txt_max = (
                    self.obj_modifier.merge_dom_lb_edit.text(),
                    self.obj_modifier.merge_dom_ub_edit.text(),
                )
                self._handle.extents = (float(txt_min), float(txt_max))
            except ValueError:
                self._handle.extents = obj.energies.min(), obj.energies.max()
        else:
            if self._has_handle:
                # Remove the handle
                self._handle.visible = False
                self._handle.clear()
                for artist in self._handle.artists:
                    try:
                        artist.remove()
                    except ValueError:
                        pass
                # Lose track of the handle
                self._handle = None
                self._has_handle = False
        # self.viewer.reset_graph()

    def on_handle_update(self, min_x: float, max_x: float):
        """
        Take handle updates, and update the modifier values.

        Parameters
        ----------
        min_x : float
            The minimum x value of the handle.
        max_x : float
            The maximum x value of the handle.
        """
        # Disable signals of the modifiers
        self.obj_modifier.merge_dom_lb_edit.blockSignals(True)
        self.obj_modifier.merge_dom_ub_edit.blockSignals(True)
        # Update the lb and ub values but don't use the min_x, max_x
        self.obj_modifier.merge_dom_lb_edit.setText(f"{min_x:.2f}")
        self.obj_modifier.merge_dom_ub_edit.setText(f"{max_x:.2f}")
        # Unblock signals
        self.obj_modifier.merge_dom_lb_edit.blockSignals(False)
        self.obj_modifier.merge_dom_ub_edit.blockSignals(False)
        return

    def on_modifier_update(self, min_x: float, max_x: float):
        """
        Take text modifier updates, and update the handle values.

        Parameters
        ----------
        min_x : float
            The minimum x value of the modifier.
        max_x : float
            The maximum x value of the modifier.
        """
        if self._has_handle and self._handle is not None:
            # Take the onselect function
            fn = self._handle.onselect
            self._handle.onselect = lambda *args: None
            # Update the values
            self._handle.extents = (min_x, max_x)
            # Restore the onselect function
            self._handle.onselect = fn
        return

    def on_view_change(self):
        """
        Catch a change of toggling viewed objects.
        """
        objs = self.obj_list.checked_objects
        if self._has_handle:
            # Create a temporary asf object to pass to the viewer using the handle
            objs = objs
        # Update the viewer
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

        # Always update the handle
        self.on_merge_handle_update(self.obj_modifier.merge_handle_checkbox.isChecked())

    def on_merge_handle_update(self, checked: bool):
        """
        Catch a change of the merge handle checkbox, and updates the handle view.

        Parameters
        ----------
        checked : bool
            Whether the merge handle checkbox is checked.
        """
        selected_obj = self.obj_list.selected_object
        requires_handle: bool = (
            selected_obj is not None and not selected_obj.is_extended and checked
        )
        self.show_handle(requires_handle)

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
