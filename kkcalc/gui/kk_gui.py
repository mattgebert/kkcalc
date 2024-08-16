"""
kk_gui.py is the GUI interface for the Kramers-Kronig Calculator. It is a simple interface that allows the user
to input data, assign to real or imaginary components, and perform a Kramers-Kronig transform on it.
The interface is built using the PyQt6 library.
"""
from PyQt6 import QtWidgets, QtCore, QtGui
import os
import numpy as np
import pandas as pd
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt

from kkcalc.gui.asf_viewer import asf_viewer, GraphType
from kkcalc.gui.asf_modifier import kk_object_modifier
from kkcalc.gui.asf_loader import kk_object_list
from kkcalc.models import asf_abstract, asp_abstract, asp_db_extended, asp_db, asp, asp_im, asp_re, asp_complex, asf, asf_im, asf_re, asf_complex
from kkcalc.stoich import stoichiometry


class kk_gui(QtWidgets.QWidget):
    def __init__(self, parent=None, objs: list[type[asf_abstract | asp_abstract]] = None, autohide_modifier: bool = False):
        super().__init__(parent=parent)
        self.setWindowTitle("Kramers-Kronig Calculator")
        self._layout = QtWidgets.QHBoxLayout()
        self.setLayout(self._layout)
        self._draggable = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._layout.addWidget(self._draggable)
        
        # Setup margins if parent is provided.
        if parent is not None:
            self.setContentsMargins(0,0,0,0)
            self._layout.setContentsMargins(0,0,0,0)
        
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
        
        # Setup the viewer
        self.on_view_change()
        
    def on_has_handle(self, has_handle: bool):
        """
        Catches the signal from the modifier, for when an extended domain is to be created
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
                if graph_type is GraphType.RE_IM_OVERLAY or graph_type is GraphType.RE_IM_SEPARATE:
                    if isinstance(obj, (asf_re, asp_re)):
                        handle_ax = self.viewer.ax1
                    elif isinstance(obj, (asf_im, asp_im)):
                        handle_ax = self.viewer.ax2
                # Create the handle
                if handle_ax is not None:
                    self._handle = SpanSelector(ax = handle_ax, 
                                          onselect=self.on_handle_update, 
                                          direction='horizontal', 
                                          useblit=True,
                                          interactive=True, 
                                          drag_from_anywhere=True
                                          )
                    self._handle.extents = (obj.energies.min(), obj.energies.max())
        else:
            if self._has_handle:
                # Remove the handle
                self._handle.visible = False
                # Lose track of the handle
                self._handle = None
            
    def on_handle_update(self, min_x: float, max_x: float):
        return
        
    def on_view_change(self):
        objs = self.obj_list.checked_objects
        if self._has_handle:
            # Create a temporary asf object to pass to the viewer using the handle
            objs = objs
        # Update the viewer
        self.viewer.scattering_objects = objs
        
    def on_object_select(self):
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
        Catches an object update, and updates the table view of the object.
        """
        self.obj_list.update_kk_obj(self.obj_modifier.object)
    
    def on_object_create(self, new_obj: type[asf_abstract | asp_abstract]):
        """
        Catches signals that generate new objects, and adds them to the object list.
        """
        if new_obj is not None:
            self.obj_list.add_kk_obj(new_obj)
        
if __name__ == "__main__":
    # Create the Application
    app = QtWidgets.QApplication([])
    app.setApplicationName("kkcalc: Kramers-Kronig Calculator")
    
    # Generate some example data
    from kkcalc.models import asp_db
    from kkcalc import stoichiometry
    PS_NAME = "Polystyrene"
    PS_STOICHIOMETRY = "CH"
    ps_stoich = stoichiometry(PS_STOICHIOMETRY)
    
    # Import Data
    data_dir = os.path.join(os.path.dirname(__file__), "../../examples/data")
    data_file = os.path.normpath(os.path.join(data_dir, "PS_004_-dc.txt"))
    data_PS = np.genfromtxt(data_file, skip_header=4)
    assert data_PS.shape[1] == 2, "Data file must have two columns"
    
    # Create the atomic scattering factors from NEXAFS data
    asf_PS = asf.from_NEXAFS(energies = data_PS[:,0], 
                             NEXAFS = data_PS[:,1],
                             name = PS_NAME,
                             stoichiometry = ps_stoich)
    
    # db_poly = asp_db(ps_stoich, name = PS_NAME)
    
    # Create the main window
    window = kk_gui(objs=[asf_PS], autohide_modifier=False)
    
    # Run the application
    window.show()
    app.exec()