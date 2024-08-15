"""
kk_gui.py is the GUI interface for the Kramers-Kronig Calculator. It is a simple interface that allows the user
to input data, assign to real or imaginary components, and perform a Kramers-Kronig transform on it.
The interface is built using the PyQt6 library.
"""
from PyQt6 import QtWidgets, QtCore, QtGui
import os
import numpy as np
import pandas as pd
from kkcalc.gui.asf_viewer import asf_viewer
from kkcalc.gui.asf_modifier import kk_object_modifier
from kkcalc.gui.asf_loader import kk_object_list
from kkcalc.models import asf_abstract, asp_abstract
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
        
        # Connect the viewer to the object list
        obj_list.viewSelectionChanged.connect(self.on_view_change)
        obj_list.selectedObjectChanged.connect(self.on_object_select)
        obj_modifier.objectModified.connect(self.on_object_modify)
        obj_modifier.objectCreated.connect(self.on_object_create)
        
        # Setup the viewer
        self.on_view_change()
        
    def on_view_change(self):
        objs = self.obj_list.checked_objects
        self.viewer.scattering_objects = objs
        
    def on_object_select(self):
        selected_obj = self.obj_list.selected_object
        if self._autohide_modifier and selected_obj is not None:
            self.obj_modifier.show()
            self.obj_modifier.object = selected_obj
        else:
            self.obj_modifier.hide()
        
    def on_object_modify(self):
        """
        Catches an object update, and updates the table view of the object.
        """
        self.obj_list.update_kk_obj(self.obj_modifier.object)
    
    def on_object_create(self, new_obj: type[asf_abstract | asp_abstract]):
        """
        Catches signals that generate new objects, and adds them to the object list.
        """
        print("Create: ", new_obj)
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
    db_poly = asp_db(ps_stoich, name = PS_NAME)
    
    # Create the main window
    window = kk_gui(objs=[db_poly], autohide_modifier=True)
    
    # Run the application
    window.show()
    app.exec()