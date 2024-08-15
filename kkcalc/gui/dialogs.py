"""
This module contains the dialog classes for the GUI.
"""
from PyQt6 import QtWidgets

class factor_complexity_dialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Complexity")
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.complexity = None
        self.complexity_buttons = [
            QtWidgets.QRadioButton("Imaginary"),
            QtWidgets.QRadioButton("Real")
        ]

class factor_dtype_dialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Data Type")
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.data_type = None
        
class ascii_pandas_import_dialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Data")
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.data = None
        self.data_type = None
        
        self.data_type_buttons = [
            QtWidgets.QRadioButton("ASCII"),
            QtWidgets.QRadioButton("Pandas")
        ]
        
        self.data_type_buttons[0].setChecked(True)
        self.data_type_buttons[0].toggled.connect(self.on_data_type_change)
        self.data_type_buttons[1].toggled.connect(self.on_data_type_change)