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
        
