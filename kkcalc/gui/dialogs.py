"""
This module contains the dialog classes for the GUI.
"""
from PyQt6 import QtWidgets
from enum import Enum

class factor_complexity_dialog(QtWidgets.QDialog):
    
    class EnumComplexity(Enum):
        REAL = 0
        IMAGINARY = 1
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Complexity")
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        self.complexity = None
        self.complexity_buttons = [
            QtWidgets.QPushButton(factor_complexity_dialog.EnumComplexity.REAL.name.lower().capitalize()),
            QtWidgets.QPushButton(factor_complexity_dialog.EnumComplexity.IMAGINARY.name.lower().capitalize()),
        ]
        label = QtWidgets.QLabel("Select the complexity of the data:")
        self._layout.addWidget(label)
        for button in self.complexity_buttons:
            button.clicked.connect(self.on_complexity_change)
            self._layout.addWidget(button)
            
    def on_complexity_change(self):
        self.complexity = factor_complexity_dialog.EnumComplexity(self.complexity_buttons.index(self.sender()))
        self.accept()

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