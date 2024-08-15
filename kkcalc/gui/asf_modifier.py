"""
Creates a widget for modifying an asf_abstract | asp_abstract object.
Allows the modification of material properties such as name, stoichiometry, number density, density, and formula mass.
"""

from PyQt6 import QtWidgets, QtCore, QtGui
from kkcalc.models import asf_abstract, asp_abstract, asf, asf_re, asf_im, asf_complex, asp_re, asp_im, asp_complex, asp
from kkcalc.stoich import stoichiometry
        
class kk_object_modifier(QtWidgets.QWidget):
    objectModified = QtCore.pyqtSignal()
    objectCreated = QtCore.pyqtSignal(asf_abstract)
    
    def __init__(self, parent = None, obj: type[asf_abstract | asp_abstract] | None = None):
        super().__init__(parent=parent)
        self.setWindowTitle("kkcalc Object Modifier")
        self._layout = QtWidgets.QGridLayout()
        self.setLayout(self._layout)
        
        # Setup margins if parent is provided.
        if parent is not None:
            self.setContentsMargins(0,0,0,0)
            self._layout.setContentsMargins(0,0,0,0)
        
        ### Create property items
        # Name
        name_label = QtWidgets.QLabel("Name:")
        name_label.setToolTip("Object name")
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setToolTip("The name of the object; used for graphing.")
        
        # Properties header and horizontal line
        properties_header = QtWidgets.QLabel("Properties")
        def hline_generator():
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            return line
        
        # Stoichiometry
        stoichiometry_label = QtWidgets.QLabel("Stoich:")
        stoichiometry_label.setToolTip("Stoichiometry")
        self.stoichiometry_edit = QtWidgets.QLineEdit()
        self.stoichiometry_edit.setToolTip("Object stoichiometry e.g. CH, C6H12O6, (CH2)0.5(F2)0.5")
        
        # Relativistic Correction
        relativistic_label = QtWidgets.QLabel("Rel Cor:")
        relativistic_label.setToolTip("Relativistic correction")
        self.relativistic_edit = QtWidgets.QLineEdit()
        self.relativistic_edit.setReadOnly(True)
        self.relativistic_edit.setToolTip("Relativistic correction factor for the object, inferred from the stochiometry.")
        
        # Number density
        number_density_label = QtWidgets.QLabel("Num. Den.:")
        number_density_label.setToolTip("Number density")
        self.number_density_edit = QtWidgets.QLineEdit()
        self.number_density_edit.setToolTip("Object number density in atoms/cm^3")
        self.number_density_edit.setValidator(QtGui.QDoubleValidator())
        
        # Density
        density_label = QtWidgets.QLabel("Density:")
        density_label.setToolTip("Density")
        self.density_edit = QtWidgets.QLineEdit()
        self.density_edit.setToolTip("Object density in g/cm^3")
        self.density_edit.setValidator(QtGui.QDoubleValidator())

        # Formula Mass
        formula_mass_label = QtWidgets.QLabel("Form. Mass:")
        formula_mass_label.setToolTip("Formula mass")
        self.formula_mass_edit = QtWidgets.QLineEdit()
        self.formula_mass_edit.setToolTip("Object formula mass in g/mol")
        self.formula_mass_edit.setValidator(QtGui.QDoubleValidator())
        
        # Transform / Conversion Header
        transform_header = QtWidgets.QLabel("Kramers Kronig Transforms / Conversions")
        
        self.kk_transform_btn = QtWidgets.QPushButton("KK Transform")
        self.kk_transform_to_complex_btn = QtWidgets.QPushButton("To Complex")
        
        ### Add elements to the layout
        self._layout.addWidget(name_label, 0, 0)
        self._layout.addWidget(self.name_edit, 0, 1, 1, 3)
        self._layout.addWidget(properties_header, 1, 0, 1, 4)
        self._layout.addWidget(hline_generator(), 2, 0, 1, 4)
        self._layout.addWidget(stoichiometry_label, 3, 0, 1, 1)
        self._layout.addWidget(self.stoichiometry_edit, 3, 1, 1, 3)
        self._layout.addWidget(relativistic_label, 4, 0, 1, 1)
        self._layout.addWidget(self.relativistic_edit, 4, 1, 1, 3)
        self._layout.addWidget(number_density_label, 5, 0, 1, 1)
        self._layout.addWidget(self.number_density_edit, 5, 1, 1, 3)
        self._layout.addWidget(density_label, 6, 0, 1, 1)
        self._layout.addWidget(self.density_edit, 6, 1, 1, 3)
        self._layout.addWidget(formula_mass_label, 7, 0, 1, 1)
        self._layout.addWidget(self.formula_mass_edit, 7, 1, 1, 3)
        self._layout.addWidget(transform_header, 8, 0, 1, 4)
        self._layout.addWidget(hline_generator(), 9, 0, 1, 4)
        self._layout.addWidget(self.kk_transform_btn, 10, 0, 1, 2)
        self._layout.addWidget(self.kk_transform_to_complex_btn, 10, 2, 1, 2)
        
        # Initialise internal object
        self._object = None
        
        ## Connect signals 
        # For updates
        self.stoichiometry_edit.editingFinished.connect(self.validate_stoichiometry_change)
        self.stoichiometry_edit.textEdited.connect(lambda: self.stoichiometry_edit.setStyleSheet(""))
        self.name_edit.textEdited.connect(self.update_object)
        self.number_density_edit.textEdited.connect(self.update_object)
        self.density_edit.textEdited.connect(self.update_object)
        self.formula_mass_edit.textEdited.connect(self.update_object)
        # For transformations
        self.kk_transform_btn.clicked.connect(self.transform)
        self.kk_transform_to_complex_btn.clicked.connect(self.to_complex)
    
    
    @property
    def object(self) -> type[asf_abstract | asp_abstract] | None:
        """
        Returns the currently selected object.
        
        Parameters
        ----------
        obj : type[asf_abstract | asp_abstract] | None
            A new object to set as the current object.

        Returns
        -------
        type[asf_abstract | asp_abstract] | None
            The currently selected object.
        """
        return self._object
    
    @object.setter
    def object(self, obj: type[asf_abstract | asp_abstract] | None):
        self.set_object(obj)
        
    def set_object(self, obj: type[asf_abstract | asp_abstract] | None):
        # While updating, block signals
        self.blockSignals(True)
        if obj is None:
            self._object = None
            self.name_edit.setText("")
            self.stoichiometry_edit.setText("")
            self.relativistic_edit.setText("")
            self.number_density_edit.setText("")
            self.density_edit.setText("")
            self.formula_mass_edit.setText("")
        else:
            # Store the reference to the object
            self._object = obj
            # Get the object properties
            name, stoich = obj.name, obj.stoichiometry
            rc = stoich.relativistic_correction
            num_den, den, fm = obj.number_density, obj.density, obj.formula_mass
            # Fill the fields
            self.name_edit.setText(obj.name if obj.name is not None else "")
            self.stoichiometry_edit.setText(str(stoich) if stoich is not None else "")
            self.relativistic_edit.setText(str(rc) if rc is not None else "")
            self.number_density_edit.setText(str(num_den) if num_den is not None else "")
            self.density_edit.setText(str(den) if den is not None else "")
            self.formula_mass_edit.setText(str(fm) if fm is not None else "")
            # Only unblock signals if an object is provided
            self.blockSignals(False)
            # Run validations
            self.run_validations()
            # Update the transform labels
            self.switch_transform_labels()
    
    def update_object(self):
        """
        Updates the object with the current values.
        """
        # Track if a change is made.
        update:bool = False
        
        # Get the current object
        obj = self._object
        if obj is None:
            return
        
        # Get the edit values
        name = self.name_edit.text()
        stoich = self.stoichiometry_edit.text()
        num_den = self.number_density_edit.text()
        den = self.density_edit.text()
        fm = self.formula_mass_edit.text()
        
        # Convert to None if empty
        name = name if name != "" else None
        stoich = stoich if stoich != "" else None
        num_den = num_den if num_den != "" else None
        den = den if den != "" else None
        fm = fm if fm != "" else None
        
        # Validate and update
        if self.valid_stoichiometry(stoich):
            new_stoich = stoichiometry(stoich)
            if new_stoich != obj.stoichiometry:
                obj.stoichiometry = new_stoich
                update = True
            
        # Compare if the values are different and update
        if name != obj.name:
            obj.name = name
            update = True
        if num_den != obj.number_density:
            obj.number_density = float(num_den)
            update = True
        if den != obj.density:
            obj.density = float(den)
            update = True
        if fm != obj.formula_mass:
            obj.formula_mass = float(fm)
            update = True
        
        # Signal if a change has been made.
        if update:
            self.objectModified.emit()
    
    @staticmethod
    def valid_stoichiometry(stoich: str) -> bool:
        try:
            stoichiometry(stoich)
        except ValueError as e:
            return False
        return True
        
    def validate_stoichiometry_change(self):
        """
        Validates the stoichiometry change, and updates the relativistic correction if valid.
        """
        stoich = self.stoichiometry_edit.text()
        if not self.valid_stoichiometry(stoich):
            self.stoichiometry_edit.setStyleSheet("background-color: red")
            return
        else:
            stoich = stoichiometry(stoich)
            self.stoichiometry_edit.setStyleSheet("background-color: green")
            self.relativistic_edit.setText(str(stoich.relativistic_correction))
            self.update_object()
        # Set the focus to the next field
        self.setFocus()
        
    def run_validations(self) -> None:
        """Runs all validations on the object."""
        self.validate_stoichiometry_change()
        return
    
            
    def switch_transform_labels(self):
        """
        Updates the labels for the transform buttons depending on the selected object type.
        """
        obj = self._object
        # Change the labels depending on the object type
        if isinstance(obj, (asf_re, asp_re)):
            self.kk_transform_btn.setText("KK Transform Inv")
        elif isinstance(obj, (asf_im, asp_im)):
            self.kk_transform_btn.setText("KK Transform")
            
        # Disable the buttons if no object or object is complex.
        if obj is None or isinstance(obj, (asf_complex, asp_complex)):
            self.kk_transform_btn.setEnabled(False)
            self.kk_transform_to_complex_btn.setEnabled(False)
        else:
            self.kk_transform_btn.setEnabled(True)
            self.kk_transform_to_complex_btn.setEnabled(True)
            
    def transform(self):
        """
        Uses the Kramers Kronig transform on the object.
        
        Emits the objectCreated signal with the new transformed object.
        """
        obj = self._object
        if obj is None:
            return
        if isinstance(obj, (asf_re, asp_re)):
            obj: asf_re | asp_re
            transform = obj.kk_transform_inv()
        elif isinstance(obj, (asf_im, asp_im)):
            obj: asf_im | asp_im
            transform = obj.kk_transform()
        # Send the transformed object
        self.objectCreated.emit(transform)
        
    def to_complex(self):
        """
        Performs the Kramers Kronig transform to on the object, then 
        combines the real and imaginary parts into a complex object.
        
        Emits the objectCreated signal with the new transformed object.
        """
        obj = self._object
        if obj is None:
            return
        if isinstance(obj, asf_re | asf_im):
            obj: asf_re | asf_im
            complex_obj = obj.calculate_complex_factors()
        elif isinstance(obj, asp_re | asp_im):
            obj: asf_im | asp_im
            complex_obj = obj.calculate_complex_polynomial()
        else:
            return
        # Send the transformed object
        self.objectModified.emit(complex_obj)