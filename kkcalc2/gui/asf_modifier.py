"""
Creates a widget for modifying an asf_abstract | asp_abstract object.
Allows the modification of material properties such as name, stoichiometry, number density, density, and formula mass.
"""

from PyQt6 import QtCore, QtGui, QtWidgets

from kkcalc2.models import (
    asf_abstract,
    asf_complex,
    asf_im,
    asf_re,
    asp_abstract,
    asp_complex,
    asp_db_extended,
    asp_db_im,
    asp_db_im_extended,
    asp_db_re_extended,
    asp_im,
    asp_re,
)
from kkcalc2.stoich import stoichiometry


class kk_object_modifier(QtWidgets.QWidget):
    objectModified = QtCore.pyqtSignal()
    objectCreated = QtCore.pyqtSignal(object)
    hasHandle = QtCore.pyqtSignal(bool)

    def __init__(
        self, parent=None, obj: type[asf_abstract | asp_abstract] | None = None
    ):
        super().__init__(parent=parent)
        self.setWindowTitle("kkcalc Object Modifier")
        # self._layout = QtWidgets.QGridLayout()
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        # Setup margins if parent is provided.
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._layout.setContentsMargins(0, 0, 0, 0)

        ### Create property items
        # Name
        name_label = QtWidgets.QLabel("Name:")
        name_label.setToolTip("Object name")
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setToolTip("The name of the object; used for graphing.")
        self.name_edit.setPlaceholderText("Object Name")

        # Points
        points_label = QtWidgets.QLabel("Points:")
        points_label.setToolTip("Object energy values")
        self.points_edit = QtWidgets.QLineEdit()
        self.points_edit.setToolTip("The energy points of the object.")
        self.points_edit.setDisabled(True)

        # Properties header and horizontal line
        properties_header = QtWidgets.QLabel("Properties")
        # Add bold font
        font = properties_header.font()
        font.setBold(True)

        def hline_generator():
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            return line

        # Stoichiometry
        stoichiometry_label = QtWidgets.QLabel("Stoich:")
        stoichiometry_label.setToolTip("Stoichiometry")
        self.stoichiometry_edit = QtWidgets.QLineEdit()
        self.stoichiometry_edit.setToolTip(
            "Object stoichiometry e.g. CH, C6H12O6, (CH2)0.5(F2)0.5"
        )
        self.stoichiometry_edit.setPlaceholderText("e.g. CH, C6H12O6, (CH2)0.5(F2)0.5")

        # Relativistic Correction
        relativistic_label = QtWidgets.QLabel("Rel Cor:")
        relativistic_label.setToolTip("Relativistic correction, f0")
        self.relativistic_edit = QtWidgets.QLineEdit()
        self.relativistic_edit.setReadOnly(True)
        self.relativistic_edit.setToolTip(
            "Relativistic correction factor for the object, also known as f0. Inferred from the stochiometry."
        )

        # Number density
        number_density_label = QtWidgets.QLabel("Num. Den.:")
        number_density_label.setToolTip("Number density")
        self.number_density_edit = QtWidgets.QLineEdit()
        self.number_density_edit.setToolTip("Object number density in atoms/cm^3")
        self.number_density_edit.setPlaceholderText("atoms/cm^3")
        self.number_density_edit.setValidator(QtGui.QDoubleValidator())

        # Density
        density_label = QtWidgets.QLabel("Density:")
        density_label.setToolTip("Density")
        self.density_edit = QtWidgets.QLineEdit()
        self.density_edit.setToolTip("Object density in g/cm^3")
        self.density_edit.setPlaceholderText("g/cm^3")
        self.density_edit.setValidator(QtGui.QDoubleValidator())

        # Formula Mass
        formula_mass_label = QtWidgets.QLabel("Form. Mass:")
        formula_mass_label.setToolTip("Formula mass")
        self.formula_mass_edit = QtWidgets.QLineEdit()
        self.formula_mass_edit.setToolTip("Object formula mass in g/mol")
        self.formula_mass_edit.setPlaceholderText("g/mol")
        self.formula_mass_edit.setValidator(QtGui.QDoubleValidator())

        # Transform / Conversion Header
        transform_header = QtWidgets.QLabel("Kramers Kronig Transforms / Conversions")

        # Transform buttons
        self.kk_transform_btn = QtWidgets.QPushButton("KK Transform")
        self.kk_transform_to_complex_btn = QtWidgets.QPushButton("To Complex")

        # Extension Header
        extension_header = QtWidgets.QLabel("Database Scaling/Extension")
        self.scale_to_db_btn = QtWidgets.QPushButton("Scale to DB")
        self.is_extended_edit = QtWidgets.QCheckBox("Extended")
        self.is_extended_edit.setDisabled(True)
        # Extend Data by Stoichiometry
        merge_dom_label = QtWidgets.QLabel("Merge Domains:")
        self.merge_dom_lb_edit = QtWidgets.QLineEdit()
        self.merge_dom_lb_edit.setToolTip("Lower bound for merging domains")
        self.merge_dom_ub_edit = QtWidgets.QLineEdit()
        self.merge_dom_ub_edit.setToolTip("Upper bound for merging domains")
        self.merge_handle_checkbox = QtWidgets.QCheckBox("Show")
        self.merge_handle_checkbox.setDisabled(True)
        self.extend_data_btn = QtWidgets.QPushButton("Extend Data")
        self.extend_data_btn.setToolTip("Extend the data by the stoichiometry database")

        # Distortion fixing (applied to the data before merging with the database).
        self.fix_distortions_checkbox = QtWidgets.QCheckBox("Fix Distortions")
        self.fix_distortions_checkbox.setToolTip(
            "Enable additional correction of the user data before merging with the database."
        )
        self.fix_distortions_method_combo = QtWidgets.QComboBox()
        self.fix_distortions_method_combo.addItems(["grad_min", "prepost_fit"])
        self.fix_distortions_method_combo.setItemData(
            0,
            "grad_min: Fits a single gradient correction to the data, minimizing the offset "
            "between the scaled data and the database data at the merge domain boundaries. "
            "Requires no additional configuration.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.fix_distortions_method_combo.setItemData(
            1,
            "prepost_fit: Fits separate linear corrections to a pre-edge and a post-edge energy "
            "range (see 'Fix Pre/Post Domain' below), then applies these to match the data to the "
            "database at the merge domain boundaries. Requires the pre/post domain fields.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.fix_distortions_method_combo.setToolTip(
            self.fix_distortions_method_combo.itemData(
                0, QtCore.Qt.ItemDataRole.ToolTipRole
            )
        )
        fix_predomain_label = QtWidgets.QLabel("Fix Pre-Domain:")
        self.fix_predomain_lb_edit = QtWidgets.QLineEdit()
        self.fix_predomain_lb_edit.setToolTip(
            "Lower bound of the pre-edge energy range used by the 'prepost_fit' method."
        )
        self.fix_predomain_ub_edit = QtWidgets.QLineEdit()
        self.fix_predomain_ub_edit.setToolTip(
            "Upper bound of the pre-edge energy range used by the 'prepost_fit' method."
        )
        fix_postdomain_label = QtWidgets.QLabel("Fix Post-Domain:")
        self.fix_postdomain_lb_edit = QtWidgets.QLineEdit()
        self.fix_postdomain_lb_edit.setToolTip(
            "Lower bound of the post-edge energy range used by the 'prepost_fit' method."
        )
        self.fix_postdomain_ub_edit = QtWidgets.QLineEdit()
        self.fix_postdomain_ub_edit.setToolTip(
            "Upper bound of the post-edge energy range used by the 'prepost_fit' method."
        )

        ### LAYOUTS
        # Names Layout
        l_names = QtWidgets.QHBoxLayout()
        l_names.addWidget(name_label)
        l_names.addWidget(self.name_edit)
        self._layout.addLayout(l_names)
        # Points Layout
        l_points = QtWidgets.QHBoxLayout()
        l_points.addWidget(points_label)
        l_points.addWidget(self.points_edit)
        self._layout.addLayout(l_points)

        ## Properties layout
        props = QtWidgets.QGridLayout()
        props.addWidget(properties_header, 0, 0, 1, 2)
        props.addWidget(hline_generator(), 1, 0, 1, 2)
        props.addWidget(stoichiometry_label, 2, 0)
        props.addWidget(self.stoichiometry_edit, 2, 1)
        props.addWidget(relativistic_label, 3, 0)
        props.addWidget(self.relativistic_edit, 3, 1)
        props.addWidget(number_density_label, 4, 0)
        props.addWidget(self.number_density_edit, 4, 1)
        props.addWidget(density_label, 5, 0)
        props.addWidget(self.density_edit, 5, 1)
        props.addWidget(formula_mass_label, 6, 0)
        props.addWidget(self.formula_mass_edit, 6, 1)
        self._layout.addLayout(props)

        ## Transform Layout
        trans = QtWidgets.QGridLayout()
        trans.addWidget(transform_header, 0, 0, 1, 4)
        trans.addWidget(hline_generator(), 1, 0, 1, 4)
        trans.addWidget(self.kk_transform_btn, 2, 0, 1, 2)
        trans.addWidget(self.kk_transform_to_complex_btn, 2, 2, 1, 2)
        self._layout.addLayout(trans)

        ## Extension Layout
        extn = QtWidgets.QGridLayout()
        extn.addWidget(extension_header, 0, 0, 1, 2)
        extn.addWidget(self.scale_to_db_btn, 1, 0, 1, 2)
        extn.addWidget(self.is_extended_edit, 1, 2, 1, 2)
        extn.addWidget(hline_generator(), 2, 0, 1, 4)
        extn.addWidget(merge_dom_label, 3, 0, 1, 1)
        extn.addWidget(self.merge_dom_lb_edit, 3, 1, 1, 1)
        extn.addWidget(self.merge_dom_ub_edit, 3, 2, 1, 1)
        # extn.addWidget(self.merge_handle_checkbox, 3, 3, 1, 1) # Temporarily removed, do not show on UI. TODO: Implement.
        extn.addWidget(self.fix_distortions_checkbox, 4, 0, 1, 2)
        extn.addWidget(self.fix_distortions_method_combo, 4, 2, 1, 2)
        extn.addWidget(fix_predomain_label, 5, 0, 1, 1)
        extn.addWidget(self.fix_predomain_lb_edit, 5, 1, 1, 1)
        extn.addWidget(self.fix_predomain_ub_edit, 5, 2, 1, 1)
        extn.addWidget(fix_postdomain_label, 6, 0, 1, 1)
        extn.addWidget(self.fix_postdomain_lb_edit, 6, 1, 1, 1)
        extn.addWidget(self.fix_postdomain_ub_edit, 6, 2, 1, 1)
        extn.addWidget(self.extend_data_btn, 7, 0, 1, 4)
        self._layout.addLayout(extn)

        # Initialise internal object
        self._object = None

        ## Connect signals
        # For updates
        self.stoichiometry_edit.editingFinished.connect(self.on_stoichiomentry_change)
        self.stoichiometry_edit.textEdited.connect(
            lambda: self.stoichiometry_edit.setStyleSheet("")
        )
        self.name_edit.editingFinished.connect(self.update_object)
        self.number_density_edit.editingFinished.connect(self.update_object)
        self.density_edit.editingFinished.connect(self.update_object)
        self.formula_mass_edit.editingFinished.connect(self.update_object)
        # For transformations
        self.kk_transform_btn.clicked.connect(self.transform)
        self.kk_transform_to_complex_btn.clicked.connect(self.to_complex)
        # For extension
        self.scale_to_db_btn.clicked.connect(self.scale)
        self.extend_data_btn.clicked.connect(self.extend)
        self.merge_handle_checkbox.stateChanged.connect(self.hasHandle.emit)
        self.fix_distortions_checkbox.stateChanged.connect(
            self.update_fix_distortions_UI
        )
        self.fix_distortions_method_combo.currentIndexChanged.connect(
            self.update_fix_distortions_UI
        )

        # Initialise UI
        self.clear()
        # Minimize the width.
        self.resize(self.minimumWidth(), self.height())

    def update_fix_distortions_UI(self) -> None:
        """
        Updates the enabled state and tooltip of the fix-distortions controls.

        The pre/post domain fields are only relevant (and enabled) when fixing distortions is
        enabled and the "prepost_fit" method is selected. The combobox tooltip is updated to
        describe the currently selected method.
        """
        index = self.fix_distortions_method_combo.currentIndex()
        self.fix_distortions_method_combo.setToolTip(
            self.fix_distortions_method_combo.itemData(
                index, QtCore.Qt.ItemDataRole.ToolTipRole
            )
        )
        needs_prepost = (
            self.fix_distortions_checkbox.isChecked()
            and self.fix_distortions_method_combo.currentText() == "prepost_fit"
        )
        for widget in (
            self.fix_predomain_lb_edit,
            self.fix_predomain_ub_edit,
            self.fix_postdomain_lb_edit,
            self.fix_postdomain_ub_edit,
        ):
            widget.setEnabled(needs_prepost)
        self.fix_distortions_method_combo.setEnabled(
            self.fix_distortions_checkbox.isChecked()
        )

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
            rc = stoich.relativistic_correction if stoich is not None else None
            num_den, den, fm = obj.number_density, obj.density, obj.formula_mass
            # Fill the fields
            self.name_edit.setText(name if name is not None else "")
            self.stoichiometry_edit.setText(str(stoich) if stoich is not None else "")
            self.relativistic_edit.setText(str(rc) if rc is not None else "")
            self.number_density_edit.setText(
                str(num_den) if num_den is not None else ""
            )
            self.density_edit.setText(str(den) if den is not None else "")
            self.formula_mass_edit.setText(str(fm) if fm is not None else "")
            # Set the number of points
            self.points_edit.setText(str(len(obj.energies)))
            # Run validations
            self.run_validations()
            # Update the transform labels
            self.update_class_dependent_UI()
            # If the object is not extended, set the extension bound defaults to be the min,max energies
            if not obj.is_extended:
                self.merge_dom_lb_edit.setText(f"{obj.energies.min():.2f}")
                self.merge_dom_ub_edit.setText(f"{obj.energies.max():.2f}")

            # Only unblock signals if an object is provided
            self.blockSignals(False)

    def update_object(self):
        """
        Updates the object with the current values.
        """
        # Track if a change is made.
        update: bool = False

        # Get the current object
        obj = self._object
        if obj is None:
            self.clear()
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
                # Update fm as well, to avoid overriding later
                fm = new_stoich.formula_mass
                update = True

        # Compare if the values are different and update
        try:
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
        except ValueError:
            pass

        # Signal if a change has been made.
        if update:
            self.objectModified.emit()
            # Reset the view for any internal changes
            self.set_object(obj)

    def clear(self):
        """
        Removes all text and disables all buttons.
        """
        self.name_edit.setText("")
        self.points_edit.setText("")
        self.stoichiometry_edit.setText("")
        self.relativistic_edit.setText("")
        self.number_density_edit.setText("")
        self.density_edit.setText("")
        self.formula_mass_edit.setText("")
        self.kk_transform_btn.setEnabled(False)
        self.kk_transform_to_complex_btn.setEnabled(False)
        self.is_extended_edit.setChecked(False)
        self.merge_dom_lb_edit.setText("")
        self.merge_dom_ub_edit.setText("")
        self.merge_dom_lb_edit.setDisabled(True)
        self.merge_dom_ub_edit.setDisabled(True)
        self.merge_handle_checkbox.setChecked(False)
        self.extend_data_btn.setEnabled(False)
        self.scale_to_db_btn.setEnabled(False)
        self.fix_distortions_checkbox.setChecked(False)
        self.fix_distortions_method_combo.setCurrentIndex(0)
        self.fix_predomain_lb_edit.setText("")
        self.fix_predomain_ub_edit.setText("")
        self.fix_postdomain_lb_edit.setText("")
        self.fix_postdomain_ub_edit.setText("")
        self.update_fix_distortions_UI()

    @staticmethod
    def valid_stoichiometry(stoich: str) -> bool:
        if stoich == "" or stoich is None:
            return False
        try:
            stoichiometry(stoich)
        except ValueError:
            return False
        return True

    def on_stoichiomentry_change(self) -> None:
        """
        Runs the UI validation and the object update on stoichiometry change.
        """
        self.validate_stoichiometry_UI()
        self.update_object()

    def validate_stoichiometry_UI(self):
        """
        Validates the stoichiometry change.

        If valid, updates the:
            - relativistic correction
            - scale button if the object is not extended
        """
        stoich = self.stoichiometry_edit.text()
        if not self.valid_stoichiometry(stoich):
            self.stoichiometry_edit.setStyleSheet("background-color: red")
            self.extend_data_btn.setEnabled(False)
            self.scale_to_db_btn.setEnabled(False)
            return
        else:
            stoich = stoichiometry(stoich)
            self.stoichiometry_edit.setStyleSheet("background-color: green")
            self.relativistic_edit.setText(str(stoich.relativistic_correction))
            if self._object is not None and not self._object.is_extended:
                self.extend_data_btn.setEnabled(True)
                self.scale_to_db_btn.setEnabled(True)
        # Set the focus to the next field
        self.setFocus()

    def run_validations(self) -> None:
        """Runs all validations on the object."""
        self.validate_stoichiometry_UI()

    def update_class_dependent_UI(self):
        """
        Updates/enables/disables the labels for the various buttons depending on the selected object type.
        """
        # Get the object
        obj: type[asf_abstract | asp_abstract] = self._object

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

        if obj.is_extended:
            self.is_extended_edit.setChecked(True)
            self.merge_dom_lb_edit.setDisabled(True)
            self.merge_dom_ub_edit.setDisabled(True)
            self.merge_handle_checkbox.setChecked(False)
            self.extend_data_btn.setEnabled(False)
            self.stoichiometry_edit.setDisabled(True)
            self.formula_mass_edit.setDisabled(True)
            self.merge_handle_checkbox.setDisabled(True)
            self.scale_to_db_btn.setDisabled(True)
        else:
            self.is_extended_edit.setChecked(False)
            self.merge_dom_lb_edit.setDisabled(False)
            self.merge_dom_lb_edit.setText("")
            self.merge_dom_ub_edit.setDisabled(False)
            self.merge_dom_ub_edit.setText("")
            self.merge_handle_checkbox.setChecked(False)
            self.extend_data_btn.setEnabled(True)
            self.stoichiometry_edit.setDisabled(False)
            self.formula_mass_edit.setDisabled(False)
            self.merge_handle_checkbox.setDisabled(False)
            self.scale_to_db_btn.setDisabled(False)

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
            # Don't improve accuracy if object is not extended, as accuracy will be very poor.
            transform = obj.kk_transform_inv(improve_accuracy=bool(obj.is_extended))
            transform.name = obj.name + "_kk_inv"
        elif isinstance(obj, (asf_im, asp_im)):
            obj: asf_im | asp_im
            # Don't improve accuracy if object is not extended, as accuracy will be very poor.
            transform = obj.kk_transform(improve_accuracy=bool(obj.is_extended))
            transform.name = obj.name + "_kk"

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
            complex_obj = obj.calculate_complex_factors(
                name=obj.name + "_complex",
                improve_accuracy=bool(obj.is_extended),
            )
        elif isinstance(obj, (asp_re, asp_im)):
            obj: asf_im | asp_im
            complex_obj = obj.calculate_complex_polynomial(
                name=obj.name + "_complex",
                improve_accuracy=bool(obj.is_extended),
            )
        else:
            return

        # Send the transformed object
        self.objectCreated.emit(complex_obj)

    def scale_obj(self) -> type[asp_abstract | asf_abstract] | None:
        """Creates a scaled object by matching endpoint amplitude to the stoichiometry database."""
        obj: type[asf_abstract | asp_abstract] = self._object
        # Ignore if no object or already extended
        if obj is None or obj.is_extended:
            return

        # Check if the `fix_distortions` option is enabled.
        fix_dist = self.fix_distortions_checkbox.isChecked()

        stoich = obj.stoichiometry
        if stoich is None:
            # Dialog to ask for stoichiometry
            diag = QtWidgets.QDialog()
            diag.setWindowTitle("Cannot Extend Data")
            diag._layout = QtWidgets.QVBoxLayout()
            diag.setLayout(diag._layout)
            diag._layout.addWidget(QtWidgets.QLabel("No stoichiometry found."))
            diag.exec()
            return

        # Scale the object.
        if isinstance(obj, (asf_im, asf_re)) and obj.stoichiometry is not None:
            copy: type[asf_im | asf_re] = obj.copy(name=obj.name + "_scaled")

            # If the `fix_distortions` option is enabled, collect the fix-distortions options and pass them to the `scale_to_database` method.
            if fix_dist:
                fix_kwargs = self.fix_distortions_kwargs()
                if fix_kwargs is not None:
                    copy.scale_to_database(**fix_kwargs)
                    return copy

            # Otherwise
            copy.scale_to_database()
            return copy

    def scale(self) -> None:
        """Creates a scaled object and emits the objectCreated signal."""
        obj = self.scale_obj()
        if obj is not None:
            self.objectCreated.emit(obj)

    def extend_obj(self) -> type[asp_db_extended] | None:
        """
        Extends the data by the stoichiometry database.
        """
        obj = self._object
        # Ignore if no object or already extended
        if obj is None or obj.is_extended:
            return

        stoich = obj.stoichiometry
        if stoich is None:
            # Dialog to ask for stoichiometry
            diag = QtWidgets.QDialog()
            diag.setWindowTitle("Cannot Extend Data")
            diag._layout = QtWidgets.QVBoxLayout()
            diag.setLayout(diag._layout)
            diag._layout.addWidget(QtWidgets.QLabel("No stoichiometry found."))
            diag.exec()
            return

        # Get the bounds
        try:
            lb = float(self.merge_dom_lb_edit.text())
            ub = float(self.merge_dom_ub_edit.text())
        except ValueError:
            # Dialog to ask for bounds
            diag = QtWidgets.QDialog()
            diag.setWindowTitle("Cannot Extend Data")
            diag._layout = QtWidgets.QVBoxLayout()
            diag.setLayout(diag._layout)
            diag._layout.addWidget(QtWidgets.QLabel("Invalid bounds."))
            diag.exec()
            return

        # Check the bounds
        if lb > ub:
            # Swap bounds
            lb, ub = ub, lb
        elif lb == ub:
            # Dialog to error bounds
            diag = QtWidgets.QDialog()
            diag.setWindowTitle("Cannot Extend Data")
            diag._layout = QtWidgets.QVBoxLayout()
            diag.setLayout(diag._layout)
            diag._layout.addWidget(QtWidgets.QLabel("Bounds are equal."))
            diag.exec()
            return

        # Get the fix-distortions options from the UI.
        fix_kwargs = self.fix_distortions_kwargs()
        if fix_kwargs is None:
            return

        # Create the merge database
        extended: asp_db_extended
        database_asp = asp_db_im(stoichiometry=stoich)
        if isinstance(obj, asp_re):
            obj: asp_re
            obj_asf = obj.to_asf()
            extended = asp_db_re_extended(
                data_asf=obj_asf,
                database=database_asp,
                merge_domain=(lb, ub),
                name=obj.name + "_ext",
                **fix_kwargs,
            )
            return extended

        elif isinstance(obj, asp_im):
            obj: asp_im
            obj_asf = obj.to_asf()
            extended = asp_db_im_extended(
                data_asf=obj_asf,
                database=database_asp,
                merge_domain=(lb, ub),
                name=obj.name + "_ext",
                **fix_kwargs,
            )
            return extended

        elif isinstance(obj, asf_re):
            obj: asf_re
            extended = asp_db_re_extended(
                data_asf=obj,
                database=database_asp,
                merge_domain=(lb, ub),
                name=obj.name + "_ext",
                **fix_kwargs,
            )
            return extended

        elif isinstance(obj, asf_im):
            obj: asf_im
            extended = asp_db_im_extended(
                data_asf=obj,
                database=database_asp,
                merge_domain=(lb, ub),
                name=obj.name + "_ext",
                **fix_kwargs,
            )
            return extended

        return

    def fix_distortions_kwargs(self) -> dict | None:
        """
        Collects the fix-distortions options from the UI, as constructor keyword arguments.

        Returns
        -------
        dict | None
            A dictionary of `fix_distortions`/`fix_distortions_method`/`fix_predomain`/
            `fix_postdomain` keyword arguments for `asp_db_extended` subclasses. `None` if the
            "prepost_fit" method was selected but the pre/post domain fields are invalid
            (a dialog is shown to the user in this case).
        """
        if not self.fix_distortions_checkbox.isChecked():
            return {"fix_distortions": False}

        method = self.fix_distortions_method_combo.currentText()
        if method == "grad_min":
            return {"fix_distortions": True, "fix_distortions_method": "grad_min"}

        # "prepost_fit": also requires valid pre/post domain bounds.
        try:
            fix_predomain = (
                float(self.fix_predomain_lb_edit.text()),
                float(self.fix_predomain_ub_edit.text()),
            )
            fix_postdomain = (
                float(self.fix_postdomain_lb_edit.text()),
                float(self.fix_postdomain_ub_edit.text()),
            )
        except ValueError:
            diag = QtWidgets.QDialog()
            diag.setWindowTitle("Cannot Extend Data")
            diag._layout = QtWidgets.QVBoxLayout()
            diag.setLayout(diag._layout)
            diag._layout.addWidget(
                QtWidgets.QLabel(
                    "Invalid pre/post-domain bounds for the 'prepost_fit' method."
                )
            )
            diag.exec()
            return None

        return {
            "fix_distortions": True,
            "fix_distortions_method": "prepost_fit",
            "fix_predomain": fix_predomain,
            "fix_postdomain": fix_postdomain,
        }

    def extend(self) -> None:
        """
        Extends the data by the stoichiometry database, and emit the objectCreated signal.
        """
        obj = self.extend_obj()
        if obj is not None:
            self.objectCreated.emit(obj)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = kk_object_modifier()
    win.show()
    app.exec()
