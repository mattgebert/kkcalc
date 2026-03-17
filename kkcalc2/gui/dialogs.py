"""
This module contains the dialog classes for the GUI.
"""

from PyQt6 import QtWidgets, QtCore, QtGui
from enum import Enum
import os
import pandas as pd
import numpy as np
from typing import Any
from kkcalc2.models.factors import KK_Datatype, KK_DATATYPE_DOCS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)


class dtype_dialog(QtWidgets.QDialog):
    """
    A dialog to select the datatype of the data.

    Includes choices for complexity (real/imaginary), and datatype (KK_Datatype).
    Choices are available in the `EnumComplexity` and `KK_Datatype` enums.
    """

    class EnumComplexity(Enum):
        REAL = 0
        r"""Real components of the atomic scattering factors (i.e. :math:`\delta`, :math:`f_1`)."""
        IMAGINARY = 1
        r"""Imaginary components of the atomic scattering factors (i.e. :math:`\beta`, :math:`f_2`, :math:`f''`)."""
        COMPLEX = 2
        r"""Complex components of the atomic scattering factors (i.e. :math:`\delta + i \beta`, :math:`f_1 + i f_2`, :math:`f' + i f''`)."""

    EnumComplexity.REAL.__doc__ = r"Real components of the atomic scattering factors (i.e. :math:`\delta`, :math:`f_1`)."
    EnumComplexity.IMAGINARY.__doc__ = r"Imaginary components of the atomic scattering factors (i.e. :math:`\beta`, :math:`f_2`, :math:`f''`)."
    EnumComplexity.COMPLEX.__doc__ = r"Complex components of the atomic scattering factors (i.e. :math:`\delta + i \beta`, :math:`f_1 + i f_2`, :math:`f' + i f''`)."

    def __init__(self, parent=None, name: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(
            "Select datatype" if name is None else "Select datatype: " + name
        )
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        self.datatype: KK_Datatype | None = None
        """The selected datatype"""
        self.complexity: dtype_dialog.EnumComplexity | None = None
        """The selected form complexity."""

        # Complexity buttons
        self.complexity_buttons = [
            QtWidgets.QPushButton(
                dtype_dialog.EnumComplexity.REAL.name.lower().capitalize()
            ),
            QtWidgets.QPushButton(
                dtype_dialog.EnumComplexity.IMAGINARY.name.lower().capitalize()
            ),
            QtWidgets.QPushButton(
                dtype_dialog.EnumComplexity.COMPLEX.name.lower().capitalize()
            ),
        ]
        label = QtWidgets.QLabel(
            "Select the complexity of the data"
            + (":" if name is None else f" for {name}:")
        )
        self._layout.addWidget(label)
        blayout_complexity = QtWidgets.QHBoxLayout()
        self._layout.addLayout(blayout_complexity)
        for button in self.complexity_buttons:
            button.clicked.connect(self.on_complexity_push)
            blayout_complexity.addWidget(button)
            button.setToolTip(
                dtype_dialog.EnumComplexity[button.text().upper()].__doc__
            )

        # Datatype buttons
        self.dtype_buttons = [
            QtWidgets.QPushButton(dtype.name.lower().capitalize())
            for dtype in KK_Datatype
            if dtype != KK_Datatype.UNDEFINED
        ]
        label = QtWidgets.QLabel(
            "Select the datatype " + (":" if name is None else f" for {name}:")
        )
        self._layout.addWidget(label)
        blayout_dtype = QtWidgets.QHBoxLayout()
        self._layout.addLayout(blayout_dtype)
        for i, button in enumerate(self.dtype_buttons):
            button.clicked.connect(self.on_dtype_push)
            blayout_dtype.addWidget(button)
            button.setToolTip(KK_DATATYPE_DOCS[KK_Datatype(i + 1).name])
            button.setEnabled(False)

        blayout_dtype.setEnabled(False)

    def on_complexity_push(self):
        """
        Collects the selected complexity, then closes the dialog.
        """
        self.complexity = dtype_dialog.EnumComplexity(
            self.complexity_buttons.index(self.sender())
        )
        if self.complexity is not None:
            # Show the datatype buttons
            for i, button in enumerate(self.dtype_buttons):
                if (
                    i != KK_Datatype.NEXAFS.value - 1
                    or self.complexity is dtype_dialog.EnumComplexity.IMAGINARY
                ):
                    button.setEnabled(True)
                else:
                    button.setEnabled(False)

    def on_dtype_push(self):
        """
        Collects the selected datatype, then closes the dialog.
        """
        # +1 to account for the UNDEFINED datatype
        self.datatype = KK_Datatype(self.dtype_buttons.index(self.sender()) + 1)
        self.accept()

    # Validate
    def validate(self) -> bool:
        """
        Validates that both complexity and datatype have been selected.
        """
        # NEXAFS cannot be real or complex
        if (
            self.complexity != dtype_dialog.EnumComplexity.IMAGINARY
            and self.datatype == KK_Datatype.NEXAFS
        ):
            return False
        # Return true if both are selected
        return self.complexity is not None and self.datatype is not None


class import_data_dialog(QtWidgets.QDialog):
    DEFAULT_X_LABEL = "Energy (eV)"
    DEFAULT_Y_LABEL = "Amplitude (A.U.)"

    PROCESSOR_DOC = {
        "NONE": "No processor selected",
        "ASCII": "Import data from an ASCII file reading each line",
        "NUMPY": "Import data from a NumPy file",
        "PANDAS": "Import data using Pandas",
    }

    class EnumProcessor(Enum):
        # These doc_strings do not persist at compilation.
        NONE = 0
        """No processor selected"""
        ASCII = 1
        """Import data from an ASCII file reading each line"""
        NUMPY = 2
        """Import data from a NumPy file"""
        PANDAS = 3
        """Import data using Pandas"""

    # Set the docstrings for the processors
    for processor in EnumProcessor:
        processor.__doc__ = PROCESSOR_DOC[processor.name]

    def __init__(self, parent=None, path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Import Data")
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        self.load_layout = QtWidgets.QHBoxLayout()
        self._layout.addLayout(self.load_layout)
        self.parse_layout = QtWidgets.QHBoxLayout()
        self._layout.addLayout(self.parse_layout)
        self.data_layout = QtWidgets.QVBoxLayout()
        self._layout.addLayout(self.data_layout)
        self.error_layout = QtWidgets.QHBoxLayout()
        self._layout.addLayout(self.error_layout)
        self._layout.setStretch(0, 0)
        self._layout.setStretch(1, 0)
        self._layout.setStretch(2, 100)
        # self._layout.setStretch(3, 1)

        # Loading
        self.filepath_edit = QtWidgets.QLineEdit(path)
        self.filepath_edit.setPlaceholderText("Enter the path to the file")
        self.filepath_select_btn = QtWidgets.QPushButton("...")
        self.processor = QtWidgets.QComboBox()
        self.processor.addItems([x.name for x in import_data_dialog.EnumProcessor])
        [
            self.processor.setItemData(
                i, x.__doc__, role=QtCore.Qt.ItemDataRole.ToolTipRole
            )
            for i, x in enumerate(import_data_dialog.EnumProcessor)
        ]
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_layout.addWidget(self.filepath_edit)
        self.load_layout.addWidget(self.filepath_select_btn)
        self.load_layout.addWidget(QtWidgets.QLabel("Loader: "))
        self.load_layout.addWidget(self.processor)
        self.load_layout.addWidget(self.load_btn)

        # Define parsing attributes
        self.load_data: Any | None = None
        """Associated data with the loaded file, determined by the `load_dtype`."""
        self.load_dtype: import_data_dialog.EnumProcessor | None = None
        """The selected processor for loading the data."""
        self.load_headers: list[str] | None = None
        """The unprocessed headers of the loaded data."""
        self.load_filename: str | None = None
        """The filename of the loaded data."""

        # Parsing
        delimiter_label = QtWidgets.QLabel("Delimiter:")
        self.delimiter_edit = QtWidgets.QLineEdit()
        self.delimiter_edit.setPlaceholderText("Tab (\\t), Space ( ), Comma (,), etc.")
        self.delimiter_edit.setToolTip(
            "".join(
                [
                    "The delimiter used to separate the data.\n",
                    "For example, a comma (,) or tab (\\t)\n",
                    "Leave blank for auto-detection.",
                ]
            )
        )
        header_label = QtWidgets.QLabel("Skip Headers:")
        self.skip_header_rows_edit = QtWidgets.QSpinBox()
        self.skip_header_rows_edit.setValue(0)
        self.skip_header_rows_edit.setMinimum(0)
        self.skip_header_rows_edit.setMinimumWidth(80)
        self.skip_header_rows_edit.setMaximum(10000)
        self.parse_layout.addWidget(delimiter_label)
        self.parse_layout.addWidget(self.delimiter_edit)
        self.parse_layout.addWidget(header_label)
        self.parse_layout.addWidget(self.skip_header_rows_edit)

        # Viewing
        self.viewer_table = QtWidgets.QTableWidget()
        self.viewer_table.setRowCount(0)
        self.viewer_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectColumns
        )
        self.viewer_table.clearSelection()
        self.viewer_outcome = QtWidgets.QHBoxLayout()
        self.result_header_btn = QtWidgets.QPushButton("View Headers")
        self.result_rows_edit = QtWidgets.QLineEdit()
        self.result_rows_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Ignored
        )
        self.result_rows_edit.setDisabled(True)
        self.result_cols_edit = QtWidgets.QLineEdit()
        self.result_cols_edit.setDisabled(True)
        self.result_cols_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Ignored
        )
        self.show_graph_checkbox = QtWidgets.QCheckBox("Graph?")
        self.show_graph_checkbox.setToolTip(
            "Show a preview of the data in a separate graph window."
        )
        self.show_graph_checkbox.setChecked(True)
        self.x_column_select = QtWidgets.QLineEdit()
        self.x_column_select.setValidator(QtGui.QIntValidator())
        self.x_column_select.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        self.__highlight_x: int = -1
        """The highlighted column for the X data."""
        self.x_column_use_btn = QtWidgets.QPushButton("Use Sel.")
        self.x_column_use_btn.setToolTip("Use the selected column as the X data.")
        self.y_column_select = QtWidgets.QLineEdit()
        self.y_column_select.setValidator(QtGui.QIntValidator())
        self.y_column_select.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        self.__highlight_y: int = -1
        self.y_column_use_btn = QtWidgets.QPushButton("Use Sel.")
        self.y_column_use_btn.setToolTip("Use the selected column as the Y data.")
        self.result_cols_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        """The highlighted column for the Y data."""
        self.result_accept_btn = QtWidgets.QPushButton("Accept")
        self.viewer_outcome.addWidget(self.result_header_btn)
        self.viewer_outcome.addWidget(QtWidgets.QLabel("Rows: "))
        self.viewer_outcome.addWidget(self.result_rows_edit)
        self.viewer_outcome.addWidget(QtWidgets.QLabel("Columns: "))
        self.viewer_outcome.addWidget(self.result_cols_edit)
        self.viewer_outcome.addWidget(self.show_graph_checkbox)
        self.viewer_outcome.addWidget(QtWidgets.QLabel("Energy Col#"))
        self.viewer_outcome.addWidget(self.x_column_select)
        self.viewer_outcome.addWidget(self.x_column_use_btn)
        self.viewer_outcome.addWidget(QtWidgets.QLabel("Amp Col#"))
        self.viewer_outcome.addWidget(self.y_column_select)
        self.viewer_outcome.addWidget(self.y_column_use_btn)
        self.viewer_outcome.addWidget(self.result_accept_btn)
        self.data_layout.addWidget(self.viewer_table)
        self.data_layout.addLayout(self.viewer_outcome)

        # Errors
        self.error_edit = QtWidgets.QTextEdit()
        self.error_edit.setDisabled(True)
        self.error_layout.addWidget(QtWidgets.QLabel("Error: "))
        self.error_layout.addWidget(self.error_edit)

        # Graphing
        self.plot_window: QtWidgets.QWidget | None = QtWidgets.QWidget()
        self.plot_window.setWindowTitle("Data Preview")
        self.plot_window.hide()
        self.plot_layout = QtWidgets.QVBoxLayout()
        self.plot_window.setLayout(self.plot_layout)
        self.plot_fig = plt.Figure()
        self.plot_canvas = FigureCanvas(self.plot_fig)
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
        self.plot_canvas.setMinimumHeight(300)
        self.plot_layout.addWidget(self.plot_canvas)
        self.plot_layout.addWidget(self.plot_toolbar)

        # Connections
        self.filepath_select_btn.clicked.connect(self.on_select_file)
        self.filepath_edit.editingFinished.connect(self.validate_file)
        self.load_btn.clicked.connect(self.on_load)
        self.result_accept_btn.clicked.connect(self.on_accept)
        self.result_header_btn.clicked.connect(self.on_view_headers)
        self.x_column_select.textEdited.connect(self.display_data)
        self.y_column_select.textEdited.connect(self.display_data)
        self.x_column_use_btn.clicked.connect(self.update_xcol_with_selected)
        self.y_column_use_btn.clicked.connect(self.update_ycol_with_selected)

        self._default_brush = None

        # Init
        self.validate_file()
        self.display_data()
        self.viewer_hide()
        self.error_hide()

    def on_select_file(self):
        """
        Open a file dialog to select the file.
        """
        diag = QtWidgets.QFileDialog(parent=self)
        diag.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)  # Single file:
        diag.show()
        if diag.exec():
            self.filepath_edit.setText(diag.selectedFiles()[0])
            self.validate_file()

    def validate_file(self):
        """Validate the file path and enable the load button."""
        # Check if the file exists
        if not os.path.isfile(self.filepath_edit.text()):
            # invalid path
            if self.filepath_edit.text() == "":
                self.filepath_edit.setStyleSheet(None)
            else:
                self.filepath_edit.setStyleSheet("background-color: red;")
            self.load_btn.setEnabled(False)
        else:
            self.filepath_edit.setStyleSheet(None)
            self.load_btn.setEnabled(True)
            self.on_load()

    def on_load(self):
        """
        Load the data from the file to view.
        """
        # Get parsing attributes
        dtype: import_data_dialog.EnumProcessor = import_data_dialog.EnumProcessor(
            self.processor.currentIndex()
        )
        fname: str = self.filepath_edit.text()
        delim: str | None = self.delimiter_edit.text()
        delim = None if delim == "" else delim
        skip_header_rows: int = self.skip_header_rows_edit.value()

        try:
            with open(fname, "r") as f:
                # Load the data
                temp_headers = [f.readline() for _ in range(skip_header_rows)]
                match dtype:
                    case import_data_dialog.EnumProcessor.NONE:
                        raise ValueError(
                            "No processor selected. Use `ASCII`, `NUMPY` or `PANDAS`."
                        )
                    case import_data_dialog.EnumProcessor.ASCII:
                        load_data: list[str] | list[list[Any]]
                        load_data = f.readlines()
                        if delim is not None:
                            load_data = [
                                line.strip().split(delim) for line in load_data
                            ]
                            # Remove empty strings from the split lines
                            load_data = [
                                [item for item in line if item != ""]
                                for line in load_data
                            ]
                            # Convert to float if possible
                            for i, line in enumerate(load_data):
                                for j, item in enumerate(line):
                                    try:
                                        load_data[i][j] = float(item)
                                    except ValueError:
                                        pass
                        self.load_data = load_data
                    case import_data_dialog.EnumProcessor.NUMPY:
                        self.load_data = np.loadtxt(f, delimiter=delim)
                    case import_data_dialog.EnumProcessor.PANDAS:
                        match fname.split(".")[-1]:
                            case "csv":
                                self.load_data = pd.read_csv(
                                    f, delimiter=delim, skiprows=skip_header_rows
                                )
                            case "xls" | "xlsx":
                                self.load_data = pd.read_excel(
                                    f, skiprows=skip_header_rows
                                )
                            case "h5" | "hdf5":
                                self.load_data = pd.read_hdf(
                                    f, skiprows=skip_header_rows
                                )
                            case "xml":
                                self.load_data = pd.read_xml(
                                    f, skiprows=skip_header_rows
                                )
                            case _:
                                # Default use table read
                                self.load_data = pd.read_table(
                                    f, delimiter=delim, skiprows=skip_header_rows
                                )
                    case _:
                        raise ValueError("Invalid processor selection")
                # Then update the parameters
                self.load_filename = fname
                self.load_dtype = dtype
                self.load_headers = temp_headers
        except Exception as e:
            print("Importing Error:", e)
            self.load_data = None
            self.load_dtype = None
            self.load_headers = None
            self.viewer_hide()
            self.error_edit.setText(str(e))
            self.error_show()
            return

        if self.load_data is not None:
            # Default to not accepting the data
            self.result_accept_btn.setEnabled(False)

            # Check if data has two dimensions (energies + amplitudes) and update counts.
            if hasattr(self.load_data, "shape"):
                self.result_rows_edit.setText(str(self.load_data.shape[0]))
                if len(self.load_data.shape) > 1:
                    self.result_cols_edit.setText(str(self.load_data.shape[1]))
                    self.result_accept_btn.setEnabled(True)
            elif hasattr(self.load_data, "__len__") and not isinstance(
                self.load_data, str
            ):
                self.result_rows_edit.setText(str(len(self.load_data)))
                if len(self.load_data) > 0:
                    if hasattr(self.load_data[0], "__len__") and not isinstance(
                        self.load_data[0], str
                    ):
                        self.result_cols_edit.setText(str(len(self.load_data[0])))
                        self.result_accept_btn.setEnabled(True)
            else:
                self.result_rows_edit.setText("")
                self.result_cols_edit.setText("")
            self.display_data()

            if self.load_headers is not None:
                self.result_header_btn.setEnabled(len(self.load_headers) > 0)

            # # Set default selections to the first two columns if the datashape is > 1
            # if hasattr(self.load_data, "shape") and len(self.load_data.shape) > 1 and self.load_data.shape[1] > 1:
            #     self.x_column_select.setText("0")
            #     self.x_column_select.textEdited.emit(self.x_column_select.text())
            #     self.y_column_select.setText("1")
            #     self.y_column_select.textEdited.emit(self.y_column_select.text())
            #     self.x_column_use_btn.clicked.emit()
            #     self.y_column_use_btn.clicked.emit()

    def display_data(self):
        """Displays the collected data"""
        if self.load_data is None:
            self.viewer_hide()
            self.error_hide()
            return
        else:
            if self._default_brush is None:
                item = self.viewer_table.item(0, 0)
                if item is not None:
                    self._default_brush = item.background()
                else:
                    self._default_brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))

            match self.load_dtype:
                case import_data_dialog.EnumProcessor.ASCII:
                    # Display the data as a table
                    self.viewer_table.setHorizontalHeaderLabels(["Lines"])
                    # Check if data is 1D or 2D:
                    if isinstance(self.load_data, list):
                        if isinstance(self.load_data[0], list):
                            # List of lists
                            self.viewer_table.setRowCount(len(self.load_data))
                            self.viewer_table.setColumnCount(len(self.load_data[0]))
                            for i, line in enumerate(self.load_data):
                                for j, item in enumerate(line):
                                    self.viewer_table.setItem(
                                        i, j, QtWidgets.QTableWidgetItem(str(item))
                                    )
                        else:
                            # List of strings
                            self.viewer_table.setRowCount(len(self.load_data))
                            self.viewer_table.setColumnCount(1)
                            for i, line in enumerate(self.load_data):
                                self.viewer_table.setItem(
                                    i, 0, QtWidgets.QTableWidgetItem(line)
                                )
                    else:
                        self.viewer_table.setRowCount(0)
                        self.viewer_table.setColumnCount(0)
                        self.viewer_table.setItem(
                            0, 0, QtWidgets.QTableWidgetItem(self.load_data)
                        )

                case import_data_dialog.EnumProcessor.NUMPY:
                    # Display the data as a table
                    rows, cols = self.load_data.shape
                    self.viewer_table.setRowCount(rows)
                    self.viewer_table.setColumnCount(cols)
                    self.viewer_table.setHorizontalHeaderLabels(
                        [str(i) for i in range(cols)]
                    )
                    for i, row in enumerate(self.load_data):
                        for j, value in enumerate(row):
                            self.viewer_table.setItem(
                                i, j, QtWidgets.QTableWidgetItem(str(value))
                            )

                case import_data_dialog.EnumProcessor.PANDAS:
                    # Display the data as a table
                    rows, cols = self.load_data.shape
                    self.viewer_table.setRowCount(rows)
                    self.viewer_table.setColumnCount(cols)
                    self.viewer_table.setHorizontalHeaderLabels(self.load_data.columns)
                    # self.viewer_table.setHorizontalHeaderLabels([str(i) for i in range(cols)])
                    for i, row in self.load_data.iterrows():
                        for j, value in enumerate(row):
                            self.viewer_table.setItem(
                                i, j, QtWidgets.QTableWidgetItem(str(value))
                            )
                case _:
                    raise ValueError("Invalid processor selection")

            # Highlight / remove highlighting of the selected columns
            xcol = self.x_column_select.text()
            ycol = self.y_column_select.text()
            colcount = self.viewer_table.columnCount()
            # By default, select the first two columns if they exist.
            if colcount >= 2:
                if xcol == "":
                    xcol = "0"
                    self.x_column_select.setText(xcol)
                elif str.isdigit(xcol) and int(xcol) >= colcount and colcount != 2:
                    # Out of range, but has been selected so set to blank.
                    xcol = ""
                    self.x_column_select.setText(xcol)
                if ycol == "":
                    ycol = "1"
                    self.y_column_select.setText(ycol)
                elif str.isdigit(ycol) and int(ycol) >= colcount and colcount != 2:
                    # Out of range, but has been selected so set to blank.
                    ycol = ""
                    self.y_column_select.setText(ycol)
            if xcol != "":
                # New selection!
                xcol = int(xcol)
                if xcol != self.__highlight_x and self.__highlight_x > -1:
                    for i in range(self.viewer_table.rowCount()):
                        item = self.viewer_table.item(i, self.__highlight_x)
                        if item is not None:
                            item.setBackground(self._default_brush)
                for i in range(self.viewer_table.rowCount()):
                    item = self.viewer_table.item(i, xcol)
                    if item is not None:
                        item.setBackground(
                            QtGui.QBrush(QtGui.QColor(117, 250, 141, 128))
                        )
                    # self.viewer_table.item(i, xcol).setForeground(QtGui.QBrush(QtGui.QColor(117, 250, 141, 128)))
                self.x_column_select.setStyleSheet(
                    "background-color: rgba(117, 250, 141, 128);"
                )
                self.__highlight_x = xcol
            else:
                # Empty selection!
                self.x_column_select.setStyleSheet(None)
                if self.__highlight_x > -1:
                    for i in range(self.viewer_table.rowCount()):
                        item = self.viewer_table.item(i, self.__highlight_x)
                        if item is not None:
                            item.setBackground(self._default_brush)
                    self.__highlight_x = -1
                    self.x_column_select.setStyleSheet(None)

            if ycol != "":
                # New selection!
                ycol = int(ycol)
                if ycol != self.__highlight_y and self.__highlight_y > -1:
                    for i in range(self.viewer_table.rowCount()):
                        item = self.viewer_table.item(i, self.__highlight_y)
                        if item is not None:
                            item.setBackground(self._default_brush)
                for i in range(self.viewer_table.rowCount()):
                    item = self.viewer_table.item(i, ycol)
                    if item is not None:
                        item.setBackground(
                            QtGui.QBrush(QtGui.QColor(159, 252, 253, 128))
                        )
                    # self.viewer_table.item(i, ycol).setForeground(QtGui.QBrush(QtGui.QColor(159, 252, 253, 128)))
                self.y_column_select.setStyleSheet(
                    "background-color: rgba(159, 252, 253, 128);"
                )
                self.__highlight_y = ycol
            else:
                # Empty selection!
                self.y_column_select.setStyleSheet(None)
                if self.__highlight_y > -1:
                    for i in range(self.viewer_table.rowCount()):
                        item = self.viewer_table.item(i, self.__highlight_y)
                        if item is not None:
                            item.setBackground(self._default_brush)
                    self.__highlight_y = -1
                    self.y_column_select.setStyleSheet(None)

            # Clear existing Figure
            self.plot_fig.clear()
            if (
                self.show_graph_checkbox.isChecked()
                and self.__highlight_x > -1
                and self.__highlight_y > -1
                and self.__highlight_x != self.__highlight_y
                and self.viewer_table.rowCount() > 0
                and self.viewer_table.columnCount() > 0
            ):
                if self.plot_window is None:
                    self.plot_window = QtWidgets.QWidget()
                    self.plot_window.setLayout(self.plot_layout)
                self.result_accept_btn.setEnabled(True)
                ax = self.plot_fig.add_subplot(111)
                sel_dat = self.selected_data
                if sel_dat is not None:
                    ax.plot(*sel_dat)
                ax.set_xlabel(import_data_dialog.DEFAULT_X_LABEL)
                ax.set_ylabel(import_data_dialog.DEFAULT_Y_LABEL)
                self.plot_fig.tight_layout()
                self.plot_canvas.draw()
                self.plot_window.show()
            else:
                self.result_accept_btn.setEnabled(False)
                if self.plot_window is not None:
                    self.plot_window.hide()

            self.viewer_table.resizeColumnsToContents()
            self.viewer_show()
            self.error_hide()

    def on_view_headers(self):
        """
        Display the headers of the loaded data.
        """
        if self.load_headers is not None:
            # Create a new dialog to display the headers
            diag = QtWidgets.QDialog(self)
            diag.setWindowTitle("Headers: " + os.path.basename(self.load_filename))
            layout = QtWidgets.QVBoxLayout()
            header_edit = QtWidgets.QTextEdit()
            layout.addWidget(header_edit)
            diag.setLayout(layout)
            for header in self.load_headers:
                header_edit.append(header.strip())
            diag.show()

    def on_accept(self):
        """
        Collects the selected datatype, then closes the dialog.
        """
        # Check all conditions are met:
        try:
            x_col = int(self.x_column_select.text())
        except ValueError:
            self.error_edit.setText("Invalid X Column selection")
            self.error_show()
            return
        try:
            y_col = int(self.y_column_select.text())
        except ValueError:
            self.error_edit.setText("Invalid Y Column selection")
            self.error_show()
            return
        try:
            _, cols = (
                int(self.result_rows_edit.text()),
                int(self.result_cols_edit.text()),
            )
        except ValueError:
            self.error_edit.setText("Data requires two dimensions, for X|Y columns.")
            self.error_show()
            return

        if x_col < 0 or x_col >= cols:
            self.error_edit.setText("Invalid X Column selection")
            self.error_show()
            return

        if y_col < 0 or y_col >= cols:
            self.error_edit.setText("Invalid Y Column selection")
            self.error_show()
            return

        self.accept()

    def viewer_hide(self):
        self.viewer_table.hide()
        for i in range(self.viewer_outcome.count()):
            child = self.viewer_outcome.itemAt(i)
            child: QtWidgets.QWidgetItem
            child.widget().hide()

    def viewer_show(self):
        self.viewer_table.show()
        for i in range(self.viewer_outcome.count()):
            child = self.viewer_outcome.itemAt(i)
            child: QtWidgets.QWidgetItem
            child.widget().show()

    def error_hide(self):
        for i in range(self.error_layout.count()):
            child = self.error_layout.itemAt(i)
            child: QtWidgets.QWidgetItem
            child.widget().hide()

    def error_show(self):
        for i in range(self.error_layout.count()):
            child = self.error_layout.itemAt(i)
            child: QtWidgets.QWidgetItem
            child.widget().show()

    def update_xcol_with_selected(self):
        """
        Updates the X column selection with the selected column.
        """
        col: int = self.viewer_table.currentColumn()
        if col > -1:
            self.x_column_select.setText(str(col))
            self.x_column_select.textEdited.emit(self.x_column_select.text())
            self.viewer_table.clearSelection()

    def update_ycol_with_selected(self):
        """
        Updates the Y column selection with the selected column.
        """
        col: int = self.viewer_table.currentColumn()
        if col > -1:
            self.y_column_select.setText(str(col))
            self.y_column_select.textEdited.emit(self.y_column_select.text())
            self.viewer_table.clearSelection()

    @property
    def selected_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Collects the user selected data from the viewer.
        """
        i = int(self.x_column_select.text())
        j = int(self.y_column_select.text())
        np_data: np.ndarray
        if self.load_dtype == import_data_dialog.EnumProcessor.PANDAS:
            data: pd.DataFrame = self.load_data
            np_data = data.to_numpy()
        elif self.load_dtype == import_data_dialog.EnumProcessor.NUMPY:
            np_data = (
                np.asarray(self.load_data)
                if self.load_data is not None
                else np.array([])
            )
        elif self.load_dtype == import_data_dialog.EnumProcessor.ASCII:
            if isinstance(self.load_data, list):
                if isinstance(self.load_data[0], list):
                    # Remove any lines that aren't homogenous.
                    ave_len = np.median([len(line) for line in self.load_data])
                    med_data = [
                        line
                        for line in self.load_data
                        if len(line) == ave_len
                        and all(
                            isinstance(item, (int, float, complex)) for item in line
                        )
                    ]
                    np_data = np.array(med_data)
                else:
                    np_data = np.array([])
            else:
                np_data = np.array([])
        else:
            return None
        if (
            len(np_data.shape) > 1
            and np_data.shape[1] > 1
            and i != j
            and i < np_data.shape[0]
            and j < np_data.shape[1]
        ):
            X = np_data[:, i]
            Y = np_data[:, j]
            return X, Y
        else:
            return None


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    # Test each dialog
    dialogs = [dtype_dialog, import_data_dialog]
    print(len(dialogs))
    for i, dialog in enumerate(dialogs):
        print(f"{i}: {dialog.__name__}")
        d = dialog()
        d.show()
        result = d.exec()
        print(result)

    sys.exit()
