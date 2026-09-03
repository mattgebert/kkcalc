"""
Scripts for the KKCalc program, when the module is run from python.

See the `main` function for the main entry point.
Use `python -m kkcalc` to run the program.
"""

# Standard Library Imports
import io
import os
import pkgutil
import traceback

# External Imports
import numpy as np

from kkcalc2 import stoichiometry

# Internal Imports
from kkcalc2.models import asf_im

# Optional Imports
hasQT: bool
try:
    import PyQt6  # noqa: F401
    from PyQt6 import QtWidgets

    from kkcalc2.gui.kk_gui import kk_gui

    hasQT = True
except ImportError:
    hasQT = False


def main():
    """
    Run a basic instance of the KKCalc gui.

    Some example polystyrene data is used to auto-fill as an example.
    """
    # Create the Application
    app = QtWidgets.QApplication([])

    app.setApplicationName("kkcalc: Kramers-Kronig Calculator")

    # Generate some example data

    PS_NAME = "Polystyrene"
    PS_STOICHIOMETRY = "C8H8"
    ps_stoich = stoichiometry(PS_STOICHIOMETRY)

    # Import Data
    try:
        # Try package relative pathing
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        data_file = os.path.normpath(os.path.join(data_dir, "PS_004_-dc.txt"))
        data_PS = np.genfromtxt(data_file, skip_header=4)
    except FileNotFoundError as e:
        # Try resource pathing via pkgutil
        data_bytes = pkgutil.get_data("kkcalc2", "data/PS_004_-dc.txt")
        if data_bytes is None:
            raise FileNotFoundError(
                "Could not find example data file from package kkcalc2 at `data/PS_004_-dc.txt`"
            ) from e
        data_PS = np.genfromtxt(io.BytesIO(data_bytes), skip_header=4)

    assert data_PS.shape[1] == 2, "Data file must have two columns"

    # Create the atomic scattering factors from NEXAFS data
    asf_PS = asf_im.from_NEXAFS(
        energies=data_PS[:, 0],
        NEXAFS=data_PS[:, 1],
        name=PS_NAME,
        stoichiometry=ps_stoich,
        scale_to_database=True,
    )

    # Create the main window
    window = kk_gui(
        objs=[
            asf_PS,
        ],
        autohide_modifier=False,
    )

    # Run the application
    window.show()
    app.exec()


def main_with_traceback():
    """
    Run the main function but catch errors.
    """
    try:
        main()
    except Exception as e:  # noqa: BLE001 - Catch all exceptions to show a GUI error message
        # Create a QT window to display the error
        app = QtWidgets.QApplication([])
        app.setApplicationName("kkcalc: Kramers-Kronig Calculator (Error)")
        error_dialog = QtWidgets.QErrorMessage()
        # Prepare the message: the error and the traceback
        msg = f"An error occurred, causing kkcalc to crash.:\
               \n{e!s}\
               \nPlease report this issue at https://github.com/xraysoftmat/kkcalc/issues"
        error_dialog.showMessage(msg)
        app.exec()

        # Add the detail
        tb = f"{traceback.format_exc()}"
        layout = error_dialog.layout()
        if layout is not None:
            layout.addWidget(QtWidgets.QLabel("Detailed traceback:"))
            layout.addWidget(QtWidgets.QLabel(tb))
        error_dialog.exec()


if __name__ == "__main__":
    if not hasQT:
        # Use stdlib python graphics to show the error.
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror(
            "KKCalc2 Error",
            "PyQt6 is not installed. Please install PyQt6 to run KKCalc.",
        )
    else:
        main_with_traceback()
