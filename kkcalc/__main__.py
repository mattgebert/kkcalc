"""
Scripts for the KKCalc program, when the module is run from python.

See the `main` function for the main entry point.
Use `python -m kkcalc` to run the program.
"""

from kkcalc.gui.kk_gui import kk_gui
from kkcalc.models import asf_im
import os, numpy as np, PyQt6.QtWidgets as QtWidgets


def main():
    """
    Run a basic instance of the KKCalc gui.

    Some example polystyrene data is used to auto-fill as an example.
    """
    # Create the Application
    app = QtWidgets.QApplication([])
    app.setApplicationName("kkcalc: Kramers-Kronig Calculator")

    # Generate some example data
    from kkcalc.models import asp_db_im
    from kkcalc import stoichiometry

    PS_NAME = "Polystyrene"
    PS_STOICHIOMETRY = "CH"
    ps_stoich = stoichiometry(PS_STOICHIOMETRY)

    # Import Data
    data_dir = os.path.join(os.path.dirname(__file__), "../examples/data")
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


if __name__ == "__main__":
    main()
