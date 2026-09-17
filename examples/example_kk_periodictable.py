"""
This example demonstrates how to switch the backend for the ASF database.
"""

# External
import matplotlib.pyplot as plt
import numpy as np

# Internal
import kkcalc2 as kk


if __name__ == "__main__":
    formula = "C10H14S"  # P3HT
    energies = np.linspace(100, 3000, 5000)

    # Get the imaginary components of an element
    henke = kk.models.asp_db_im(formula)
    henke_factors = henke(energies)

    # Switch the backend to "periodictable"
    kk.asf_database.db_backend("periodictable")
    pt = kk.models.asp_db_im(formula)
    pt_factors = pt(energies)

    # Plot both
    fig, ax = plt.subplots()
    ax.plot(energies, henke_factors, label="Henke")
    ax.plot(energies, pt_factors, label="Periodictable")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Imaginary ASF")
    ax.set_yscale("log")
    ax.legend()
    plt.show()
