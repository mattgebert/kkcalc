"""
This module is the atomic scattering factor database module for KK calc.

It's primarily used to generate expected atomic scattering factors of a given stochiometry.
This can be used to extend an existing spectra to calculate a more reliable KK-transform,
or to generate basic model spectra.

Data is sourced from
- Briggs and Lighthill, 1976, J. Phys. Chem. Ref. Data, 5, 581-837,
- Henke et al., 1993, At. Data Nucl. Data Tables, 54, 181-342.

The atomic scattering factors found in `ASF.json` are calculated using `asf_generator_script.py`.
These scattering factors can be accessed via the `ASF_DATABASE` variable in this module,
or can be loaded using the `load_asf_database` function.

Each element in the database is a dictionary consisting of the following keys:
- 'E': A numpy array of `N+1` photon energies corresponding to intervals of the scattering factor data.
- 'Re': A numpy array of `N-3` real coefficient of the scattering factor. TODO: Why is this N-3?
- 'Im': A 2D numpy array of dimensions `N, 5`, with values of `5` piecewise polynomial coefficients
for the imaginary part of the scattering factors, corresponding to the energies intervals.
"""

from typing import Literal

from kkcalc2.asf_database.db_loader import load_asf_database, ASFElement

ASF_DATABASE: dict[int, ASFElement] = (
    load_asf_database()
)  # spectral data, plus atomic masses

#: The currently active backend used to populate `ASF_DATABASE`.
_database_backend: Literal["kkcalc", "periodictable"] = "kkcalc"

__all__ = [
    "ASF_DATABASE",
    "ASFElement",
    "set_database_backend",
    "get_database_backend",
]


def set_database_backend(backend: Literal["kkcalc", "periodictable"], **kwargs) -> None:
    """
    Switch the backend used to populate `ASF_DATABASE`, at runtime.

    Mutates `ASF_DATABASE` in place (rather than replacing the module attribute), so
    that other modules which imported it by reference (e.g. `from kkcalc2.asf_database
    import ASF_DATABASE`) automatically see the updated data.

    Parameters
    ----------
    backend : Literal["kkcalc", "periodictable"]
        The database backend to switch to.
        - "kkcalc": The default, bundled-file based database (`db_loader.load_asf_database`).
        - "periodictable": An alternative database computed on-demand using the optional
          `periodictable` package (`periodictable_loader.load_asf_database_periodictable`).
    **kwargs
        Additional keyword arguments passed to the backend's loader function.

    Raises
    ------
    ValueError
        If `backend` is not a recognised database backend.
    ImportError
        If `backend` is "periodictable" but the optional `periodictable` package is not installed.

    See Also
    --------
    get_database_backend : Query the currently active database backend.
    kkcalc2.asf_database.periodictable_loader.load_asf_database_periodictable :
        The alternative, `periodictable`-based database loader.
    """
    global _database_backend

    if backend == "kkcalc":
        new_data = load_asf_database(**kwargs)
    elif backend == "periodictable":
        from kkcalc2.asf_database.periodictable_loader import (
            load_asf_database_periodictable,
        )

        new_data = load_asf_database_periodictable(**kwargs)
    else:
        raise ValueError(
            f"Unknown database backend: {backend!r}. Must be 'kkcalc' or 'periodictable'."
        )

    ASF_DATABASE.clear()
    ASF_DATABASE.update(new_data)
    _database_backend = backend


def get_database_backend() -> Literal["kkcalc", "periodictable"]:
    """
    Query the currently active database backend used to populate `ASF_DATABASE`.

    Returns
    -------
    Literal["kkcalc", "periodictable"]
        The currently active database backend.

    See Also
    --------
    set_database_backend : Switch the active database backend at runtime.
    """
    return _database_backend


# Example usage
if __name__ == "__main__":
    for i, (z, data) in enumerate(ASF_DATABASE.items()):
        print(data["name"], data["E"].shape, data["Re"].shape, data["Im"].shape)
        if i >= 5:
            break
