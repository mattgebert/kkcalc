"""
Alternative atomic scattering factor database loader, using the `periodictable` package.

The `periodictable` package (https://periodictable.readthedocs.io/) bundles the same
Henke x-ray scattering factor tables (Lawrence Berkeley Laboratory, Center for X-ray Optics)
used to build kkcalc2's own bundled database, but computes them on demand for an
arbitrary energy grid rather than from a pre-packaged file.

This module is entirely optional: `periodictable` is not a required dependency of kkcalc2.
Use `kkcalc2.asf_database.set_database_backend("periodictable")` to switch the active
`kkcalc2.asf_database.ASF_DATABASE` to use this loader at runtime, or call
`load_asf_database_periodictable` directly to build a standalone database dictionary.

See Also
--------
kkcalc2.asf_database.db_loader.load_asf_database : The default, bundled-file based loader.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from kkcalc2 import conversions
from kkcalc2.asf_database.db_loader import ASFElement

try:
    import periodictable

    has_periodictable = True
except ImportError:
    has_periodictable = False

#: Default photon energy range (eV) used to sample the `periodictable` Henke tables.
DEFAULT_ENERGY_RANGE: tuple[float, float] = (10.0, 30000.0)
#: Default number of energy points used to sample the `periodictable` Henke tables.
DEFAULT_NUM_POINTS: int = 500
#: Atomic numbers covered by the `periodictable`/Henke x-ray scattering factor tables (H to U).
DEFAULT_ELEMENTS: tuple[int, ...] = tuple(range(1, 93))


def load_asf_database_periodictable(
    elements: Sequence[int] | None = None,
    energy_range: tuple[float, float] = DEFAULT_ENERGY_RANGE,
    num_points: int = DEFAULT_NUM_POINTS,
) -> dict[int, ASFElement]:
    """
    Build an atomic scattering factor database using the `periodictable` package.

    For each requested element, samples `periodictable`'s x-ray scattering factors
    (:math:`f'`, :math:`f''`) over a log-spaced photon energy grid, and reshapes the
    result into the same `ASFElement` format used by `kkcalc2.asf_database.ASF_DATABASE`
    (i.e. real point-values, and imaginary piecewise-linear polynomial coefficients).

    Parameters
    ----------
    elements : Sequence[int] | None, optional
        Atomic numbers of the elements to load. By default `None`, which loads every
        element covered by the Henke tables (`DEFAULT_ELEMENTS`, atomic numbers 1-92).
    energy_range : tuple[float, float], optional
        The (min, max) photon energy range in eV to sample. By default `DEFAULT_ENERGY_RANGE`.
    num_points : int, optional
        The number of (log-spaced) energy points to sample within `energy_range`.
        By default `DEFAULT_NUM_POINTS`.

    Returns
    -------
    dict[int, ASFElement]
        A dictionary of elements atomic number keys, with each value consisting of a
        dictionary of values (see `ASFElement`).

    Raises
    ------
    ImportError
        If the optional `periodictable` package is not installed.

    See Also
    --------
    kkcalc2.asf_database.db_loader.load_asf_database : The default, bundled-file based loader.
    kkcalc2.asf_database.set_database_backend : Switch the active database backend at runtime.
    """
    if not has_periodictable:
        raise ImportError(
            "The optional `periodictable` package is required to use this database backend. "
            "Install it with `pip install kkcalc2[periodictable]` or `pip install periodictable`."
        )

    if elements is None:
        elements = DEFAULT_ELEMENTS

    # `periodictable` expects energies in keV, not eV.
    energies_eV: npt.NDArray[np.floating] = np.logspace(
        np.log10(energy_range[0]), np.log10(energy_range[1]), num_points
    )
    energies_keV = energies_eV / 1000.0

    database: dict[int, ASFElement] = {}
    for z in elements:
        try:
            element = periodictable.elements[z]
        except (KeyError, IndexError):
            continue

        f1, f2 = element.xray.scattering_factors(energy=energies_keV)
        f1 = np.asarray(f1, dtype=float)
        f2 = np.asarray(f2, dtype=float)

        # Trim to the (contiguous, interior) subset of energies with valid (non-NaN) data.
        valid = ~(np.isnan(f1) | np.isnan(f2))
        if valid.sum() < 3:
            # Not enough data to build a usable entry (e.g. element not covered by the tables).
            continue
        valid_indices = np.flatnonzero(valid)
        lo, hi = valid_indices[0], valid_indices[-1] + 1
        e_valid = energies_eV[lo:hi]
        f1_valid = f1[lo:hi]
        f2_valid = f2[lo:hi]

        # `Im` stores piecewise-linear polynomial coefficients (matching the bundled database),
        # `Re` stores raw point values aligned with `E[:-1]` (matching the bundled database).
        im_coefs = conversions.ASF_to_ASP(e_valid, f2_valid)

        database[z] = ASFElement(
            E=e_valid,
            Re=f1_valid[:-1],
            Im=im_coefs,
            name=str(element.name).capitalize(),
            symbol=str(element.symbol),
            mass=float(element.mass),
        )

    return database
