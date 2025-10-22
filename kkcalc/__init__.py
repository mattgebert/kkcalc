"""
The Kramers Kronig module.

This base modules contains bindings to essential classes and functions for the calculation
of Kramer-Kronig transforms. In particular, the module provides the following classes:
- `stoichiometry`: A class for the calculation of stoichiometry in a chemical compound.
- `kk_transforms`: A set of functions for the calculation of Kramers-Kronig transforms.
- `conversions`: A set of functions to convert between different data types
(e.g. atomic scattering factors to absorption/dispersion coefficients).
- `KK_Datatype`: An enumeration class for the data types used in `factors` object.
- `factors`: A set of classes to wrap and add methods to experimental data.
- `polynomials`: A set of classes for the calculation of the Kramer-Kronig transforms.
"""

from kkcalc.stoich import stoichiometry
from kkcalc import kk_transforms
from kkcalc.models import (
    conversions,
    polynomials,
    factors,
    KK_Datatype,
    PROPERTIES_DICT,
    PROPERTIES_DICT_NO_STOICH,
)
from kkcalc import models

# Traversable items
__all__ = [
    "stoichiometry",
    "kk_transforms",
    "conversions",
    "polynomials",
    "KK_Datatype",
    "factors",
    "PROPERTIES_DICT",
    "PROPERTIES_DICT_NO_STOICH",
    "models",
]

from importlib.metadata import metadata, version

try:
    # Import the GUI module if appropriate packages are available:
    for key, value in metadata("kkcalc").items():
        print(f"Seen {key}, {value}")
        if "Requires-Dist" in key and 'extra == "gui"' in value:
            # TODO: When is "extra == gui" shown?
            module = __import__(value.split(";")[0])
    # Attempted import on GUI module.
    from kkcalc.gui import kk_gui

    # Cleanup extra names
    locs = locals()
    if "key" in locs:
        del key
    if "value" in locs:
        del value
    if "module" in locs:
        del module

except ImportError:
    pass

# Define the version of the package:
__version__ = version("kkcalc")

# Cleanup extra names
del metadata
del version
