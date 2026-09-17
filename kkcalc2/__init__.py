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

import importlib.metadata

from kkcalc2 import asf_database, models, transforms
from kkcalc2.models import (
    PROPERTIES_DICT,
    PROPERTIES_DICT_NO_STOICH,
    KK_Datatype,
    conversions,
    factors,
    polynomials,
)
from kkcalc2.stoich import stoichiometry

# Define the version of the package:
__version__ = importlib.metadata.version("kkcalc2")

__all__ = [
    "PROPERTIES_DICT",
    "PROPERTIES_DICT_NO_STOICH",
    "KK_Datatype",
    "__version__",
    "asf_database",
    "conversions",
    "factors",
    "models",
    "polynomials",
    "stoichiometry",
    "transforms",
]
