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
from kkcalc.models import conversions, polynomials, factors, KK_Datatype

from kkcalc.gui import kk_gui

# Define the version of the package:
import importlib.metadata

__version__ = importlib.metadata.version("kkcalc")
