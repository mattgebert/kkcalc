"""
This module is the main entry point for the models of the Kramers-Kronig Calculator.

It consists of the following submodules:
        * common: Contains the common material models used by the other models
        * factors: Contains the models for the atomic scattering factors (`asf`)
        * polynomials: Contains the models for polynomial respresentations of the `asf`.
        * conversions: A library of functions for the conversions between datatypes.
        * db_models: Contains the models for accessing and using the atomic scattering
                     factor database.

Additionally, the database models are imported from the `kkcalc2.asf_database.db_models` module,
for model completeness.
"""

# Import the conversions module
from kkcalc2 import conversions
from kkcalc2 import transforms as transforms

# Import the usage models
from kkcalc2.models import common, db_models, factors, polynomials

# Import the common base models
from kkcalc2.models.common import (
    PROPERTIES_DICT,
    PROPERTIES_DICT_NO_STOICH,
    atomic_scattering,
    atomic_scattering_abstract,
)

# Import the database models
from kkcalc2.models.db_models import (
    asp_db_abstract,
    asp_db_complex,
    asp_db_complex_extended,
    asp_db_extended,
    asp_db_im,
    asp_db_im_extended,
    asp_db_re,
    asp_db_re_extended,
)
from kkcalc2.models.factors import (
    KK_Datatype,
    asf,
    asf_abstract,
    asf_complex,
    asf_im,
    asf_re,
)
from kkcalc2.models.polynomials import asp, asp_abstract, asp_complex, asp_im, asp_re

__all__ = [
    "PROPERTIES_DICT",  # Common models and types
    "PROPERTIES_DICT_NO_STOICH",  # Common models and types
    "KK_Datatype",  # Atomic Scattering Factor models and types
    "asf",  # Atomic Scattering Factor models and types
    "asf_abstract",  # Atomic Scattering Factor models and types
    "asf_complex",  # Atomic Scattering Factor models and types
    "asf_im",  # Atomic Scattering Factor models and types
    "asf_re",  # Atomic Scattering Factor models and types
    "asp",  # Atomic Scattering Polynomial models and types
    "asp_abstract",  # Atomic Scattering Polynomial models and types
    "asp_complex",  # Atomic Scattering Polynomial models and types
    "asp_db_abstract",  # Database models
    "asp_db_complex",  # Database models
    "asp_db_complex_extended",  # Database models
    "asp_db_extended",  # Database models
    "asp_db_im",  # Database models
    "asp_db_im_extended",  # Database models
    "asp_db_re",  # Database models
    "asp_db_re_extended",  # Database models
    "asp_im",  # Atomic Scattering Polynomial models and types
    "asp_re",  # Atomic Scattering Polynomial models and types
    "atomic_scattering",  # Common models and types
    "atomic_scattering_abstract",  # Common models and types
    "common",  # Class modules
    "conversions",  # Conversions module
    "db_models",  # Class modules
    "factors",  # Class modules
    "polynomials",  # Class modules
    "transforms",  # Kramers-Kronig Transforms module
]
