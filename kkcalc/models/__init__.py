"""
This module is the main entry point for the models of the Kramers-Kronig Calculator.

It consists of the following submodules:
        * common: Contains the common material models used by the other models
        * factors: Contains the models for the atomic scattering factors (`asf`)
        * polynomials: Contains the models for polynomial respresentations of the `asf`.
        * conversions: A library of functions for the conversions between datatypes.

Additionally, the database models are imported from the `kkcalc.asf_database.db_models` module,
for model completeness.
"""

# Import the common base models
from kkcalc.models.common import (
    atomic_scattering_abstract,
    atomic_scattering,
    PROPERTIES_DICT,
)

# Import the usage models
from kkcalc.models.factors import (
    asf,
    asf_im,
    asf_re,
    asf_complex,
    asf_abstract,
    KK_Datatype,
)
from kkcalc.models.polynomials import asp, asp_im, asp_re, asp_complex, asp_abstract

# Import the conversions module
from kkcalc.models import conversions

# Import the database models
from kkcalc.asf_database.db_models import (
    asp_db_abstract,
    asp_db_im,
    asp_db_re,
    asp_db_complex,
    asp_db_extended,
    asp_db_im_extended,
    asp_db_re_extended,
    asp_db_complex_extended,
)
