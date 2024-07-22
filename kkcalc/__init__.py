"""
The Kramers Kronig module. 

This module contains all the neccessary ingredients needed to calculate the Kramers-Kronig transform of a given data set.
"""

from kkcalc.models.factors import KK_Datatype
from kkcalc.stoich import stoichiometry
from kkcalc.util import doc_copy

import numpy as np
import numpy.typing as npt

# class asf:
#     """
#     Container for atomic scattering factors.
#     """
#     def __init__(self, 
#                  energies: npt.NDArray,
#                  factors: npt.NDArray
#                  ) -> None:
#         pass