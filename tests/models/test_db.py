"""
Tests the creation and properties of database atomic scattering polynomial objects.
"""

import pytest
import warnings
from kkcalc.asf_database import (
    asp_db_im,
    asp_db_re,
    asp_db_im_extended,
    asp_db_re_extended,
    asp_db_complex,
    asp_db_complex_extended,
)

from ..test_stoich import basic_stoichs as bs


class TestDatabasePolynomials:
    """Tests the building and methods of stoichiometric polynomial objects from the database."""

    @pytest.mark.parametrize(
        "model",
        [
            asp_db_im,
            asp_db_re,
            asp_db_complex,
        ],
    )
    def test_creation(self, model):
        stoich = bs.POLYMER_P3HT
        density = 1.33
        poly = model(stoich, density=density)
        assert poly.stoichiometry == stoich
