"""
Tests the loading, creation and properties of database objects.

Includes:
- Database loader
- Atomic scattering polynomials
- Extended atomic scattering polynomials
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
from kkcalc.asf_database.db_loader import load_asf_database

from ..test_stoich import basic_stoichs as bs
import numpy as np


class TestDatabaseLoading:
    """Test that the database loads the atomic scattering factor data correctly."""

    database: None | dict = None

    def load_database(self):
        self.database = load_asf_database()
        return self.database

    def test_load_database(self):
        """Tests that the database loads without error."""
        db = self.load_database()
        assert db is not None
        assert isinstance(db, dict)

    @pytest.mark.parametrize("element", [1, 6, 26])  # Hydrogen, Carbon, Iron
    def test_database_contents(self, element: int):
        """Tests that the database contains expected elements and data formats."""
        db = self.database if self.database is not None else self.load_database()

        # Check a few known elements
        assert element in db

        # Check that the data arrays are numpy arrays
        asf_data = db[element]
        assert isinstance(asf_data["E"], np.ndarray)
        assert isinstance(asf_data["Im"], np.ndarray)
        assert isinstance(asf_data["Re"], np.ndarray)

        # Check that the arrays have some significantly large length
        assert len(asf_data["E"]) > 50
        assert len(asf_data["Im"]) > 50
        assert len(asf_data["Re"]) > 50


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
