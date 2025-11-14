"""
Tests the loading, creation and properties of database objects.

Includes:
- Database loader
- Atomic scattering polynomials
- Extended atomic scattering polynomials
"""

# Stdlib
import warnings
import pkgutil
import io

# External
import pytest
import numpy as np

# Internal
from kkcalc.models import (
    asp_db_im,
    asp_db_re,
    asp_db_im_extended,
    asp_db_re_extended,
    asp_db_complex,
    asp_db_complex_extended,
    asf_im,
)
from kkcalc.asf_database.db_loader import load_asf_database
from ..test_stoich import basic_stoichs as bs
from kkcalc import stoichiometry as kk_stoich


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


class TestDbScaling:
    """Tests that ensure the consistency of scaling functions"""

    @pytest.mark.parametrize("fix_distortions", [False, True])
    @pytest.mark.parametrize(
        "merge_domain",
        [
            None,
            (275.0, 390.0),
            # (280.0, 290.0),
        ],
    )
    def test_scaling_consistency(
        self,
        merge_domain: None | tuple[float, float],
        fix_distortions: bool,
    ):
        """
        Test that the scaling between the following functions is consistent:
        - `asp_db_abstract.scale_data`
        - `asp_db_extended.extend_data_with_db`
        """

        # Load example data
        PS_datafile = pkgutil.get_data(
            "kkcalc",
            "data/PS_004_-dc.txt",
        )
        PS_data = np.genfromtxt(
            io.BytesIO(PS_datafile),
            skip_header=4,
        )
        PS_energies, PS_NEXAFS = PS_data[:, 0], PS_data[:, 1]

        PS_stoich = kk_stoich(bs.POLYMER_PS)
        density = 1.05  # g/cm^3

        # Create an ASF dataset
        PS_asf_dataset = asf_im.from_NEXAFS(
            PS_energies, PS_NEXAFS, stoichiometry=PS_stoich
        )

        # Get a database of the elements in the stoichiometry
        PS_asp_database = asp_db_im(PS_stoich, density=density)

        # Create an extended polynomial
        PS_asf_extended = asp_db_im_extended(
            data_asf=PS_asf_dataset,
            database=PS_asp_database,
            fix_distortions=fix_distortions,
            merge_domain=merge_domain,
        )

        # Data covers 275 to 390 eV
        PS_asf_dataset_scaled = PS_asf_dataset.copy()
        PS_asf_dataset_scaled.scale_to_database(
            fix_distortions=fix_distortions,
            merge_domain=merge_domain,
        )

        # Check the two methods give the same result in the overlapping region
        PS_asf_extended = PS_asf_extended.to_asf(
            PS_asf_dataset_scaled.energies,
        )

        assert np.allclose(
            PS_asf_extended.factors,
            PS_asf_dataset_scaled.factors,
        )
