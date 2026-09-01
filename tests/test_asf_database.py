"""
Tests for the optional `periodictable`-based atomic scattering factor database backend.
"""

import numpy as np
import pytest

import kkcalc2.asf_database as db
from kkcalc2.asf_database.periodictable_loader import load_asf_database_periodictable
from kkcalc2.models import asp_db_im, asp_db_re


@pytest.fixture(autouse=True)
def restore_default_backend():
    """Ensure the default 'kkcalc' database backend is restored after each test."""
    yield
    db.set_database_backend("kkcalc")


class TestLoadAsfDatabasePeriodictable:
    """Tests the standalone `load_asf_database_periodictable` loader function."""

    @pytest.mark.parametrize("z", [1, 6, 26])  # Hydrogen, Carbon, Iron
    def test_element_shapes_are_consistent(self, z: int) -> None:
        """Each element's `E`/`Re`/`Im` arrays have mutually consistent shapes."""
        database = load_asf_database_periodictable(elements=[z])

        assert z in database
        element = database[z]
        assert isinstance(element["E"], np.ndarray)
        assert isinstance(element["Re"], np.ndarray)
        assert isinstance(element["Im"], np.ndarray)
        assert element["Im"].shape == (element["E"].shape[0] - 1, 5)
        assert element["Re"].shape == (element["E"].shape[0] - 1,)
        assert not np.any(np.isnan(element["Re"]))
        assert not np.any(np.isnan(element["Im"]))
        assert element["symbol"] is not None
        assert element["mass"] > 0

    def test_default_elements_cover_hydrogen_to_uranium(self) -> None:
        """With no `elements` argument, the full Henke-table element range is loaded."""
        database = load_asf_database_periodictable(
            energy_range=(100.0, 20000.0), num_points=50
        )
        assert 1 in database  # Hydrogen
        assert 92 in database  # Uranium
        assert len(database) > 50

    def test_unknown_atomic_number_is_skipped(self) -> None:
        """Atomic numbers outside the periodic table are silently skipped, not raised."""
        database = load_asf_database_periodictable(elements=[6, 999])
        assert 6 in database
        assert 999 not in database

    def test_missing_periodictable_raises_import_error(self, monkeypatch) -> None:
        """Raises `ImportError` with a helpful message if `periodictable` is not installed."""
        import kkcalc2.asf_database.periodictable_loader as loader_module

        monkeypatch.setattr(loader_module, "has_periodictable", False)
        with pytest.raises(ImportError, match="periodictable"):
            loader_module.load_asf_database_periodictable(elements=[6])


class TestDatabaseBackendSwitching:
    """Tests runtime switching between the 'kkcalc' and 'periodictable' database backends."""

    def test_default_backend_is_kkcalc(self) -> None:
        """The default database backend is 'kkcalc'."""
        assert db.get_database_backend() == "kkcalc"

    def test_switch_to_periodictable_and_back(self) -> None:
        """Switching backends mutates `ASF_DATABASE` in place and updates the reported backend."""
        original_database_id = id(db.ASF_DATABASE)
        original_carbon_E = db.ASF_DATABASE[6]["E"].copy()

        db.set_database_backend("periodictable", elements=[1, 6, 26], num_points=100)

        assert db.get_database_backend() == "periodictable"
        # The dict object itself is mutated in place, not replaced.
        assert id(db.ASF_DATABASE) == original_database_id
        assert set(db.ASF_DATABASE.keys()) == {1, 6, 26}
        # Data has actually changed (different energy grid/resolution).
        assert not np.array_equal(db.ASF_DATABASE[6]["E"], original_carbon_E)

        db.set_database_backend("kkcalc")
        assert db.get_database_backend() == "kkcalc"
        assert id(db.ASF_DATABASE) == original_database_id
        assert np.array_equal(db.ASF_DATABASE[6]["E"], original_carbon_E)

    def test_invalid_backend_raises_value_error(self) -> None:
        """An unrecognised backend name raises `ValueError`."""
        with pytest.raises(ValueError, match="Unknown database backend"):
            db.set_database_backend("not_a_real_backend")

    def test_periodictable_backend_usable_by_models(self) -> None:
        """`asp_db_im`/`asp_db_re` can build usable objects from the `periodictable` backend."""
        db.set_database_backend(
            "periodictable",
            elements=[1, 6],
            energy_range=(100.0, 1000.0),
            num_points=200,
        )

        poly_im = asp_db_im("CH", density=1.05)
        poly_re = asp_db_re("CH", density=1.05)
        asf_im_result = poly_im.to_asf()
        asf_re_result = poly_re.to_asf()

        assert np.all(np.isfinite(asf_im_result.factors))
        assert np.all(np.isfinite(asf_re_result.factors))
