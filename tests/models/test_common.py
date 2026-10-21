"""
Model tests for the common base information of atomic scattering.
"""

import pytest

from kkcalc2 import models
from kkcalc2 import stoichiometry as kk_stoich

from ..test_stoich import fractional_stoichs as fs

# Define some mass values for testing
MASS_VALUES = {
    "1": 1.00784,  # Hydrogen
    "2": 4.0026,  # Helium
    "3": 6.94,  # Lithium
    "4": 9.0122,  # Beryllium
    "5": 10.81,  # Boron
    "6": 12.011,  # Carbon
    "7": 14.007,  # Nitrogen
    "8": 15.999,  # Oxygen
    "16": 32.06,  # Sulfur
    "17": 35.45,  # Chlorine
}


class TestCommon:
    @pytest.mark.parametrize(
        "kwargs, msgs",
        [
            ({}, []),
            ({"name": "Sample"}, []),
            ({"name": "Sample", "number_density": 1.1}, []),
            ({"name": "Sample", "number_density": 1.1, "stoichiometry": "CH"}, []),
            (
                {
                    "name": "Sample",
                    "number_density": 1.1,
                    "stoichiometry": "CH",
                    "density": 1.8,
                },
                ["Competing information"],
            ),
        ],
    )
    def test_instantiation_atomic_scattering(
        self, kwargs: dict, msgs: list[str]
    ) -> None:
        """Tests the creation of an `atomic_scattering` object, with expected errors."""
        # Create the object
        if msgs:
            with pytest.raises(ValueError, match=msgs[0]):
                _ = models.atomic_scattering(**kwargs)
        else:
            _ = models.atomic_scattering(**kwargs)

    @pytest.mark.parametrize("composition", [fs.POLYMER_PS])
    def test_propogation(self, composition: kk_stoich.COMPOSITION_TYPING) -> None:
        """Tests the propogation of the `atomic_scattering` object."""
        # Create the object
        atomic_scattering = models.atomic_scattering(
            name="Sample",
            number_density=1.1,
            stoichiometry=composition,
        )
        # Parse the stoichiometry:
        stoich = kk_stoich(composition)
        # Get the propogated formula mass
        fm = atomic_scattering.formula_mass
        # Calculate the fomula mass from known values
        fm_calc = sum(
            [MASS_VALUES[str(atom)] * counts for atom, counts in stoich.composition]
        )

        assert f"{fm:0.2f}" == f"{fm_calc:0.2f}", (
            f"Formula mass {fm} != {fm_calc} at 2 decimal places."
        )

        old_density = atomic_scattering.density
        # Modify the formula mass
        with pytest.raises(ValueError):
            # Cannot do this while a stoich is defined...
            atomic_scattering.formula_mass = fm * 2
        # Remove the stoichiometry
        atomic_scattering.stoichiometry = None
        # Re-attempt
        atomic_scattering.formula_mass = fm * 2
        # Check the density has updated
        assert old_density * 2 == atomic_scattering.density, (
            "Density did not update correctly after formula mass change."
        )

    def test_density_and_number_density_are_derived_from_material_properties(
        self,
    ) -> None:
        """Density, formula mass, and stoichiometry consistently determine number density."""
        sample = models.atomic_scattering(density=1.0, stoichiometry="H2O")
        expected_number_density = 1.0 * 6.02214076e23 / sample.formula_mass

        assert sample.number_density == pytest.approx(expected_number_density)
        assert sample.formula_mass == pytest.approx(18.01528, rel=1e-4)

    def test_density_is_derived_from_number_density_and_stoichiometry(self) -> None:
        """An explicit number density and stoichiometry determine the mass density."""
        sample = models.atomic_scattering(number_density=2.0e22, stoichiometry="C")

        assert sample.density == pytest.approx(
            2.0e22 * sample.formula_mass / 6.02214076e23
        )

    def test_name_defaults_to_formula_when_only_stoichiometry_is_known(self) -> None:
        """A stoichiometric formula supplies the display name when no name is given."""
        sample = models.atomic_scattering(stoichiometry="CH")

        assert sample.name == "CH"
        sample.name = "Film"
        assert sample.name == "Film"

    def test_extended_material_rejects_stoichiometry_changes(self) -> None:
        """An extended sample keeps its original stoichiometry immutable."""
        sample = models.atomic_scattering(stoichiometry="CH", is_extended=True)

        with pytest.raises(ValueError, match="immutable"):
            sample.stoichiometry = "H2O"
        with pytest.raises(ValueError, match="immutable"):
            del sample.stoichiometry

    def test_copy_preserves_properties_without_sharing_mutable_formula_state(
        self,
    ) -> None:
        """Copying keeps material properties while giving the copy its own stoichiometry."""
        sample = models.atomic_scattering(
            name="Sample", density=1.2, stoichiometry="CH"
        )

        clone = sample.copy()
        clone.stoichiometry = "H2O"

        assert clone is not sample
        assert sample.stoichiometry == "CH"
        assert clone.stoichiometry == "H2O"
        assert clone.name == sample.name
