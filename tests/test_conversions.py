"""Tests for the conversions module."""

import numpy as np
import pytest

from kkcalc2 import conversions


class TestConversions:
    """Tests numeric conversion round trips and validation of polynomial inputs."""

    def test_energy_and_wavelength_conversions_are_inverse(self) -> None:
        """Photon energies convert to wavelengths in Angstroms using the physical relation."""
        energies = np.array([100.0, 500.0, 1000.0])

        wavelengths = conversions.energy_to_wavelength(energies)

        np.testing.assert_allclose(
            wavelengths,
            1.2398419843320026e4 / energies,
            rtol=1e-12,
        )
        assert np.isscalar(conversions.energy_to_wavelength(500.0))

    @pytest.mark.parametrize(
        "energies,factors",
        [
            (np.array([100.0, 200.0, 500.0]), np.array([1.0, 2.5, 4.0])),
            (500.0, 2.5),
        ],
    )
    def test_nexafs_and_asf_are_inverse_conversions(self, energies, factors) -> None:
        """Converting NEXAFS to ASF and back recovers scalar and array input values."""
        nexafs = conversions.ASF_to_NEXAFS(energies, factors)

        recovered = conversions.NEXAFS_to_ASF(energies, nexafs)

        np.testing.assert_allclose(recovered, factors)

    def test_refractive_and_asf_conversions_round_trip_complex_data(self) -> None:
        """Complex refractive components round-trip through ASF at fixed number density."""
        energies = np.array([100.0, 250.0, 500.0])
        refractive_component = np.array([1e-5 + 2e-6j, 3e-5 + 4e-6j, 5e-5 + 7e-6j])

        factors = conversions.refractive_to_ASF(
            energies, refractive_component, number_density=1.2e21
        )
        recovered = conversions.ASF_to_refractive(
            energies, factors, number_density=1.2e21
        )

        np.testing.assert_allclose(recovered, refractive_component)

    def test_refractive_conversion_requires_density_information(self) -> None:
        """Refractive-to-ASF conversion reports missing material density inputs."""
        with pytest.raises(ValueError, match="No material density information"):
            conversions.refractive_to_ASF(500.0, 1e-5)
        with pytest.raises(ValueError, match="formula_mass.*stoichiometry"):
            conversions.refractive_to_ASF(500.0, 1e-5, density=1.0)

    def test_polynomial_conversion_round_trips_linear_factors_with_custom_orders(
        self,
    ) -> None:
        """Linear ASF samples round-trip through coefficients with explicit polynomial orders."""
        energies = np.array([100.0, 200.0, 400.0])
        factors = np.array([2.0, 5.0, 11.0])
        orders = np.array([1, 0, -1])

        coefs = conversions.ASF_to_ASP(energies, factors, N=len(orders))
        recovered = conversions.ASP_to_ASF(energies, coefs, orders=orders)

        np.testing.assert_allclose(recovered, factors)
        assert coefs.shape == (len(energies) - 1, len(orders))

    @pytest.mark.parametrize(
        "energies",
        [
            np.array([1.0, 1.0, 2.0]),
            np.array([2.0, 1.0, 3.0]),
        ],
    )
    def test_polynomial_conversion_rejects_unordered_energy_boundaries(
        self, energies
    ) -> None:
        """ASP coefficient generation requires unique, strictly increasing energies."""
        with pytest.raises(ValueError, match="ordered and unique"):
            conversions.ASF_to_ASP(energies, np.array([1.0, 2.0, 3.0]))

    def test_polynomial_evaluation_rejects_order_coefficient_count_mismatch(
        self,
    ) -> None:
        """ASP evaluation requires one declared order for each coefficient column."""
        with pytest.raises(ValueError, match="Number of orders"):
            conversions.ASP_to_ASF(
                100.0,
                np.array([1.0, 2.0]),
                orders=np.array([1]),
            )
