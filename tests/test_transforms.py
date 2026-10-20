"""
Tests for the `kk_transforms` module.

This module tests basic functions that ought to match consistency of the Kramers-Kronig relations.
The tested example functions are listed in `Hilbert Transforms` table from wikipedia:
https://en.wikipedia.org/wiki/Hilbert_transform#Table_of_selected_Hilbert_transforms
"""

import numpy as np
import pytest

import kkcalc2 as kkc


class TestKKTransforms:
    """
    Test the `kk_transforms` module.

    This class contains tests for the Kramers-Kronig transforms.
    """

    class KK_Functions:
        """
        Test functions for the Kramers-Kronig transforms.

        This class contains test functions for the Kramers-Kronig transforms.
        """

    def test_kk_transform_trigonometric(self):
        """
        Test the `kk_transforms.kk_transform` function.

        Use a cosine to test the simple kronig-kramers transform.
        """

        # Setup the function
        f = 1  # Hz
        omega = 2 * np.pi * f  # 1 Hz
        x1 = np.linspace(2 * f, 10 * f, 1000)  # Cannot perform transform at 0.
        f1 = np.cos(omega * x1)

        # Convert the discrete function to a continuous function
        f1_poly = kkc.conversions.ASF_to_ASP(x1, f1)

        # Create a Kramers-Kronig transform
        f2 = kkc.transforms.KK_PP(x1, x1, f1_poly, 0)

        # Get the expected result (phase shift of -pi/2)
        f2_expected = np.cos(omega * x1 - np.pi / 2)

        # Compare values within a subset to avoid edge effects
        idx_subset = (x1 > np.pi * f * 4) & (x1 < np.pi * f * 8)
        f2_sub_transform = f2[idx_subset]
        f2_sub_expected = f2_expected[idx_subset]

        assert np.allclose(f2_sub_transform, f2_sub_expected, atol=1e-6), (
            "Kramers-Kronig transform of cosine did not match expected result."
        )

    def test_kk_transform_round_trip_with_relativistic_correction(self):
        """
        Test that `KK_PP` followed by `KK_PP_inv` recovers the original data.

        Regression test for a bug where `KK_PP_inv` leaked the `relativistic_correction`
        into the result as a spurious `-target_energies * relativistic_correction` term
        (instead of only removing it as an additive offset on the real/f1 data), causing the
        inverse transform to grow unboundedly (linearly) with energy instead of recovering
        the original imaginary/f2 data.
        """
        f = 1
        omega = 2 * np.pi * f
        x1 = np.linspace(2 * f, 10 * f, 2000)
        relativistic_correction = 5.0

        # Original 'imaginary' (f2) data.
        f2_true = np.sin(omega * x1)
        f2_poly = kkc.conversions.ASF_to_ASP(x1, f2_true)

        # Forward transform: f2 -> f1 (adds the relativistic correction as an additive offset).
        f1 = kkc.transforms.KK_PP(x1, x1, f2_poly, relativistic_correction)
        f1_poly = kkc.conversions.ASF_to_ASP(x1, f1)

        # Inverse transform: f1 -> f2 (should remove the offset, not reintroduce it scaled by energy).
        f2_recovered = kkc.transforms.KK_PP_inv(
            x1, x1, f1_poly, relativistic_correction
        )

        # Compare values within a subset to avoid edge effects (trim away from the domain endpoints).
        idx_subset = (x1 > 3 * f) & (x1 < 9 * f)
        assert idx_subset.sum() > 0, "Test subset is empty; check the domain bounds."
        assert np.allclose(f2_recovered[idx_subset], f2_true[idx_subset], atol=1e-1), (
            "Round-trip KK_PP -> KK_PP_inv did not recover the original data."
        )

        # The previous bug caused an unbounded, energy-linear growth in error, with a slope
        # approximately equal to `relativistic_correction`; guard against that specific failure
        # mode by checking the residual's linear trend in energy is small in comparison.
        residual = f2_recovered[idx_subset] - f2_true[idx_subset]
        slope, _intercept = np.polyfit(x1[idx_subset], residual, 1)
        assert abs(slope) < 0.5 * relativistic_correction, (
            "Residual error grows linearly with energy, indicating the relativistic "
            "correction is leaking into the inverse transform again."
        )

    def test_general_transform_matches_optimized_default_orders(self) -> None:
        """The general and optimized forward transforms agree for the default five orders."""
        energies = np.linspace(1.0, 20.0, 20)
        target_energies = np.linspace(1.123, 19.877, 37)
        coefs = np.zeros((len(energies) - 1, 5))
        coefs[:, 0] = 0.2
        coefs[:, 1] = 1.0

        optimized = kkc.transforms.KK_PP(
            target_energies, energies, coefs, relativistic_correction=2.0
        )
        general = kkc.transforms.KK_General_PP(
            target_energies,
            energies,
            coefs,
            orders=kkc.transforms.DEF_ORDERS,
            relativistic_correction=2.0,
        )

        np.testing.assert_allclose(general, optimized, rtol=1e-12, atol=1e-12)

    def test_general_transform_applies_relativistic_correction_as_offset(self) -> None:
        """Changing relativistic correction changes each transformed value by that offset."""
        energies = np.array([1.0, 2.0, 4.0])
        targets = np.array([1.3, 2.7, 3.6])
        coefs = np.array([[3.0], [5.0]])

        without_correction = kkc.transforms.KK_General_PP(
            targets, energies, coefs, orders=np.array([0]), relativistic_correction=0.0
        )
        with_correction = kkc.transforms.KK_General_PP(
            targets, energies, coefs, orders=np.array([0]), relativistic_correction=4.5
        )

        np.testing.assert_allclose(with_correction - without_correction, 4.5)

    @pytest.mark.parametrize(
        "coefs,orders,message",
        [
            (np.ones((1, 2)), np.array([1, 0]), "First axis"),
            (np.ones((2, 2)), np.array([1]), "Second axis"),
        ],
    )
    def test_general_transform_rejects_incompatible_coefficient_shapes(
        self, coefs, orders, message
    ) -> None:
        """General transforms require one coefficient row per interval and order per column."""
        with pytest.raises(ValueError, match=message):
            kkc.transforms.KK_General_PP(
                np.array([1.3]),
                np.array([1.0, 2.0, 3.0]),
                coefs,
                orders=orders,
            )

    def test_accuracy_improvement_with_zero_iterations_preserves_inputs(self) -> None:
        """A zero iteration limit returns the original energy and real-factor arrays unchanged."""
        energies = np.array([1.0, 2.0, 3.0])
        real_factors = np.array([4.0, 5.0, 6.0])
        coefs = np.zeros((2, 5))

        result_energies, result_factors = kkc.transforms.improve_accuracy(
            energies,
            real_factors,
            coefs,
            relativistic_correction=0.0,
            max_iter=0,
        )

        np.testing.assert_array_equal(result_energies, energies)
        np.testing.assert_array_equal(result_factors, real_factors)

    # def test_kk_transform_sq_reciporical(self):
    #     """
    #     Test the `kk_transforms.kk_transform` function.

    #     Use a squared reciporical to test the kronig-kramers transform.
    #     """

    #     # Setup the function
    #     x1 = np.linspace(2, 100, 1000) # Cannot perform transform at 0.
    #     f1 = 1 / (x1 ** 2 + 1)

    #     # Convert the discrete function to a continuous function
    #     f1_poly = kkc.conversions.ASF_to_ASP(x1, f1)

    #     # Create a Kramers-Kronig transform
    #     f2 = kkc.kk_transforms.KK_PP(
    #         x1, x1, f1_poly, 0
    #     )

    #     # Get the expected result
    #     f2_expected = x1 / (x1 ** 2 + 1)
    #     print(f2_expected)
    #     print(f2)

    #     # Compare values within a subset to avoid edge effects
    #     idx_subset = (x1 > 10) & (x1 < 92)
    #     f2_sub_transform = f2[idx_subset]
    #     f2_sub_expected = f2_expected[idx_subset]

    #     assert np.allclose(f2_sub_transform, f2_sub_expected, atol=1e-6), \
    #         "Kramers-Kronig transform of cosine did not match expected result."

    # def test_kk_transform_delta(self):
    #     """
    #     Test the `kk_transforms.kk_transform` function.

    #     Use a delta function to test the kronig-kramers transform.
    #     """
    #     x1 = np.linspace(10, 30, 201)
    #     f1 = np.zeros_like(x1)
    #     f1[10] = 1 # Dirac delta function at x = 20

    #     # Convert the discrete function to a continuous function
    #     f1_poly = kkc.conversions.ASF_to_ASP(x1, f1)

    #     # Create a Kramers-Kronig transform
    #     f2 = kkc.kk_transforms.KK_PP(
    #         x1, x1, f1_poly, 0
    #     )

    #     # Get the expected result
    #     f2_expected = 1 / (np.pi * (x1 - 20))

    #     print(f2_expected)
    #     print(f2)

    #     # Compare values within a subset to avoid edge effects
    #     idx_subset = (x1 > 15) & (x1 < 25)
    #     f2_sub_transform = f2[idx_subset]
    #     f2_sub_expected = f2_expected[idx_subset]
    #     assert np.allclose(f2_sub_transform, f2_sub_expected, atol=1e-6), \
    #         "Kramers-Kronig transform of delta function did not match expected result."


# class kk_transforms_tests():
#     def test_improve_accuracy() -> None:
#         """
#         Test the `kk_transforms.improve_accuracy` function.

#         Use a gaussian to test the simple kronig-kramers transform.
#         """
#         pass
