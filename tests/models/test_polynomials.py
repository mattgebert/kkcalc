"""
Tests the creation and properties of atomic scattering polynomial objects (ASP).
"""

import numpy as np
import pytest

from kkcalc2.models.polynomials import asp, asp_complex, asp_im, asp_re

from .conftest import NexafsAsf


class TestPolynomials:
    pass


class TestAspInstantiation:
    """Tests the static/class-method instantiation routes for `asp_im` and `asp_re`."""

    def test_asp_im_from_asp(self, nexafs_asf: NexafsAsf) -> None:
        """`asp_im.from_asp` designates a generic/undesignated `asp` object as imaginary."""
        im_poly = nexafs_asf.im.to_ASP()
        generic = asp(
            energies=im_poly.energies,
            coefs=im_poly.coefs,
            orders=im_poly.orders,
            **im_poly._properties_dict,
        )

        result = asp_im.from_asp(generic)

        assert isinstance(result, asp_im)
        assert np.allclose(result.coefs, im_poly.coefs)
        assert np.allclose(result.energies, im_poly.energies)

    def test_asp_re_from_asp(self, nexafs_asf: NexafsAsf) -> None:
        """`asp_re.from_asp` designates a generic/undesignated `asp` object as real."""
        re_poly = nexafs_asf.re.to_ASP()
        generic = asp(
            energies=re_poly.energies,
            coefs=re_poly.coefs,
            orders=re_poly.orders,
            **re_poly._properties_dict,
        )

        result = asp_re.from_asp(generic)

        assert isinstance(result, asp_re)
        assert np.allclose(result.coefs, re_poly.coefs)
        assert np.allclose(result.energies, re_poly.energies)


class TestAspCopy:
    """Tests the `copy` method reproduces equivalent, independently allocated objects."""

    def test_asp_im_copy(self, nexafs_asf: NexafsAsf) -> None:
        """`asp_im.copy` reproduces the same coefficients with a new memory allocation."""
        im_poly = nexafs_asf.im.to_ASP()
        result = im_poly.copy()

        assert isinstance(result, asp_im)
        assert result is not im_poly
        assert result.coefs is not im_poly.coefs
        assert np.allclose(result.coefs, im_poly.coefs)
        assert np.allclose(result.energies, im_poly.energies)

    def test_asp_complex_copy(self, nexafs_asf: NexafsAsf) -> None:
        """`asp_complex.copy` reproduces the same real/imaginary coefficients independently."""
        complex_poly = nexafs_asf.complex.to_ASP()
        result = complex_poly.copy()

        assert isinstance(result, asp_complex)
        assert result is not complex_poly
        assert np.allclose(result.coefs, complex_poly.coefs)
        assert np.allclose(result.energies, complex_poly.energies)


class TestAspScalarArray:
    """
    Tests that per-energy evaluation methods return a scalar for scalar (singular) input,
    and an array for array-like input, of a matching length.
    """

    def test_call_scalar_vs_array(self, nexafs_asf: NexafsAsf) -> None:
        """`asp.__call__`/`eval_asf` returns a scalar for a scalar energy, and an array otherwise."""
        poly = nexafs_asf.im.to_ASP()
        energy = float(poly.energies[1])

        scalar_result = poly(energy)
        array_result = poly(np.array([energy, energy]))

        assert np.ndim(scalar_result) == 0
        assert isinstance(array_result, np.ndarray)
        assert array_result.shape == (2,)
        assert np.allclose(scalar_result, array_result)

    def test_eval_betas_scalar_vs_array(self, nexafs_asf: NexafsAsf) -> None:
        """`asp_im.eval_betas` returns a scalar for a scalar energy, and an array otherwise."""
        poly = nexafs_asf.im.to_ASP()
        energies = poly.energies[:3]

        scalar_result = poly.eval_betas(float(energies[0]))
        array_result = poly.eval_betas(energies)

        assert np.ndim(scalar_result) == 0
        assert isinstance(array_result, np.ndarray)
        assert array_result.shape == energies.shape
        assert np.isclose(scalar_result, array_result[0])

    def test_eval_deltas_scalar_vs_array(self, nexafs_asf: NexafsAsf) -> None:
        """`asp_re.eval_deltas` returns a scalar for a scalar energy, and an array otherwise."""
        poly = nexafs_asf.complex.re.to_ASP()
        energies = poly.energies[:3]

        scalar_result = poly.eval_deltas(float(energies[0]))
        array_result = poly.eval_deltas(energies)

        assert np.ndim(scalar_result) == 0
        assert isinstance(array_result, np.ndarray)
        assert array_result.shape == energies.shape
        assert np.isclose(scalar_result, array_result[0])

    def test_eval_NEXAFS_scalar_vs_array(self, nexafs_asf: NexafsAsf) -> None:
        """`asp_im.eval_NEXAFS` returns a scalar for a scalar energy, and an array otherwise."""
        poly = nexafs_asf.im.to_ASP()
        energies = poly.energies[:3]

        scalar_result = poly.eval_NEXAFS(float(energies[0]))
        array_result = poly.eval_NEXAFS(energies)

        assert np.ndim(scalar_result) == 0
        assert isinstance(array_result, np.ndarray)
        assert array_result.shape == energies.shape
        assert np.isclose(scalar_result, array_result[0])

    def test_attenuation_length_scalar_vs_array(self, nexafs_asf: NexafsAsf) -> None:
        """`asp_im.attenuation_length` returns a scalar for a scalar energy, and an array otherwise."""
        poly = nexafs_asf.im.to_ASP()
        energies = poly.energies[:3]

        scalar_result = poly.attenuation_length(float(energies[0]))
        array_result = poly.attenuation_length(energies)

        assert np.ndim(scalar_result) == 0
        assert isinstance(array_result, np.ndarray)
        assert array_result.shape == energies.shape
        assert np.isclose(scalar_result, array_result[0])

    def test_critical_angle_scalar_vs_array(self, nexafs_asf: NexafsAsf) -> None:
        """`asp_re.critical_angle` returns a scalar for a scalar energy, and an array otherwise."""
        poly = nexafs_asf.complex.re.to_ASP()
        energies = poly.energies[:3]

        scalar_result = poly.critical_angle(float(energies[0]))
        array_result = poly.critical_angle(energies)

        assert np.ndim(scalar_result) == 0
        assert isinstance(array_result, np.ndarray)
        assert array_result.shape == energies.shape

    def test_asp_complex_attenuation_length_and_critical_angle(
        self, nexafs_asf: NexafsAsf
    ) -> None:
        """`asp_complex.attenuation_length`/`critical_angle` delegate correctly, preserving scalar/array shape."""
        complex_poly = nexafs_asf.complex.to_ASP()
        energies = complex_poly.energies[:3]

        scalar_att = complex_poly.attenuation_length(float(energies[0]))
        array_att = complex_poly.attenuation_length(energies)
        assert np.ndim(scalar_att) == 0
        assert isinstance(array_att, np.ndarray)
        assert array_att.shape == energies.shape

        scalar_angle = complex_poly.critical_angle(float(energies[0]))
        array_angle = complex_poly.critical_angle(energies)
        assert np.ndim(scalar_angle) == 0
        assert isinstance(array_angle, np.ndarray)
        assert array_angle.shape == energies.shape

    @pytest.mark.parametrize("n_points", [1, 3])
    def test_call_array_length_matches_input(
        self, nexafs_asf: NexafsAsf, n_points: int
    ) -> None:
        """Array input of any length (including length 1) returns an array of the same length."""
        poly = nexafs_asf.im.to_ASP()
        energies = poly.energies[:n_points]

        result = poly(energies)

        assert isinstance(result, np.ndarray)
        assert result.shape == (n_points,)


class TestAspArithmatic:
    """Tests that arithmetic operations on `asp` objects return the expected types."""

    def test_addition_returns_asp(self, nexafs_asf: NexafsAsf) -> None:
        """Adding two `asp` objects returns a new `asp` object."""
        poly1 = nexafs_asf.im.to_ASP()
        poly2 = nexafs_asf.im.to_ASP()

        result = poly1 + poly2

        assert isinstance(result, asp_im)
        assert np.allclose(result.coefs, poly1.coefs + poly2.coefs)

    def test_subtraction_returns_asp(self, nexafs_asf: NexafsAsf) -> None:
        """Subtracting two `asp` objects returns a new `asp` object."""
        poly1 = nexafs_asf.im.to_ASP()
        poly2 = nexafs_asf.im.to_ASP()

        result = poly1 - poly2

        assert isinstance(result, asp_im)
        assert np.allclose(result.coefs, poly1.coefs - poly2.coefs)

    def test_addition_returns_asp_with_num(self, nexafs_asf: NexafsAsf) -> None:
        """Adding a number to an `asp` objects returns a new `asp` object."""
        poly1 = nexafs_asf.im.to_ASP()
        num = 1.0  # A float number to add to the polynomial

        result = poly1 + num
        assert isinstance(result, asp_im)
        modified_coefs = poly1.coefs.copy()
        modified_coefs[:, 1] += num  # Add the number to the constant term
        assert np.allclose(result.coefs, modified_coefs)

    def test_subtraction_returns_asp_with_num(self, nexafs_asf: NexafsAsf) -> None:
        """Subtracting a number from an `asp` objects returns a new `asp` object."""
        poly1 = nexafs_asf.im.to_ASP()
        num = 1.0  # A float number to subtract from the polynomial

        result = poly1 - num

        assert isinstance(result, asp_im)

        modified_coefs = poly1.coefs.copy()
        modified_coefs[:, 1] -= num  # Subtract the number from the constant term
        assert np.allclose(result.coefs, modified_coefs)

    def test_invalid_addition_raises(self, nexafs_asf: NexafsAsf) -> None:
        """Adding an `asp` object to a non-`asp` object raises a TypeError."""
        poly = nexafs_asf.im.to_ASP()
        non_poly = "not an asp object"  # A string, not an asp object

        with pytest.raises(TypeError):
            _ = poly + non_poly

    def test_invalid_subtraction_raises(self, nexafs_asf: NexafsAsf) -> None:
        """Subtracting a non-`asp` object from an `asp` object raises a TypeError."""
        poly = nexafs_asf.im.to_ASP()
        non_poly = "not an asp object"  # An integer, not an asp object

        with pytest.raises(TypeError):
            _ = poly - non_poly

    def test_im_re_invalid_addition_raises(self, nexafs_asf: NexafsAsf) -> None:
        """Adding an `asp_im` object to an `asp_re` object raises a TypeError."""
        im_poly = nexafs_asf.im.to_ASP()
        re_poly = nexafs_asf.re.to_ASP()

        with pytest.raises(TypeError):
            _ = im_poly + re_poly

    def test_im_re_invalid_subtraction_raises(self, nexafs_asf: NexafsAsf) -> None:
        """Subtracting an `asp_re` object from an `asp_im` object raises a TypeError."""
        im_poly = nexafs_asf.im.to_ASP()
        re_poly = nexafs_asf.re.to_ASP()

        with pytest.raises(TypeError):
            _ = im_poly - re_poly


class TestAspPolynomialOrders:
    """Tests order defaults, coefficient remapping, and order-related errors."""

    @staticmethod
    def make_poly(
        orders: list[int] | None = None,
        coefs: list[list[float]] | None = None,
    ) -> asp_im:
        """Build a small, deterministic polynomial for order-handling tests."""
        if coefs is None:
            coefs = [[2.0, 3.0], [4.0, 5.0]]
        return asp_im(
            energies=[1.0, 2.0, 3.0],
            coefs=coefs,
            orders=orders,
        )

    def test_default_orders_are_used_for_scalar_arithmetic(self) -> None:
        """Unspecified orders use the default constant column for scalar operations."""
        poly = self.make_poly()

        result = poly + 7.0

        assert poly.orders is None
        assert result.orders is None
        np.testing.assert_allclose(result.coefs, [[2.0, 10.0], [4.0, 12.0]])
        np.testing.assert_allclose(poly.coefs, [[2.0, 3.0], [4.0, 5.0]])

    def test_arithmetic_matches_equivalent_orders_by_power(self) -> None:
        """Addition aligns coefficients when the same powers appear in a different order."""
        left = self.make_poly(orders=[1, 0], coefs=[[2.0, 3.0], [4.0, 5.0]])
        right = self.make_poly(orders=[0, 1], coefs=[[7.0, 11.0], [13.0, 17.0]])

        result = left + right

        assert result.orders.tolist() == [1, 0]
        np.testing.assert_allclose(result.coefs, [[13.0, 10.0], [21.0, 18.0]])

    def test_arithmetic_rejects_nonzero_unrepresentable_order(self) -> None:
        """An operand with a nonzero power absent from the target cannot be re-expressed."""
        target = self.make_poly(orders=[1, 0])
        incompatible = self.make_poly(orders=[2, 0])

        with pytest.raises(ValueError, match="Cannot express order 2"):
            _ = target + incompatible

    @pytest.mark.parametrize("operation", ["add", "subtract"])
    def test_scalar_arithmetic_requires_a_constant_term(self, operation: str) -> None:
        """Scalar arithmetic fails clearly when the polynomial has no zero-power term."""
        poly = self.make_poly(orders=[1, -1])

        with pytest.raises(ValueError, match="zero-order coefficient"):
            if operation == "add":
                _ = poly + 1.0
            else:
                _ = poly - 1.0

    def test_coefficient_setter_checks_order_count(self) -> None:
        """Replacing coefficients cannot silently change the number of polynomial orders."""
        poly = self.make_poly(orders=[1, 0])

        with pytest.raises(ValueError, match="number of orders"):
            poly.coefs = np.ones((2, 3))

    def test_constructor_rejects_invalid_energy_and_order_shapes(self) -> None:
        """Energy, coefficient, and order arrays must have compatible dimensions."""
        with pytest.raises(ValueError, match="Energies must be a 1D array"):
            asp_im(energies=[[1.0, 2.0]], coefs=[[1.0]])
        with pytest.raises(ValueError, match="Coefficients must be a 2D array"):
            asp_im(energies=[1.0, 2.0], coefs=[1.0])
        with pytest.raises(ValueError, match="Orders must be a 1D array"):
            asp_im(energies=[1.0, 2.0], coefs=[[1.0]], orders=[[0]])
        with pytest.raises(ValueError, match="Number of orders"):
            asp_im(energies=[1.0, 2.0], coefs=[[1.0]], orders=[1, 0])


class TestAspEnergyDomains:
    """Tests energy-domain operations using simple known interval coefficients."""

    @staticmethod
    def make_poly() -> asp_im:
        """Create three constant-valued intervals over the domain [1, 4]."""
        return asp_im(
            energies=[1.0, 2.0, 3.0, 4.0],
            coefs=[[10.0], [20.0], [30.0]],
            orders=[0],
        )

    def test_extend_energies_splits_intervals_without_changing_values(self) -> None:
        """Adding an interior boundary copies the coefficient for its source interval."""
        poly = self.make_poly()

        extended = poly.extend_energies(np.array([1.0, 1.5, 2.0, 3.0, 4.0]))

        np.testing.assert_allclose(extended.energies, [1.0, 1.5, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(extended.coefs[:, 0], [10.0, 10.0, 20.0, 30.0])
        assert extended.orders.tolist() == [0]
        assert extended is not poly

    def test_extend_energies_rejects_points_outside_domain(self) -> None:
        """Extension only adds boundaries within the polynomial's existing domain."""
        poly = self.make_poly()

        with pytest.raises(ValueError, match="subset of the existing energies"):
            poly.extend_energies(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))

    def test_truncate_energies_keeps_coefficients_on_requested_domain(self) -> None:
        """Truncation inserts interior boundaries and retains their source interval values."""
        poly = self.make_poly()

        truncated = poly.truncate_energies((1.5, 3.5))

        np.testing.assert_allclose(truncated.energies, [1.5, 2.0, 3.0, 3.5])
        np.testing.assert_allclose(truncated.coefs[:, 0], [10.0, 20.0, 30.0])
        assert truncated.orders.tolist() == [0]

    def test_truncate_energies_rejects_domain_outside_source(self) -> None:
        """Truncation cannot request energies beyond the existing energy bounds."""
        poly = self.make_poly()

        with pytest.raises(ValueError, match="within the existing energies"):
            poly.truncate_energies((0.5, 2.0))

    def test_slice_and_iteration_preserve_segment_boundaries(self) -> None:
        """Indexing selects a segment and iteration yields each interval with its coefficients."""
        poly = self.make_poly()

        second_segment = poly[1]
        selected_segments = poly[1:3]
        iterated_segments = list(poly)

        np.testing.assert_allclose(second_segment.energies, [2.0, 3.0])
        np.testing.assert_allclose(second_segment.coefs[:, 0], [20.0])
        np.testing.assert_allclose(selected_segments.energies, [2.0, 3.0, 4.0])
        np.testing.assert_allclose(selected_segments.coefs[:, 0], [20.0, 30.0])
        assert [interval for interval, _ in iterated_segments] == [
            (1.0, 2.0),
            (2.0, 3.0),
            (3.0, 4.0),
        ]
        for (_, actual_coefs), expected_coefs in zip(
            iterated_segments, ([10.0], [20.0], [30.0]), strict=True
        ):
            np.testing.assert_allclose(actual_coefs, expected_coefs)

    def test_evaluation_includes_domain_endpoints_and_rejects_outside_values(
        self,
    ) -> None:
        """Polynomial evaluation handles both endpoints and rejects out-of-domain energies."""
        poly = self.make_poly()

        np.testing.assert_allclose(poly([1.0, 2.5, 4.0]), [10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="outside the defined energy range"):
            poly(4.1)

    def test_addition_aligns_different_interval_boundaries(self) -> None:
        """Addition splits intervals as needed before summing their coefficients."""
        left = asp_im(
            energies=[1.0, 2.0, 4.0],
            coefs=[[10.0], [20.0]],
            orders=[0],
        )
        right = asp_im(
            energies=[1.0, 3.0, 4.0],
            coefs=[[1.0], [2.0]],
            orders=[0],
        )

        result = left + right

        np.testing.assert_allclose(result.energies, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(result.coefs[:, 0], [11.0, 21.0, 22.0])

    def test_polynomial_to_factor_conversion_uses_requested_energies(self) -> None:
        """Conversion to scattering factors evaluates the polynomial on target energies."""
        poly = self.make_poly()
        target_energies = np.array([1.5, 2.5, 3.5])

        factors = poly.to_atomic_scattering_factors(target_energies=target_energies)

        np.testing.assert_allclose(factors.energies, target_energies)
        np.testing.assert_allclose(factors.factors, [10.0, 20.0, 30.0])

    def test_changing_energy_count_clears_incompatible_coefficients(self) -> None:
        """Changing the number of intervals warns and clears coefficients of the wrong length."""
        poly = self.make_poly()

        with pytest.warns(UserWarning, match="Coefficients set to `None`"):
            poly.energies = np.array([1.0, 2.0])

        assert poly.coefs is None

    def test_dataframe_labels_columns_with_polynomial_orders(self) -> None:
        """The dataframe view names coefficient columns using the explicit orders."""
        pytest.importorskip("pandas")
        poly = asp_im(
            energies=[1.0, 2.0, 3.0],
            coefs=[[10.0, 1.0], [20.0, 2.0]],
            orders=[1, 0],
        )

        frame = poly.dataframe()

        assert list(frame.columns) == ["Energy LB", "Energy UB", "A1", "A0"]
        assert frame.shape == (2, 4)
        np.testing.assert_allclose(frame[["A1", "A0"]].to_numpy(), poly.coefs)
        assert poly.to_pandas().equals(frame)


class TestAspComplexArithmetic:
    """Tests real/imaginary order checks and complex scalar arithmetic."""

    @staticmethod
    def make_complex(orders: list[int] | None = None) -> asp_complex:
        """Build matching real and imaginary polynomials with two coefficients per interval."""
        energies = [1.0, 2.0, 3.0]
        re = asp_re(energies=energies, coefs=[[2.0, 3.0], [4.0, 5.0]], orders=orders)
        im = asp_im(energies=energies, coefs=[[7.0, 11.0], [13.0, 17.0]], orders=orders)
        return asp_complex(re=re, im=im)

    def test_complex_constructor_rejects_mismatched_part_orders(self) -> None:
        """Real and imaginary components must describe the same powers."""
        re = asp_re(energies=[1.0, 2.0], coefs=[[1.0, 2.0]], orders=[1, 0])
        im = asp_im(energies=[1.0, 2.0], coefs=[[1.0, 2.0]], orders=[0, 1])

        with pytest.raises(ValueError, match="same polynomial orders"):
            asp_complex(re=re, im=im)

    def test_real_scalar_changes_only_real_component(self) -> None:
        """A real scalar changes the constant term of the real polynomial only."""
        poly = self.make_complex(orders=[1, 0])

        result = poly + 5.0

        np.testing.assert_allclose(result.re.coefs, [[2.0, 8.0], [4.0, 10.0]])
        np.testing.assert_allclose(result.im.coefs, poly.im.coefs)

    def test_complex_scalar_changes_both_components(self) -> None:
        """A complex scalar changes real and imaginary constant terms independently."""
        poly = self.make_complex(orders=[1, 0])

        result = poly - (3.0 + 2.0j)

        np.testing.assert_allclose(result.re.coefs, [[2.0, 0.0], [4.0, 2.0]])
        np.testing.assert_allclose(result.im.coefs, [[7.0, 9.0], [13.0, 15.0]])

    def test_complex_subtraction_rejects_mismatched_orders(self) -> None:
        """Complex subtraction rejects operands that use different polynomial orders."""
        left = self.make_complex(orders=[1, 0])
        right = self.make_complex(orders=[2, 0])

        with pytest.raises(ValueError, match="Cannot express order 2"):
            _ = left - right

    def test_complex_subtraction_aligns_equivalent_orders(self) -> None:
        """Subtraction remaps coefficients when both operands contain the same powers."""
        left = self.make_complex(orders=[1, 0])
        right_re = asp_re(
            energies=[1.0, 2.0, 3.0],
            coefs=[[3.0, 2.0], [5.0, 4.0]],
            orders=[0, 1],
        )
        right_im = asp_im(
            energies=[1.0, 2.0, 3.0],
            coefs=[[11.0, 7.0], [17.0, 13.0]],
            orders=[0, 1],
        )
        right = asp_complex(re=right_re, im=right_im)

        result = left - right

        np.testing.assert_allclose(result.re.coefs, [[0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(result.im.coefs, [[0.0, 0.0], [0.0, 0.0]])

    def test_complex_addition_sums_each_component(self) -> None:
        """Adding complex polynomials independently sums matching real and imaginary coefficients."""
        left = self.make_complex(orders=[1, 0])
        right = self.make_complex(orders=[1, 0])

        result = left + right

        np.testing.assert_allclose(result.re.coefs, [[4.0, 6.0], [8.0, 10.0]])
        np.testing.assert_allclose(result.im.coefs, [[14.0, 22.0], [26.0, 34.0]])

    def test_complex_subtraction_subtracts_each_component(self) -> None:
        """Subtracting complex polynomials independently subtracts both components."""
        left = self.make_complex(orders=[1, 0])
        right = self.make_complex(orders=[1, 0])

        result = left - right

        np.testing.assert_allclose(result.re.coefs, np.zeros((2, 2)))
        np.testing.assert_allclose(result.im.coefs, np.zeros((2, 2)))

    def test_complex_arithmetic_rejects_non_complex_operands(self) -> None:
        """Complex polynomial arithmetic rejects unrelated object types with TypeError."""
        poly = self.make_complex(orders=[1, 0])

        with pytest.raises(TypeError, match="must be of type `asp_complex`"):
            _ = poly + "not a polynomial"
        with pytest.raises(TypeError, match="must be of type `asp_complex`"):
            _ = poly - "not a polynomial"
