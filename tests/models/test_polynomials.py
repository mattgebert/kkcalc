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
