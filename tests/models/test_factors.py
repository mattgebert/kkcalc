"""
Tests the creation and properties of atomic scattering factor objects (ASF).
"""

import numpy as np
import pytest

from kkcalc2.models.factors import asf, asf_complex, asf_im, asf_re

from .conftest import (
    NEXAFS_MATERIALS,
    NexafsAsf,
    NexafsMaterial,
    NexafsRefractive,
    NexafsRefractiveIndex,
)


class TestFactors:
    pass


class TestAsfInstantiation:
    """Tests the static/class-method instantiation routes for `asf_re`, `asf_im` and `asf_complex`."""

    def test_asf_re_from_asf(self, nexafs_asf: NexafsAsf) -> None:
        """`asf_re.from_asf` designates a generic/undesignated `asf`-like object as real."""
        result = asf_re.from_asf(nexafs_asf.re)
        assert isinstance(result, asf_re)
        assert np.allclose(result.factors, nexafs_asf.re.factors)
        assert np.allclose(result.energies, nexafs_asf.re.energies)

    def test_asf_re_from_deltas(
        self,
        nexafs_material: NexafsMaterial,
        nexafs_asf: NexafsAsf,
        nexafs_refractive: NexafsRefractive,
    ) -> None:
        """`asf_re.from_deltas` reconstructs the real ASF from the dispersive (delta) component."""
        result = asf_re.from_deltas(
            nexafs_asf.re.energies,
            dispersion=nexafs_refractive.real,
            stoichiometry=nexafs_material.stoichiometry,
            density=nexafs_material.density,
        )
        assert np.allclose(result.factors, nexafs_asf.re.factors)

    def test_asf_re_from_refractive_index(
        self,
        nexafs_material: NexafsMaterial,
        nexafs_asf: NexafsAsf,
        nexafs_refractive_index: NexafsRefractiveIndex,
    ) -> None:
        """`asf_re.from_refractive_index` reconstructs the real ASF from the real refractive index component."""
        result = asf_re.from_refractive_index(
            nexafs_asf.re.energies,
            refractive_index=nexafs_refractive_index.real,
            stoichiometry=nexafs_material.stoichiometry,
            density=nexafs_material.density,
        )
        assert np.allclose(result.factors, nexafs_asf.re.factors)

    def test_asf_im_from_asf(self, nexafs_asf: NexafsAsf) -> None:
        """`asf_im.from_asf` designates a generic/undesignated `asf`-like object as imaginary."""
        result = asf_im.from_asf(nexafs_asf.complex.im)
        assert isinstance(result, asf_im)
        assert np.allclose(result.factors, nexafs_asf.complex.im.factors)
        assert np.allclose(result.energies, nexafs_asf.complex.im.energies)

    def test_asf_im_from_NEXAFS(
        self, nexafs_material: NexafsMaterial, nexafs_asf: NexafsAsf
    ) -> None:
        """`asf_im.from_NEXAFS` reconstructs the imaginary ASF from raw NEXAFS data."""
        result = asf_im.from_NEXAFS(
            nexafs_material.energies,
            NEXAFS=nexafs_material.nexafs,
            stoichiometry=nexafs_material.stoichiometry,
            density=nexafs_material.density,
        )
        assert np.allclose(result.factors, nexafs_asf.im.factors)

    def test_asf_im_from_betas(
        self,
        nexafs_material: NexafsMaterial,
        nexafs_asf: NexafsAsf,
        nexafs_refractive: NexafsRefractive,
    ) -> None:
        """`asf_im.from_betas` reconstructs the imaginary ASF from the absorptive (beta) component."""
        result = asf_im.from_betas(
            nexafs_asf.complex.im.energies,
            absorption=nexafs_refractive.imag,
            stoichiometry=nexafs_material.stoichiometry,
            density=nexafs_material.density,
        )
        assert np.allclose(result.factors, nexafs_asf.complex.im.factors)

    def test_asf_im_from_refractive_index(
        self,
        nexafs_material: NexafsMaterial,
        nexafs_asf: NexafsAsf,
        nexafs_refractive_index: NexafsRefractiveIndex,
    ) -> None:
        """`asf_im.from_refractive_index` reconstructs the imaginary ASF from the imaginary refractive index component."""
        result = asf_im.from_refractive_index(
            nexafs_asf.complex.im.energies,
            refractive_index=nexafs_refractive_index.imag,
            stoichiometry=nexafs_material.stoichiometry,
            density=nexafs_material.density,
        )
        assert np.allclose(result.factors, nexafs_asf.complex.im.factors)

    def test_asf_complex_from_asf(
        self, nexafs_material: NexafsMaterial, nexafs_asf: NexafsAsf
    ) -> None:
        """`asf_complex.from_asf` reconstructs a complex ASF from a complex-valued factors array."""
        result = asf_complex.from_asf(
            nexafs_asf.complex.energies,
            nexafs_asf.complex.factors,
            stoichiometry=nexafs_material.stoichiometry,
            density=nexafs_material.density,
        )
        assert np.allclose(result.factors, nexafs_asf.complex.factors)

    def test_asf_complex_from_NEXAFS(self, nexafs_material: NexafsMaterial) -> None:
        """`asf_complex.from_NEXAFS` reconstructs a complex ASF from complex-valued NEXAFS data."""
        # Use a small subset: the default `improve_accuracy=True` KK transform used internally
        # by this classmethod is not exposed, and is impractically slow/memory-hungry on the
        # full, noisy example datasets.
        energies = nexafs_material.energies[:8]
        nexafs = nexafs_material.nexafs[:8].astype(complex)

        result = asf_complex.from_NEXAFS(
            energies,
            NEXAFS=nexafs,
            stoichiometry=nexafs_material.stoichiometry,
            density=nexafs_material.density,
        )
        assert isinstance(result, asf_complex)
        assert np.allclose(result.re.energies, result.im.energies)
        assert np.allclose(result.im.factors.imag, 0)
        # Should reproduce the same imaginary factors as a direct `asf_im.from_NEXAFS` conversion.
        expected_im = asf_im.from_NEXAFS(
            energies,
            NEXAFS=nexafs.imag,
            stoichiometry=nexafs_material.stoichiometry,
            density=nexafs_material.density,
        )
        assert np.allclose(result.im.factors, expected_im.factors)

    def test_asf_complex_from_refractive(self, nexafs_material: NexafsMaterial) -> None:
        """`asf_complex.from_refractive` reconstructs a complex ASF from a complex refractive value."""
        energies = nexafs_material.energies[:8]
        im = asf_im.from_NEXAFS(
            energies,
            NEXAFS=nexafs_material.nexafs[:8],
            stoichiometry=nexafs_material.stoichiometry,
            density=nexafs_material.density,
        )
        expected = im.calculate_complex_factors(improve_accuracy=False)

        result = asf_complex.from_refractive(
            energies,
            refractive=expected.refractive,
            stoichiometry=nexafs_material.stoichiometry,
            density=nexafs_material.density,
        )
        assert np.allclose(result.factors, expected.factors)

    def test_asf_complex_from_refractive_index(
        self, nexafs_material: NexafsMaterial
    ) -> None:
        """`asf_complex.from_refractive_index` reconstructs a complex ASF from a complex refractive index."""
        energies = nexafs_material.energies[:8]
        im = asf_im.from_NEXAFS(
            energies,
            NEXAFS=nexafs_material.nexafs[:8],
            stoichiometry=nexafs_material.stoichiometry,
            density=nexafs_material.density,
        )
        expected = im.calculate_complex_factors(improve_accuracy=False)
        refractive_index = 1 - expected.refractive.real + 1j * expected.refractive.imag

        result = asf_complex.from_refractive_index(
            energies,
            refractive_index=refractive_index,
            stoichiometry=nexafs_material.stoichiometry,
            density=nexafs_material.density,
        )
        assert np.allclose(result.factors, expected.factors)


class TestAsfBaseInstantiation:
    """Tests the (partially abstract) generic `asf` base class instantiation routes."""

    def test_asf_base_cannot_be_instantiated(self) -> None:
        """`asf` is abstract (via the unimplemented `from_refractive_index`) and cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            asf(energies=np.array([1.0, 2.0]), factors=np.array([1.0, 2.0]))

    def test_asf_from_refractive_index_not_implemented(
        self,
        nexafs_material: NexafsMaterial,
        nexafs_refractive_index: NexafsRefractiveIndex,
    ) -> None:
        """`asf.from_refractive_index` is deliberately unimplemented, as the conversion is ambiguous."""
        with pytest.raises(NotImplementedError):
            asf.from_refractive_index(
                nexafs_material.energies,
                refractive_index=nexafs_refractive_index.real,
                stoichiometry=nexafs_material.stoichiometry,
                density=nexafs_material.density,
            )


class TestAsfCopy:
    """Tests the `copy` method reproduces equivalent, independently allocated objects."""

    def test_asf_re_copy(self, nexafs_asf: NexafsAsf) -> None:
        """`asf_re.copy` reproduces the same factors with a new memory allocation."""
        result = nexafs_asf.re.copy()

        assert isinstance(result, asf_re)
        assert result is not nexafs_asf.re
        assert result.factors is not nexafs_asf.re.factors
        assert np.allclose(result.factors, nexafs_asf.re.factors)
        assert np.allclose(result.energies, nexafs_asf.re.energies)

    def test_asf_im_copy(self, nexafs_asf: NexafsAsf) -> None:
        """`asf_im.copy` reproduces the same factors with a new memory allocation."""
        result = nexafs_asf.im.copy()

        assert isinstance(result, asf_im)
        assert result is not nexafs_asf.im
        assert np.allclose(result.factors, nexafs_asf.im.factors)
        assert np.allclose(result.energies, nexafs_asf.im.energies)

    def test_asf_complex_copy(self, nexafs_asf: NexafsAsf) -> None:
        """`asf_complex.copy` reproduces the same real/imaginary factors independently."""
        result = nexafs_asf.complex.copy()

        assert isinstance(result, asf_complex)
        assert result is not nexafs_asf.complex
        assert np.allclose(result.factors, nexafs_asf.complex.factors)
        assert np.allclose(result.energies, nexafs_asf.complex.energies)


class TestKKTransformRoundTrip:
    """
    Regression tests for `asf_im.kk_transform`/`asf_re.kk_transform_inv` round-trip consistency.

    Mirrors the `kkcalc2.__main__`/GUI example flow: load NEXAFS data for Polystyrene, perform a
    KK transform (imaginary -> real) then an inverse KK transform (real -> imaginary) with
    `improve_accuracy=False` (as used for non-extended data in the GUI's `transform()` handler),
    and check that the round-tripped data matches the original away from the domain endpoints.

    This guards against a bug where `transforms.KK_PP_inv` leaked the `relativistic_correction`
    into the result as a spurious `-energy * relativistic_correction` term, causing the inverse
    transform to diverge (unboundedly, growing with energy) instead of recovering the original data.
    """

    def test_kk_transform_round_trip_polystyrene(self) -> None:
        """Round-tripping Polystyrene example data through KK transform/inverse recovers the original."""
        material = NEXAFS_MATERIALS["PS_C"]
        im = asf_im.from_NEXAFS(
            energies=material.energies,
            NEXAFS=material.nexafs,
            name=material.name,
            stoichiometry=material.stoichiometry,
            scale_to_database=True,
        )

        re = im.kk_transform(improve_accuracy=False)
        im2 = re.kk_transform_inv(improve_accuracy=False)

        # Truncate the first/last two datapoints at each end, where edge effects dominate.
        trim = slice(2, -2)
        original = im.factors[trim]
        round_tripped = im2.factors[trim]
        energies = im.energies[trim]

        # The two representations should be strongly correlated (same overall shape/trend).
        correlation = np.corrcoef(original, round_tripped)[0, 1]
        assert correlation > 0.9, (
            f"Round-tripped imaginary factors do not correlate with the original data (r={correlation:.3f})."
        )

        # No individual point should be wildly (order-of-magnitude) off from the original scale.
        assert np.max(np.abs(round_tripped - original)) < 10 * np.mean(
            np.abs(original)
        ), "Round-tripped imaginary factors deviate far beyond the expected data scale."

        # Guard against the specific historical bug: the residual should not grow linearly with
        # energy at a slope comparable to the material's relativistic correction.
        residual = round_tripped - original
        slope, _intercept = np.polyfit(energies, residual, 1)
        relativistic_correction = im.stoichiometry.relativistic_correction
        assert abs(slope) < 0.5 * relativistic_correction, (
            "Residual error grows linearly with energy, indicating the relativistic "
            "correction is leaking into the inverse transform again."
        )
