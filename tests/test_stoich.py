"""Tests for the `stoich` module."""

import pytest
import matplotlib.pyplot as plt
import numpy as np

from kkcalc import stoichiometry as kk_stoich


class basic_stoichiomety:
    """Basic functions for creating and testing `stoichiometry` objects."""

    # Define chemical formulae for testing
    POLYMER_PS: str = "C8H8"
    """Polystyrene (PS) chemical formula."""
    POLYMER_P3HT: str = "C10H14S"
    """Poly(3-hexylthiophene-2,5-diyl) (P3HT) chemical formula."""
    POLYMER_PMMA: str = "C5H8O2"
    """Poly(methyl methacrylate) (PMMA) chemical formula."""

    def basic_PS(self) -> kk_stoich:
        """
        Creates a simple integer `stoichiometry` of polystyrene.
        """
        return kk_stoich(self.POLYMER_PS)

    def basic_PMMA(self) -> kk_stoich:
        """
        Creates a simple integer `stoichiometry` of PMMA.
        """
        return kk_stoich(self.POLYMER_PMMA)

    def basic_P3HT(self) -> kk_stoich:
        """
        Creates a simple integer `stoichiometry` of P3HT.
        """
        return kk_stoich(self.POLYMER_P3HT)


class TestStoichiometryInit(basic_stoichiomety):
    """Tests for the creation of integer `stoichiometry` objects."""

    def test_stiochiometry(self) -> None:
        """
        Tests the creation of a simple integer `stiochiometry`.
        """
        obj = self.basic_PS()
        assert str(obj) == "C8H8"


class fractional_stoichiometry(basic_stoichiomety):
    """Basic functions for creating and testing fractional `stoichiometry` objects."""

    # Define fractional chemical formulae
    """Poly(methyl methacrylate) (PMMA) chemical formula."""
    BLOCK_COPOLYMER_PS_PMMA: str = (
        f"({basic_stoichiomety.POLYMER_PS})1.0({basic_stoichiomety.POLYMER_PMMA})1.2"
    )

    def copolymer_PS_PMMA(self) -> kk_stoich:
        """
        Creates a fractional `stiochiometry`, combining Polystyrene and PMMA stoichiometries.
        """
        return kk_stoich(self.BLOCK_COPOLYMER_PS_PMMA)


class TestStoichiometryFractional(fractional_stoichiometry):
    """Tests for the creation of fractional `stoichiometry` objects."""

    def test_copolymer(self) -> None:
        """
        Tests the creation of a fractional `stoichiometry`.
        """
        obj = self.copolymer_PS_PMMA()
        assert str(obj) == "C14H17.6O2.4"


class TestStoichometryOperations(fractional_stoichiometry):
    """Tests for the numerical and functional operators of `stoichiometry` objects."""

    def test_stiochiometry_numeric(self) -> None:
        """
        Tests the addition, multiplication and division of stoichiometry objects
        """
        # Create the integer components
        pmma = self.basic_PMMA()
        p3ht = self.basic_P3HT()
        ps = self.basic_PS()

        # Create the block
        block = self.copolymer_PS_PMMA()

        # Test numeric operations
        assert 0.5 * pmma == kk_stoich(f"({self.POLYMER_PMMA})0.5")
        assert (1.0 * ps + 1.2 * pmma) == block

        # Test division
        assert p3ht / 2 == kk_stoich(f"({self.POLYMER_P3HT})0.5")
        assert (p3ht // 2) == "C5H7"

    def test_stoichiometry_equivalence(self) -> None:
        """
        Tests the equivalence of various forms of `stiochiometry` objects.
        """
        #
        ps = self.basic_PS()

        # Test equivalence operations
        assert ps == "C8H8"
        assert "C8H8" == ps
        assert ps == ps.copy()
        assert ps == 1.0 * ps
        assert ps != "C9H9"

    def test_stoichiometry_len(self):
        """
        Tests the atomic length of a `stoichiometry` object.
        """
        # Create the integer components
        pmma = self.basic_PMMA()
        p3ht = self.basic_P3HT()
        ps = self.basic_PS()

        # Create the block
        block = self.copolymer_PS_PMMA()

        # Test length
        assert len(pmma) == 15
        assert len(p3ht) == 25
        assert len(ps) == 16
        assert len(block) == 33
