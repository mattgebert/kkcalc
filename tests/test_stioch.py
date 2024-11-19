"""Tests for the `stoich` module."""

import pytest
import matplotlib.pyplot as plt
import numpy as np

from kkcalc import stoichiometry as kk_stoich


def stoichiometry() -> kk_stoich:
    """
    Test the `stoichiometry` class.
    """
    return kk_stoich("C9H12O6S2")


def test_stiochiometry() -> None:
    """
    Test the `stoichiometry` class.
    """
    obj = stoichiometry()
    assert str(obj) == "C9H12O6S2"
