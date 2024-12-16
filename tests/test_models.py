"""
Model tests for polynomial and factor representations.
"""

import pytest
import warnings
from kkcalc import models

# from test_stoich import fractional_stoichs as fs


class TestCommon:
    @pytest.mark.parametrize(
        "kwargs, msgs",
        [
            (dict(), []),
            (dict(name="Sample"), []),
            (dict(name="Sample", number_density=1.1), []),
            (dict(name="Sample", number_density=1.1, stoichiometry="CH"), []),
            (
                dict(
                    name="Sample",
                    number_density=1.1,
                    stoichiometry="CH",
                    density=1.8,
                ),
                ["Competing information"],
            ),
        ],
    )
    def test_instantiation_atomic_scattering(
        self, kwargs: dict, msgs: list[str]
    ) -> None:
        """Tests the creation of an `atomic_scattering` object, with expected errors."""
        # Create the object
        with warnings.catch_warnings(record=True) as w:
            atomic_scattering = models.atomic_scattering(**kwargs)

        # Check each msg is included by at least one warning
        for msg in msgs:
            assert any(msg in str(warn.message) for warn in w)
