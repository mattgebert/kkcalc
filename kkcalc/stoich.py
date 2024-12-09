"""
Module to define classes and properties relating to the chemical composition of materials.

Includes
"""

import re

has_periodictable: bool
"""Flag to indicate if the periodictable module is available."""
try:
    import periodictable as pt
    from periodictable.formulas import Formula as Formula
    from periodictable.core import Element as Element

    has_periodictable = True
except ImportError:
    has_periodictable = False

import numpy as np
import numpy.typing as npt
from typing import Self, TYPE_CHECKING, TypeAlias, Iterable
from kkcalc.util import doc_copy

if TYPE_CHECKING:
    # Do not compile at runtime due to circular import.
    from kkcalc.asf_database.db_models import asp_db_im, asp_db_re, asp_db_complex
    from periodictable.formulas import Formula

# Generate a list of atomic elements. Should already be sorted from the periodictable module.
ELEMENTS: list[tuple[str, int, float]]
"""A list of tuples containing the atomic symbol, atomic number and atomic mass of each element."""
if has_periodictable:
    ELEMENTS = [
        # Also contains N=0, i.e. neutral, as the first element. So ELEMENTS[1] = H.
        (element.symbol, element.number, element.mass)
        for element in pt.elements
    ]
else:
    # Use the asf database
    import os

    path_elements = __file__.replace("stoich.py", "asf_database/data/elements.dat")
    if os.path.exists(path_elements):
        data_elements = np.loadtxt(path_elements, dtype=str)
        atomic_nums = data_elements[:, 0].astype(int)
        atomic_syms = data_elements[:, 1]
        atomic_masses = data_elements[:, 3].astype(float)
        ELEMENTS = [
            (
                "n",
                0,
                1.008,
            ),  # Neutron first element, so H is ELEMENTS[1], consistent with periodictable.
            *zip(atomic_syms, atomic_nums, atomic_masses),
        ]
    else:
        raise FileNotFoundError("Element data file not found.")


def relativistic_correction_eq(composition: list[tuple[int, float]]) -> float:
    r"""
    Calculate a relativistic correction to the Kramers-Kronig transform owing to the elemental composition.

    Automatically calculable for a `stoichiometry` using `stoichiometry.relativistic_correction`. Each element contributes
    (z - (z/82.5)**2.37) * n to the correction, where z is the atomic number and n is the relative stoichiometry.

    .. math::
        \mathcal{Z}^\star = \sum_i (Z_i - (Z_i/82.5)^{2.37}) \cdot n_i

    Parameters
    ----------
    composition : list[tuple[int, float]]
        A list of tuples, where each tuple contains the atomic number and the counts of an element.
        Counts may be fractional.

    Returns
    -------
    float
        The relativistic corection to the Kramers-Kronig transform.
    """
    return sum([(z - (z / 82.5) ** 2.37) * n for z, n in composition])


CompositionAlias: TypeAlias = "Iterable[tuple[int, float]] | Formula | str | Self"


class stoichiometry:
    """
    Defines the stoichiometry of a chemical compound.

    Internally uses a periodictable.formulas.Formula object,
    or a list of tuples to represent the composition of a compound.

    Parameters
    ----------
    composition : list[tuple[int, float]] | Formula | str | Self
        The stoichiometry of the compound, i.e. the elemental composition.
        Can be a list of tuples, a Formula object, a string or another stoichiometry object.

        Examples:
        - [(6, 9), (1, 12), (8, 6), (16, 2)] for C9H12O6S2
        - "C9H12O6S2" for C9H12O6S2
        - pt.formula("C9H12O6S2") for C9H12O6S2
        - "(A)1.2(B)0.8" for a combined composition.
    """

    def __init__(self, composition: CompositionAlias) -> None:  # numpydoc ignore=GL08
        if isinstance(composition, type(self)):
            # Copy the formula / list.
            self._composition = composition._composition.copy()
        elif has_periodictable and isinstance(composition, Formula):
            self._composition = composition
        elif isinstance(composition, str):
            # Convert string to composition.
            c = stoichiometry.__parse_chemical_formula(composition)
            c = stoichiometry.__consolidate_elements(c)
            self._composition = c
        elif hasattr(composition, "__iter__"):
            # Check validity of composition, collect duplicate elements
            final_comp: list[tuple[int, int | float]] = []
            for elem, n in composition:
                if elem < 1 or elem > 92:
                    raise ValueError("Atomic number out of range.")
                if n < 0:
                    raise ValueError("Negative stoichiometry.")
                # Check if element is already accounted for
                exists: bool = False
                for i, (elem2, n2) in enumerate(final_comp):
                    if elem == elem2:
                        final_comp[i] = (elem, n + n2)
                        exists = True
                        continue
                if not exists:
                    final_comp.append((elem, n))
            self._composition = final_comp
        else:
            raise ValueError("Invalid stoichiometry.")

    def __eq__(self, other: Self | str) -> bool:
        """
        Compare the stoichiometry of two compounds.

        Comparison is made by calling the `composition` property, rather than the `_composition` attribute.

        Parameters
        ----------
        other : stoichiometry | str
            The stoichiometry to compare with the current stoichiometry.
            Can also be a string representation of a stoichiometry.

        Returns
        -------
        bool
            True if the stoichiometry of the compounds are equal, False otherwise.
        """
        if isinstance(other, self.__class__):
            return self.composition == other.composition
        elif isinstance(other, str):
            # Try to convert the string to a stoichiometry object.
            try:
                other = stoichiometry(other)
                return self.composition == other.composition
            except ValueError:
                # Try to convert
                return str(self) == other
        return False

    def __req__(self, other: Self | str) -> bool:
        """
        Compare the stoichiometry of two compounds.

        Comparison is made by calling the `composition` property, rather than the `_composition` attribute.

        Parameters
        ----------
        other : stoichiometry | str
            The stoichiometry to compare with the current stoichiometry.
            Can also be a string representation of a stoichiometry.

        Returns
        -------
        bool
            True if the stoichiometry of the compounds are equal, False otherwise.
        """
        return self.__eq__(other)

    def __str__(self) -> str:
        """
        Generate a string representation of the stoichiometry.

        Returns
        -------
        str
            A string representation of the stoichiometry.

        Examples
        --------
        >>> str(stoichiometry("C9H12") + stoichiometry("O6S2"))
        'C9H12O6S2'
        """
        return "".join(
            [
                ELEMENTS[element[0]][0]
                + (
                    str(element[1])
                    if (element[1] * 10) % 10 != 0
                    else (  # Check if the number is an integer
                        str(int(element[1])) if element[1] != 1 else ""
                    )  # Check if the number is 1
                )
                for element in self._composition
            ]
        )

    def __repr__(self) -> str:
        return f"stoichiometry({self._composition})"

    @property
    def elements(self) -> list[str]:
        """
        Return a string list of the elements present in the stoichiometry.

        Returns
        -------
        list[str]
            A list of the elemental symbols present in the stoichiometry.
        """
        return [ELEMENTS[element][0] for element, _ in self.composition]

    def __len__(self) -> float | int:
        """
        Return the summed number of each element in the stoichiometry.

        Returns
        -------
        float | int
            The summed number of each element in the stoichiometry. Can be fractional.
        """
        return sum([int(count) for _, count in self.composition])

    def __add__(self, other: Self | str) -> Self:
        """
        Combine two stoichiometries, or a stoichiometry and a string.

        Parameters
        ----------
        other : stoichiometry | str
            The stoichiometry to combine with the current stoichiometry.
            Can also be a string representation of a stoichiometry.

        Returns
        -------
        stoichiometry
            A new stoichiometry object with the composition of the two combined.
        """
        if isinstance(other, str):
            return self.__class__(self.composition + stoichiometry(other).composition)
        return self.__class__(self.composition + other.composition)

    def __radd__(self, other: Self | str) -> Self:
        """
        Combine two stoichiometries, or a stoichiometry and a string.

        Parameters
        ----------
        other : stoichiometry | str
            The stoichiometry to combine with the current stoichiometry.
            Can also be a string representation of a stoichiometry.

        Returns
        -------
        stoichiometry
            A new stoichiometry object with the composition of the two combined.
        """
        return self.__add__(other)

    def __iadd__(self, other: Self | str) -> None:
        """
        Add another stoichiometry to the current stoichiometry.

        Parameters
        ----------
        other : stoichiometry
            The stoichiometry to add to the current stoichiometry.

        Returns
        -------
        stoichiometry
            The current stoichiometry object with the composition of the two combined.
        """
        initial_comp = self.composition
        if isinstance(other, str):
            other_comp = stoichiometry(other).composition
        # Check validity of composition, collect duplicate elements
        final_comp: list[tuple[int, int | float]] = []
        for elem, n in initial_comp + other_comp:
            # Check if element is already accounted for
            exists: bool = False
            for i, (elem2, n2) in enumerate(final_comp):
                if elem == elem2:
                    final_comp[i] = (elem, n + n2)
                    exists = True
                    continue
            if not exists:
                final_comp.append((elem, n))
        self._composition = final_comp

    def __mul__(self, other: float) -> Self:
        """
        Multiply the stoichiometry by a scalar.

        Parameters
        ----------
        other : float
            The scalar to multiply the stoichiometry by.

        Returns
        -------
        Self
            A new stoichiometry object with the composition multiplied by the scalar.
        """
        return self.__class__(
            [(elem, count * other) for elem, count in self.composition]
        )

    def __rmul__(self, other: float) -> Self:
        """
        Reflection operator for multiplication.

        Parameters
        ----------
        other : float
            The scalar to multiply the stoichiometry by.

        Returns
        -------
        Self
            A new stoichiometry object with the composition multiplied by the scalar.
        """
        return self.__mul__(other)

    def __truediv__(self, other: float) -> Self:
        """
        Calculate the true division of a stoichiometry object by a scalar.

        Parameters
        ----------
        other : float
            The scalar to divide the stoichiometry by.

        Returns
        -------
        stoichiometry
            A new stoichiometry object with the composition divided by the scalar.

        Examples
        --------
        >>> stoichiometry("C9H12O6S2") / 2
        C4.5H6O3S1
        """
        return self.__mul__(1 / other)

    def __floordiv__(self, other: float) -> Self:
        """
        Calculate the floor division of a stoichiometry object by a scalar.

        Parameters
        ----------
        other : float
            The scalar by which to divide the stoichiometry.

        Returns
        -------
        stoichiometry
            A new stoichiometry object with the composition divided by the scalar.

        Examples
        --------
        >>> stoichiometry("C9H12O6S2") // 2
        C4H6O3S1
        """
        return self.__class__(
            [(elem, int(count // other)) for elem, count in self.composition]
        )

    def copy(self) -> Self:
        """
        Create a copy of stoichiometry.

        Returns
        -------
        stoichiometry
            A copy of the stoichiometry object, with a unique composition reference.
        """
        return self.__class__(self.composition.copy())

    @property
    def composition(self) -> list[tuple[int, float]]:
        """
        The stoichiometry of the compound, i.e. the elemental composition.

        Returns
        -------
        list[tuple[int, float]]
            A list of tuples, where each tuple contains the atomic number and the counts of an element.
            Counts may be fractional.
        """
        if has_periodictable and isinstance(self._composition, Formula):
            c = []
            element: Element
            count: float
            for element, count in self._composition.atoms.items():
                c.append((element.number, count))
            return c
            # return [(element.number, count) for element, count in self._composition.atoms.items()]
        elif isinstance(self._composition, list):
            return self._composition.copy()
        else:
            raise ValueError("Composition is not a valid type.")

    @property
    def relativistic_correction(self) -> float:
        r"""
        Calculate the relativistic correction to the Kramers-Kronig transform owing to the elemental composition.

        Uses `stoich.relativistic_correction_eq`. Each element contributes (z - (z/82.5)**2.37) * n to the correction,
        where z is the atomic number and n is the relative stoichiometry.

        .. math::
            \mathcal{Z}^\star = \sum_i (Z_i - (Z_i/82.5)^{2.37}) \cdot n_i

        Returns
        -------
        float
            The relativistic corection to the Kramers-Kronig transform.
        """
        return relativistic_correction_eq(composition=self.composition)

    @property
    def formula_mass(self) -> float:
        """
        The sum of atomic masses.

        Returns
        -------
        float
            The sum of atomic masses for the given stoichiometry.
        """
        if has_periodictable:
            return sum(
                [
                    number * pt.elements[element].mass
                    for element, number in self.composition
                ]
            )
        else:
            return sum(
                [number * ELEMENTS[element][2] for element, number in self.composition]
            )

    def asp_im(self) -> "asp_db_im":
        """
        Generate a piecewise polynomial of the imaginary atomic scattering factors for the given stoichiometry.

        Uses the energy-dependent atomic scattering factor data from the Henke, Briggs and Lighthill database.

        Returns
        -------
        asp_db_im
            An object representing the piecewise polynomial calculated from the summation of scattering factor data.
        """
        from kkcalc.asf_database.db_models import asp_db_im

        return asp_db_im(self)

    @doc_copy(asp_im)
    def atomic_scattering_polynomial_im(self) -> "asp_db_im":  # numpydoc ignore=RT01
        """
        An alias for `asp_im`.
        """
        return self.asp_im()

    def asp_re(self) -> "asp_db_re":
        """
        Generate a piecewise polynomial of the real atomic scattering factors for the given stoichiometry.

        Uses the energy-dependent atomic scattering factor data from the Henke, Briggs and Lighthill database.

        Returns
        -------
        asp_db_re
            An object representing the dispersive piecewise polynomial calculated from the summation of scattering factor data.
        """
        from kkcalc.asf_database.db_models import asp_db_re

        return asp_db_re(self)

    @doc_copy(asp_re)
    def atomic_scattering_polynomial_re(self) -> "asp_db_re":  # numpydoc ignore=RT01
        """
        An alias for `asp_re`.
        """
        return self.asp_re()

    def asp_complex(self) -> "asp_db_complex":
        """
        Generate a piecewise polynomial of the complex atomic scattering factors for the given stoichiometry.

        Uses the energy-dependent atomic scattering factor data from the Henke, Briggs and Lighthill database.

        Returns
        -------
        asp_db_complex
            An object representing the complex piecewise polynomial calculated from the summation of scattering factor data.
        """
        from kkcalc.asf_database.db_models import asp_db_complex

        return asp_db_complex(self)

    @doc_copy(asp_complex)
    def atomic_scattering_polynomial_complex(
        self,
    ) -> "asp_db_complex":  # numpydoc ignore=RT01
        """
        An alias for `asp_complex`.
        """
        return self.asp_complex()

    @staticmethod
    def __consolidate_elements(
        composition: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        """
        Consolidate a list of elements and quantities into a unique list of elements and quantities.

        Parameters
        ----------
        composition : list[tuple[int, float]]
            A list of tuples, where each tuple contains the atomic number and the counts of an element.

        Returns
        -------
        list[tuple[int, float]]
            A list of tuples, where each tuple contains the atomic number and the counts of an element.
        """
        # Setup a dictionary to store the composition
        consolidated = {}
        for element, count in composition:
            if element in consolidated:
                consolidated[element] += count
            else:
                consolidated[element] = count
        return [(element, count) for element, count in consolidated.items()]

    @staticmethod
    def __parse_chemical_formula(
        formula: str, recursion: bool = True
    ) -> list[tuple[int, float]]:
        """
        Convert a chemical compound string into a list of elements and quantities.

        Parameters
        ----------
        formula : str
            A string consisting of element symbols, numbers and parentheses.
        recursion : bool, optional
            Flag to enable recursion in the parsing of the formula string, by default True.

        Returns
        -------
        list[tuple[int, int]]
            A list of tuples, where each tuple contains the atomic number and the counts of an element.
        """
        # Setup a list to store the composition
        composition = []
        ## Regex explaination:
        # ?P<groupname> is a named group to capture.
        # Here we 1st capture either an element symbol or a parenthesized group.
        # Then we capture a number (if present) and the remainder of the formula.
        # <Paren> or <Remainder> groups are then also processed by a recursive call.
        # +? is a non-greedy match, to capture the smallest possible group.
        search = re.compile(
            "".join(
                [
                    r"((?P<Element>[A-Z][a-z]?)|\((?P<Paren>.+?)\))",
                    r"(?P<Number>\d*(\.\d+)?)(?P<Remainder>.*)",
                ]
            )
        )
        # Perform the search on the formula
        m = re.search(search, formula)
        if m is None:
            raise ValueError(f"No formula match: {formula}")
        # Process the search.
        if len(m.group("Number")) != 0:
            Number = float(m.group("Number"))
        else:
            Number = 1.0
        if m.group("Element") is not None:
            Z = stoichiometry._element_to_atomic_number(m.group("Element"))
            if Z != 0:
                composition.append((Z, Number))
        elif len(m.group("Paren")) > 0:
            composition += [
                (x[0], x[1] * Number)
                for x in stoichiometry.__parse_chemical_formula(
                    m.group("Paren"), recursion=recursion
                )
            ]
        if len(m.group("Remainder")) != 0:
            composition += stoichiometry.__parse_chemical_formula(
                m.group("Remainder"), recursion=recursion
            )
        return composition

    @staticmethod
    def from_chemical_formula(
        formula: str, recursion: bool = True, use_peroidictable: bool = True
    ) -> "stoichiometry":
        """
        Parse a chemical formula string to obtain a stoichiometry.

        Parameters
        ----------
        formula : str
            A string consisting of element symbols, numbers and parentheses.
        recursion : bool, optional
            Whether to use recursion to parse the formula, by default True.
        use_peroidictable : bool, optional
            Whether to use the periodictable module to parse the formula, by default True.

        Returns
        -------
        stoichiometry
            A stoichiometry object representing the composition of the formula.
        """
        if use_peroidictable:
            return stoichiometry(pt.formula(formula))
        else:
            # Parse the formula string
            composition = stoichiometry.__parse_chemical_formula(
                formula=formula, recursion=recursion
            )
            # Consolidate the elements
            composition = stoichiometry.__consolidate_elements(composition)
            # Create the stoichiometry object
            return stoichiometry(composition)

    @staticmethod
    def _element_to_atomic_number(SymbolString: str) -> int:
        """
        Replace list of elemental symbols with the corresponding atomic numbers.

        Parameters
        ----------
        SymbolString : str
            An elemental symbol (i.e. "H", "C", "O", etc.).

        Returns
        -------
        int
            The function returns an integer atomic number corresponding to the input symbol.
            Zero is returned when the string is not recognised.
        """
        for i in range(len(ELEMENTS)):
            if ELEMENTS[i][0] == SymbolString:
                return ELEMENTS[i][1]
        raise ValueError(f"`{SymbolString}` is not a known element!")

    @staticmethod
    def _atomic_number_to_element(Z: int) -> str:
        """
        Replace list of atomic numbers with the corresponding elemental symbols.

        Parameters
        ----------
        Z : int
            Integer representing an atomic number.

        Returns
        -------
        str
            The function returns a string elemental symbol corresponding to the input atomic number.
        """
        # Z'th list item should match the element.
        if ELEMENTS[Z][1] == Z:
            return ELEMENTS[Z][0]
        # If not, search for the element index. This should not be necessary.
        for i in range(len(ELEMENTS)):
            if ELEMENTS[i][1] == Z:
                return ELEMENTS[i][0]
        raise ValueError(f"Element #{Z} is not a known atomic number to kkcalc!")


if __name__ == "__main__":
    # Test the stoichiometry class
    P3MEET1 = "C9H12O6S2"  # C9H11O3S
    P3MEET2 = "(C9H12O6S2)0.1(C9H11O3S)0.9"
    if has_periodictable:
        P3MEET3 = pt.formula("C9H12O6S2")

    compounds = [P3MEET1, P3MEET2] + ([P3MEET3] if has_periodictable else [])
    data_titles = [
        "Stoichiometry",
        "Composition",
        "Relativistic Correction",
        "Formula Mass",
    ]
    data = []

    for c, compound in enumerate(compounds):
        stoich = stoichiometry(compound)
        comp = stoich.composition
        for i, (atom, count) in enumerate(comp):
            if type(count) is float and int(count) != count:
                comp[i] = (atom, float(f"{count:.2f}"))  # Round to 3 decimal places
        data.append(
            [compound, comp, stoich.relativistic_correction, stoich.formula_mass]
        )
        if c == 1:
            print(
                f"Testing bracketed formula: {stoich.composition[0]} == {9*0.1 + 9*0.9}? {stoich.composition[0][1] == 9*0.1 + 9*0.9}"
            )

    import pandas as pd

    df = pd.DataFrame(data, columns=data_titles)
    print(df)
