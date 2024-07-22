import re
import periodictable as pt
from periodictable.formulas import Formula
from periodictable.core import Element
import numpy as np
import numpy.typing as npt
from typing import Self, TYPE_CHECKING
from kkcalc.util import doc_copy


if TYPE_CHECKING:
    # Do not compile at runtime due to circular import.
    from kkcalc.asf_db.asf_spectra import asp_db



# Generate a list of atomic elements. Should already be sorted from the periodictable module.
ELEMENTS: list[tuple[str, int]] = [
    # Also contains N=0, i.e. neutral, as the first element. So ELEMENTS[1] = H.
    (element.symbol, element.number)
    for element in pt.elements
]

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
        
    """
    def __init__(self, 
                 composition: list[tuple[int, float]] | Formula | str | Self = None) -> None:
        if isinstance(composition, type(self)):
            # Copy the formula / list.
            self._composition = composition._composition.copy()
        elif isinstance(composition, Formula):
            self._composition = composition
        elif isinstance(composition, str):
            # Convert string to composition.
            c = stoichiometry.__parse_chemical_formula(composition)
            c = stoichiometry.__consolidate_elements(c)
            self._composition = c
        elif hasattr(composition, "__iter__"):
            # Check validity of composition.
            for (elem, n) in composition:
                if elem < 1 or elem > 92: 
                    raise ValueError("Atomic number out of range.")
                if n < 0:
                    raise ValueError("Negative stoichiometry.")    
            self._composition = composition.copy()
        else:
            raise ValueError("Invalid stoichiometry.")
    
    def __str__(self) -> str:
        return "".join([
            ELEMENTS[element[0]][0]+(str(element[1]) if element[1] != 1 else "")
            for element in self._composition
        ])
        
    def __repr__(self) -> str:
        return f"stoichiometry({self._composition})"
    
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
        if isinstance(self._composition, Formula):
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
        """
        Calculates the relativistic correction to the Kramers-Kronig transform owing to the elemental composition.
        
        Each element contributes (z - (z/82.5)**2.37) * n to the correction, where z is the atomic number and 
        n is the relative stoichiometry.

        Returns
        -------
        float
            The relativistic corection to the Kramers-Kronig transform.
        """
        return sum([(z - (z/82.5)**2.37) * n for z, n in self.composition])
    
    @property
    def formula_mass(self) -> float:
        """
        The sum of atomic masses.

        Returns
        -------
        float
            The sum of atomic masses for the given stoichiometry.
        """
        return sum([
            number*pt.elements[element].mass
            for element, number in self.composition
        ])
    
    def atomic_scattering_polynomial_im(self) -> "asp_db":
        """
        Generates a piecewise polynomial of the imaginary atomic scattering factors for the given stoichiometry.
        
        Uses the energy-dependent atomic scattering factor data from the Henke, Briggs and Lighthill database.
        
        Returns
        -------
        asf_imag_pp
            An object representing the piecewise polynomial calculated from the summation of scattering factor data.
        """
        if not "asp_db" in locals():
            from kkcalc.asf_db.asf_spectra import asp_db
        return asp_db(self)
    
    @doc_copy(atomic_scattering_polynomial_im)
    def asp_im(self) -> "asp_db":
        """
        Alias for `atomic_scattering_polynomial_im`.
        """
        return self.atomic_scattering_polynomial_im()
    
    @staticmethod
    def __consolidate_elements(composition: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """
        Consolidates a list of elements and quantities into a unique list of elements and quantities.

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
    def __parse_chemical_formula(formula: str, 
                                 recursion: bool = True
                                 ) -> list[tuple[int, float]]:
        """
        Converts a chemical compound string into a list of elements and quantities.

        Parameters
        ----------
        formula : str
            A string consisting of element symbols, numbers and parentheses.
        recursion : bool, optional
            Flag to enable recursion in the parsing of the formula string, by default True

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
        search = re.compile(r'((?P<Element>[A-Z][a-z]?)|\((?P<Paren>.*)\))(?P<Number>\d*(\.\d+)?)(?P<Remainder>.*)')
        # Perform the search on the formula
        m=re.search(search,formula)
        # Process the search.
        if len(m.group('Number')) != 0:
            Number = float(m.group('Number'))
        else:
            Number = 1.0
        if m.group('Element') is not None:
            Z = stoichiometry._element_to_atomic_number(m.group('Element'))
            if Z != 0:
                composition.append((Z,Number))
        elif len(m.group('Paren')) > 0:
            composition += [(x[0],x[1]*Number) for x in stoichiometry.__parse_chemical_formula(m.group('Paren'), recursion=recursion)]
        if len(m.group('Remainder')) != 0:
            composition += stoichiometry.__parse_chemical_formula(m.group('Remainder'), recursion=recursion)
        return composition
    
    @staticmethod
    def from_chemical_formula(formula: str, recursion: bool = True, use_peroidictable: bool = True) -> "stoichiometry":
        """Parse a chemical formula string to obtain a stoichiometry.

        Parameters
        ----------
        formula : str
            A string consisting of element symbols, numbers and parentheses
        recursion : bool, optional
            Whether to use recursion to parse the formula, by default True
        use_peroidictable : bool, optional
            Whether to use the periodictable module to parse the formula, by default True

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
                formula=formula,
                recursion=recursion
            )
            # Consolidate the elements
            composition = stoichiometry.__consolidate_elements(composition)
            # Create the stoichiometry object
            return stoichiometry(composition)
    
    @staticmethod
    def _element_to_atomic_number(SymbolString: str) -> int:
        """Replace list of elemental symbols with the corresponding atomic numbers.

        Parameters
        ----------
        SymbolString : String representing an elemental symbol

        Returns
        -------
        The function returns an integer atomic number corresponding to the input symbol.
        Zero is returned when the string is not recognised.
        """
        for i in range(len(ELEMENTS)):
            if ELEMENTS[i][0] == SymbolString:
                return ELEMENTS[i][1]
        raise ValueError(f"`{SymbolString}` is not a known element!")

    @staticmethod
    def _atomic_number_to_element(Z: int) -> str:
        """Replace list of atomic numbers with the corresponding elemental symbols.

        Parameters
        ----------
        Z : Integer representing an atomic number

        Returns
        -------
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
    P3MEET1 = "C9H12O6S2" #C9H11O3S
    P3MEET2 = "(C9H12O6S2)0.1(C9H11O3S)0.9"
    P3MEET3 = pt.formula("C9H12O6S2")
    
    compounds = [P3MEET1, P3MEET2, P3MEET3]
    data_titles = ["Stoichiometry", "Composition", "Relativistic Correction", "Formula Mass"]
    data = []
    
    for compound in compounds:
        stoich = stoichiometry(compound)
        comp = stoich.composition
        for i, (atom, count) in enumerate(comp):
            if type(count) is float and int(count) != count:
                comp[i] = (atom, f"{count:.2f}") # Round to 3 decimal places
        data.append([compound, comp, stoich.relativistic_correction, stoich.formula_mass])

    import pandas as pd
    df = pd.DataFrame(data, columns=data_titles)
    print(df)