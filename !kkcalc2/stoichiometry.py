import re
import periodictable as pt
from periodictable.formulas import Formula
import numpy as np
import numpy.typing as npt

# Generate a list of atomic elements. Should already be sorted from the periodictable module.
ELEMENTS = [
    # Also contains N=0, i.e. neutral, as the first element. So ELEMENTS[1] = H.
    (element.symbol, element.number)
    for element in pt.elements
]
# Load the real/imag scattering factors as they vary with energy
from .asf import ASF_DATABASE

class stoichiometry:
    """
    Defines the stoichiometry of a chemical compound.
    
    Internally uses a periodictable.formulas.Formula object,
    or a list of tuples to represent the composition of a compound.
    """
    def __init__(self, 
                 composition: list[tuple[int, int]] | Formula = None) -> None:
        if isinstance(Formula):
            self._composition = composition
        else:
            # Check validity of composition.
            for element in composition:
                if element[0] < 1 or element[0] > 92: 
                    raise ValueError("Atomic number out of range.")
                if element[1] < 0:
                    raise ValueError("Negative stoichiometry.")
        self._composition = composition.copy()
    
    def __str__(self) -> str:
        return "".join([ELEMENTS[element[0]]+(str(element[1]) if element[1] != 1 else "") for element in self._composition])
        
    def __repr__(self) -> str:
        return f"stoichiometry({self._composition})"
    
    @property
    def composition(self) -> list[tuple[int, int]]:
        """
        The stoichiometry of the compound, i.e. the elemental composition.
        
        Returns
        -------
        list[tuple[int, int]]
            A list of tuples, where each tuple contains the atomic number and the counts of an element.
        """
        if isinstance(self._composition, Formula):
            return [(element, count) for element, count in self._composition.atoms]
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
    def calculate_asf(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Summation of scattering factor data given the chemical stoichiometry.
        
        Returns
        -------
        numpy.ndarray
            An array listing the starting photon energies of the segments that the spectrum is broken up into.
        total_Im_coeffs: nx5 numpy array in which each row lists the polynomial coefficients describing the shape of the spectrum in that segment.
        """
        numpy = np
        stoich = self.composition
        
        # Get unique energy points for all elements
        energies = np.unique(numpy.array(
            [ASF_DATABASE[z]['E'] for z,_ in stoich]
        ))
        
        # Add weighted asf data sets for KK calculation
        im_coefs = numpy.zeros((len(energies)-1, 5)) # Stores summations of imaginary coefficients at each energy
        # Create an array to keep track of the current elemental energy as we iterate over the unique energies.
        counters = numpy.zeros((len(stoich)), dtype=int)
        # Iterate over the unique energies
        for i, energy in enumerate(energies[1:]): # iterate over the energies of the ASF_DATABASE
            sum_im = 0
            # Sum the imaginary coefficients at each energy
            for j, (z, n) in enumerate(stoich):
                # Imaginary coefs at current energy
                im_coefs = ASF_DATABASE[z]['Im'][counters[j],:] # the imaginary piecewise polynomial coefficients
                sum_im += n * im_coefs  # Multiply by stoichiometry n
                
                # Check if the next energy matches the currently used elemental energy, i.e. end of the valid interval.
                if ASF_DATABASE[z]['E'][counters[j]+1] == energy:
                    counters[j] += 1 # Increment counter[j] by 1 if the energy matches
            # Store the sum of the imaginary coefficients at the current energy
            im_coefs[i,:] = sum_im
        return energies, im_coefs
    
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
                    composition.append([Z,Number])
            elif len(m.group('Paren')) > 0:
                composition +=[[x[0],x[1]*Number] for x in stoichiometry.from_chemical_formula(m.group('Paren'), recursion=recursion)]
            if len(m.group('Remainder')) != 0:
                composition += stoichiometry.from_chemical_formula(m.group('Remainder'), recursion=recursion)
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
        raise ValueError("`"+SymbolString+"` is not a known element!")

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
        raise ValueError(str(Z)+" is not a known atomic number!")
    
