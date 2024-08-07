"""
'Atomic scattering factor' data models.

Defines the types of data that can be used, and conversion between.
"""

from kkcalc.stoich import stoichiometry as kk_stoichiometry
from kkcalc.util import doc_copy
from kkcalc.models.conversions import conversions
from kkcalc.models.polynomials import asp as asp_type, asp_abstract, asp_im, asp_re, asp_complex

import numpy as np
import numpy.typing as npt
import abc
from enum import Enum

class KK_Datatype(Enum):
    """
    Enum for the type of data to be used in the Kramers-Kronig calculation
    """
    UNDEFINED = 0
    NEXAFS = 1 # AKA Photoabsorption, XANES.
    XANES = 1 # AKA Photoabsorption, NEXAFS.
    PHOTOABSORPTION = 1 # AKA NEXAFS, XANES.
    BETA = 2 # Index of refraction
    ASF = 3 # Atomic scattering factors
    
class asf_abstract(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def energies(self) -> np.ndarray:
        """
        Returns / Sets the energies of the atomic scattering factors.
        
        Parameters
        ----------
        energies : np.ndarray
            Energies in eV.
        
        Returns
        -------
        np.ndarray
            Energies in eV.
        """
        pass
    
    @property
    @abc.abstractmethod
    def factors(self) -> np.ndarray:
        """
        Returns / sets the atomic scattering factors.
        
        Parameters
        ----------
        factors : array_like
            Atomic scattering factors.
        
        Returns
        -------
        np.ndarray
            Atomic scattering factors.
        """
        pass
    
    @property 
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Gets/sets atomic scattering factor data (energies and amplitudes).
        
        Parameters
        ----------
        data : tuple[np.ndarray, np.ndarray]
            Tuple of energies (eV) and atomic scattering factors.
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of energies (eV) and atomic scattering factors.
        """
        return self.energies, self.factors

    @data.setter
    def data(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        if not isinstance(data, tuple) or len(data) != 2 or len(data[0]) != len(data[1]):
            raise ValueError("Data must be a tuple of two equal length arrays.")
        self.energies, self.factors = np.asarray(data[0]), np.asarray(data[1])
    
    
    @property
    def betas(self,
                number_density:float=None,
                density:float=None, 
                formula_mass:float=None, 
                stoichiometry:kk_stoichiometry|str=None,) -> np.ndarray:
        """
        Converts object atomic scattering factors and energies to Beta values (index of refraction).
        
        Parameters
        ----------
        number_density : float
            Number density of the material in atoms per millilitre (cm^3).
        density : float
            Material density in grams per millilitre (cm^3).
        formula_mass : float
            Atomic mass sum of the materials chemical formula (molecular mass).
            Equivalent to providing a stoichiometry.
        stoichiometry : stoichiometry | str
            Description of the combination of elements composing the material.

        Returns
        -------
        np.ndarray
            Beta values.
        """
        return conversions.ASF_to_betas(
            energies=self.energies, 
            factors=self.factors,
            number_density=number_density,
            density=density,
            formula_mass=formula_mass,
            stoichiometry=stoichiometry            
        )
        
    def to_betas(self,
                number_density:float=None,
                density:float=None, 
                formula_mass:float=None, 
                stoichiometry:kk_stoichiometry|str=None,
                ) -> tuple[np.ndarray, np.ndarray]:
        """
        A tuple of energies and Beta (index of refraction) values.
        
        Beta values are converted from atomic scattering factors and energies.

        Parameters
        ----------
        number_density : float
            Number density of the material in atoms per millilitre (cm^3).
        density : float
            Material density in grams per millilitre (cm^3).
        formula_mass : float
            Atomic mass sum of the materials chemical formula (molecular mass).
            Equivalent to providing a stoichiometry.
        stoichiometry : stoichiometry | str
            Description of the combination of elements composing the material.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of energies (eV) and Beta values.
        """
        return self.energies, self.betas(number_density, density, formula_mass, stoichiometry)
        
    @staticmethod
    def from_betas(energies: npt.NDArray,
                  beta: npt.NDArray,
                  number_density:float=None,
                  density:float=None, 
                  formula_mass:float=None, 
                  stoichiometry:kk_stoichiometry|str=None,
                ) -> "asf":
        """
        Converts Beta values (index of refraction) to atomic scattering factors (ASF).
        
        The Beta value is the imaginary part of the index of refraction, representing absorption.
        Requires some form of material density information to convert to ASF.
        As per positional argument order, the function will use the first available density information.
        This can either be:
        - `number_density` in atoms per millilitre (cm^3),
        - `density` in grams per millilitre (cm^3), and `formula_mass` (molecular mass),
        - `stoichiometry` as a list of elemental symbol, number pairs or string of a formula.
        
        Parameters
        ----------
        energies : array_like
            Photon energies in eV.
        beta : array_like
            Imaginary part of the index of refraction.
        number_density : float, optional
            Material density in atoms per millilitre (cm^3).
        density : float
            Material density in grams per millilitre (cm^3).
        formula_mass : float
            Atomic mass sum of the materials chemical formula (molecular mass).
            Equivalent to providing a `stoichiometry`.
        stoichiometry : stoichiometry | str
            Description of the combination of elements composing the material.
        """
        # Convert energy and beta data to numpy arrays.
        energies = np.asarray(energies)
        beta = np.asarray(beta)
        # Perform conversion
        factors = conversions.betas_to_ASF(energies, beta, number_density, density, formula_mass, stoichiometry)
        # Return asf instance
        return asf(energies, factors, KK_Datatype.BETA, beta)

    @property
    def NEXAFS(self) -> np.ndarray:
        """
        Converts atomic scattering factors to NEXAFS/XANES/Photoabsorption data.

        Returns
        -------
        np.ndarray
            NEXAFS photoabsorption values corresponding to the `energies` property.
        """
        return conversions.ASF_to_NEXAFS(self.energies, self.factors)
    
    def to_NEXAFS(self) -> tuple[np.ndarray, np.ndarray]:
        """
        A tuple of energies and NEXAFS photoabsorption values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of energies (eV) and NEXAFS photoabsorption values.
        """
        return self.energies, self.NEXAFS
    
    @staticmethod
    def from_NEXAFS(energies: npt.NDArray, 
                    NEXAFS: npt.NDArray) -> "asf":
        """
        Converts NEXAFS photoabsorption data to atomic scattering factors (ASF).
        
        Parameters
        ----------
        energies : array_like
            Photon energies in eV.
        NEXAFS : array_like
            NEXAFS/XANES/photoabsorption data.
        """
        return asf(energies, conversions.NEXAFS_to_ASF(energies, NEXAFS), KK_Datatype.NEXAFS, np.c_[energies, NEXAFS])

    @staticmethod
    @doc_copy(conversions.ASF_to_ASP)
    def atomic_scattering_factors_to_coefs(energies: npt.NDArray, 
                                            factors: npt.NDArray) -> npt.NDArray:
        """
        Alias for `conversions.ASF_to_ASP` to calculate the atomic scattering polynomial coefficients from 
        atomic scattering `factors` defined at `energies`.
        """
        return conversions.ASF_to_ASP(energies, factors)

    @property
    def atomic_scattering_polynomial(self) -> npt.NDArray:
        """
        Converts atomic scattering factors to atomic scattering polynomial coefficients.
        
        Uses `energies` and `factors` with length `N` to calculate the atomic scattering polynomial coefficients.
        To convert to an `asp` object, use the `to_atomic_scattering_polynomial` method.

        Returns
        -------
        npt.NDArray
            An array with dimension (`N-1`, 5) of atomic scattering polynomial coefficients.
            
        """
        return self.atomic_scattering_factors_to_coefs(self.energies, self.factors)
    
    @property
    @doc_copy(atomic_scattering_polynomial)
    def asp(self) -> npt.NDArray:
        """
        Alias for `atomic_scattering_polynomial`.
        """
        return self.atomic_scattering_polynomial
    
    @abc.abstractmethod
    def to_atomic_scattering_polynomial(self) -> type[asp_abstract]:
        """
        Converts asf object to asp object.
        
        Uses the `energies` and `factors` attributes (with length `N`) of the object
        to create an atomic scattering polynomial object of coefficients with length (N-1).
        
        For an array of polynomial coefficients, use the `atomic_scattering_polynomial` property.

        Returns
        -------
        asp
            Atomic scattering polynomial object.
        """
        pass

    @abc.abstractmethod
    @doc_copy(to_atomic_scattering_polynomial)
    def to_ASP(self) -> type[asp_abstract]:
        """
        Alias for `to_atomic_scattering_polynomial`.
        """
        return self.to_atomic_scattering_polynomial()
    
class asf(asf_abstract):
    """
    Generic class for handling atomic scattering factors.
    
    Provides static methods to convert from `KK_Datatype` to scattering factors.
    
    Parameters
    ----------
    energies : np.ndarray
        Energies in eV.
    factors : np.ndarray
        Atomic scattering factors. 
        If data y-data is instead betas or NEXAFS, use the respective `from_<name>` method.
    origin_dtype : KK_Datatype, optional
        Original data type of the atomic scattering factors.
        If not provided, the original data is assumed to be the same as the input data.
    origin_data : np.ndarray, optional
        Original data of the atomic scattering factors.
        If not provided, the original data is assumed to be the same as the input data.
    """
    def __init__(self,
                 energies: npt.NDArray,
                 factors: npt.NDArray,
                 origin_dtype: KK_Datatype | None = None,
                 origin_data: npt.NDArray | None = None
                 ) -> None:
        self._energies = energies = np.asarray(energies)
        self._factors = factors = np.asarray(factors)
        if origin_dtype is None:
            # If no original data type is provided, assume the input data is the original data.
            self._origin_dtype = KK_Datatype.ASF
        else:
            self._origin_dtype = origin_dtype
        if origin_data is not None:
            # Store a copy of original data.
            origin_data = np.asarray(origin_data)
            origin_data = origin_data.copy()
            if len(origin_data.shape) != 2:
                raise ValueError("Original data must contain two columns.")
            self._origin_data = origin_data
        elif origin_dtype is None:
            # If no original data is provided, assume the input data is the original data.
            self._origin_data = np.c_[energies, factors].copy()
            
    @property
    def energies(self) -> np.ndarray:
        """
        Returns / Sets the energies of the atomic scattering factors.
        
        Parameters
        ----------
        energies : np.ndarray
            Energies in eV.
        
        Returns
        -------
        np.ndarray
            Energies in eV.
        """
        return self._energies
    
    @energies.setter
    def energies(self, energies: np.ndarray) -> None:
        self._energies = np.asarray(energies)

    @property 
    def factors(self) -> np.ndarray:
        """
        Returns / sets the atomic scattering factors.
        
        Parameters
        ----------
        factors : array_like
            Atomic scattering factors.
        
        Returns
        -------
        np.ndarray
            Atomic scattering factors.
        """
        return self._factors
    
    @factors.setter
    def factors(self, factors: np.ndarray) -> None:
        self._factors = np.asarray(factors)
    
    @property
    def origin_dtype(self) -> KK_Datatype:
        """
        Returns the original data type of the atomic scattering factors.

        Returns
        -------
        KK_Datatype
            Enumerate of the original data type.
        """
        return self._origin_dtype
    
    @property
    def origin_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.origin_dtype == KK_Datatype.UNDEFINED:
            return None
        if self.origin_dtype == KK_Datatype.ASF:
            return self.data
        return self.energies, self._origin_data.copy()
    
    def to_atomic_scattering_polynomial(self) -> asp_type:
        """
        Converts asf object to asp object.
        
        Uses the `energies` and `factors` attributes (with length `N`) of the object
        to create an atomic scattering polynomial object of coefficients with length (N-1).
        
        For an array of polynomial coefficients, use the `atomic_scattering_polynomial` property.

        Returns
        -------
        asp
            Atomic scattering polynomial object.
        """
        return asp_type(
            energies=self.energies[:-1],
            coefs=self.atomic_scattering_polynomial
        )

    @doc_copy(to_atomic_scattering_polynomial)
    def to_ASP(self) -> asp_type:
        """
        Alias for `to_atomic_scattering_polynomial`.
        """
        return self.to_atomic_scattering_polynomial()
    
    
class asf_re(asf):
    """
    Identical to `asf`, but reserved for real component factors.
    """
    
    @staticmethod
    def from_asf(asf: asf) -> "asf_re":
        """
        Converts an `asf` object to an `asf_re` object.
        
        Parameters
        ----------
        asf : asf
            Atomic scattering factors object.
        """
        return asf_re(asf.energies, 
                      asf.factors, 
                      asf.origin_dtype, 
                      asf.origin_data)
        
    def to_atomic_scattering_polynomial(self) -> asp_re:
        """
        Converts `asf_re` object to `asp_re` object.
        
        Uses the `energies` and `factors` attributes (with length `N`) of the object
        to create an atomic scattering polynomial object of coefficients with length (N-1).
        
        For an array of polynomial coefficients, use the `atomic_scattering_polynomial` property.

        Returns
        -------
        asp
            Atomic scattering polynomial object.
        """
        return asp_re(
            energies=self.energies[:-1],
            coefs=self.atomic_scattering_polynomial
        )

    @doc_copy(to_atomic_scattering_polynomial)
    def to_ASP(self) -> asp_re:
        """
        Alias for `to_atomic_scattering_polynomial`.
        """
        return self.to_atomic_scattering_polynomial()
        
class asf_im(asf):
    """
    Identical to `asf`, but reserved for imaginary component factors.
    """
    
    @staticmethod
    def from_asf(asf: asf) -> "asf_im":
        """
        Converts an `asf` object to an `asf_im` object.
        
        Parameters
        ----------
        asf : asf
            Atomic scattering factors object.
        """
        return asf_im(asf.energies, 
                      asf.factors, 
                      asf.origin_dtype, 
                      asf.origin_data)
        
    def to_atomic_scattering_polynomial(self) -> asp_im:
        """
        Converts `asf_im` object to `asp_im` object.
        
        Uses the `energies` and `factors` attributes (with length `N`) of the object
        to create an atomic scattering polynomial object of coefficients with length (N-1).
        
        For an array of polynomial coefficients, use the `atomic_scattering_polynomial` property.

        Returns
        -------
        asp
            Atomic scattering polynomial object.
        """
        return asp_im(
            energies=self.energies[:-1],
            coefs=self.atomic_scattering_polynomial
        )

    @doc_copy(to_atomic_scattering_polynomial)
    def to_ASP(self) -> asp_im:
        """
        Alias for `to_atomic_scattering_polynomial`.
        """
        return self.to_atomic_scattering_polynomial()
        
class asf_complex(asf_abstract):
    """
    Container for a pair of atomic scattering factors, reflecting
    the real and imaginary parts.

    Parameters
    ----------
    re : asf_re | asf
        Real part of the atomic scattering factors.
    im : asf_im | asf
        Imaginary part of the atomic scattering factors.
    """
    def __init__(self,
                 re: asf_re | asf,
                 im: asf_im | asf,
                 ):
        if not np.all(re.energies == im.energies):
            raise ValueError("Real and imaginary parts must have the same energy intervals.")        
        
        # Convert to appropriate instance objects
        if isinstance(re, asf):
            re = asf_re.from_asf(re)
        if isinstance(im, asf):
            im = asf_im.from_asf(im)
            
        # Store attributes
        self._re : asf_re = re
        self._im : asf_im = im
    
    @property
    def energies(self) -> npt.NDArray:
        return self._re.energies
    
    @energies.setter
    def energies(self, energies: npt.NDArray) -> None:
        self._re.energies = energies
        self._im.energies = energies
    
    @property
    def factors(self) -> tuple[npt.NDArray, npt.NDArray]:
        return self._re.factors, self._im.factors
    
    @factors.setter
    def factors(self, 
                factors: tuple[npt.NDArray, npt.NDArray] 
                         | npt.NDArray[np.complex_]) -> None:
        if isinstance(factors, tuple):
            self._re.factors = factors[0]   
            self._im.factors = factors[1]
        else:
            self._re.factors = factors.real
            self._im.factors = factors.imag
    
    @property
    def re(self) -> "asf_re":
        return self._re

    @property
    def im(self) -> "asf_im":
        return self._im
    
    def to_atomic_scattering_polynomial(self) -> "asp_complex":
        """
        Converts `asf_complex` object to `asp_complex` object.
        
        Uses the `energies` and `factors` attributes (with length `N`) of the object
        to create an atomic scattering polynomial object of coefficients with length (N-1).
        
        For an array of polynomial coefficients, use the `atomic_scattering_polynomial` property.

        Returns
        -------
        asp
            Atomic scattering polynomial object.
        """
        return asp_complex(
            re=self._re.to_atomic_scattering_polynomial(),
            im=self._im.to_atomic_scattering_polynomial()
        )

    @doc_copy(to_atomic_scattering_polynomial)
    def to_ASP(self) -> asp_complex:
        """
        Alias for `to_atomic_scattering_polynomial`.
        """
        return self.to_atomic_scattering_polynomial()