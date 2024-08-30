"""
'Atomic scattering factor' data models.

Defines the types of data that can be used, and conversion between.
"""

# In polynomials.py, the equivalent import is only done via type checking or in functions, to prevent recursion.
from kkcalc.models.polynomials import asp as asp_type, asp_abstract, asp_im, asp_re, asp_complex
## ..
from kkcalc.models.common import atomic_scattering_abstract, atomic_scattering
from kkcalc.models.conversions import conversions
from kkcalc.stoich import stoichiometry as kk_stoichiometry # To prevent overlap use with the `stoichiometry` argument.
from kkcalc.util import doc_copy
from kkcalc.kk_transforms import DEF_ITER, DEF_TOL

import numpy as np
import numpy.typing as npt
import abc
import warnings
import pandas as pd
from enum import Enum
from typing import Self, Union, Iterator, override

KK_DATATYPE_DOCS: dict[str, str] = {
    "UNDEFINED":"""For undefined data types.""",
    "NEXAFS":"""Near edge X-ray absorption fine structure.""",
    "XANES":"""X-ray absorption near edge structure.""",
    "PHOTOABSORPTION":"""Photoabsorption.""",
    "BETA":r"""
    Index of refraction, with dispersive :math:`\delta` and absorptive :math:`\beta` components.
    
    .. math::
    
        n(E) = 1 - \delta(E) - i\beta(E)   
    """,
    "ASF":r"""
    Atomic scattering factors, real :math:`f_1` & imaginary :math:`f_2` components.
    Both scattering strengths are relative to the Thompson scattering of a free electron.
    Calculated for a set of :math:`m` elements, with number density :math:`N_m`, 
    wavelength :math:`\lambda`, photon energy :math:`E` and classical electron radius :math:`r_0`.
    
    .. math::
        n(E) = 1 - \frac{r_0}{2\pi}\lambda^2\sum_m \left(f_{1m}(E) + i f_{2m}(E)\right)
    
    See Also: 
    kkcalc.stoich.relativistic_correction_eq : The sum of relativistic corrections for a elemental composition.    
    """,
    
    "ASF_DASH":r"""
    Atomic scattering factors, real :math:`f^{0}`, :math:`f^{'}` and imaginary :math:`f^{''}` components.
    Here the relativistic correction is :math:`f^{0}`, and :math:`f^{'}` is the energy dependent real part.
    Both scattering strengths are relative to the Thompson scattering of a free electron.
    
    See Also
    --------
    kkcalc.stoich.relativistic_correction_eq : The relativistic correction for a elemental composition.
    """,
}

class KK_Datatype(Enum):
    """
    Enum for the type of data to be used in the Kramers-Kronig calculation
    """
    UNDEFINED = 0
    """For undefined data types.""",
    NEXAFS = 1 # AKA Photoabsorption, XANES.
    """Near edge X-ray absorption fine structure.""",
    XANES = 1 # AKA Photoabsorption, NEXAFS.
    """X-ray absorption near edge structure.""",
    PHOTOABSORPTION = 1 # AKA NEXAFS, XANES.
    """Photoabsorption."""
    BETA = 2 # Index of refraction
    r"""
    Index of refraction, with dispersive :math:`\delta` and absorptive :math:`\beta` components.
    
    .. math::
    
        n(E) = 1 - \delta(E) - i\beta(E)   
    """
    ASF = 3 # Atomic scattering factors as per the original KK Calc; f1 & f2, 
    r"""
    Atomic scattering factors, real :math:`f_1` & imaginary :math:`f_2` components.
    Both scattering strengths are relative to the Thompson scattering of a free electron.
    Calculated for a set of :math:`m` elements, with number density :math:`N_m`, 
    wavelength :math:`\lambda`, photon energy :math:`E` and classical electron radius :math:`r_0`.
    
    .. math::
        n(E) = 1 - \frac{r_0}{2\pi}\lambda^2\sum_m \left(f_{1m}(E) + i f_{2m}(E)\right)
    
    See Also: 
    kkcalc.stoich.relativistic_correction_eq : The sum of relativistic corrections for a elemental composition.
    """
    ASF_DASH = 4 # Atomic scattering factors f0, f' & f'',
    r"""
    Atomic scattering factors, real :math:`f^{0}`, :math:`f^{'}` and imaginary :math:`f^{''}` components.
    Here the relativistic correction is :math:`f^{0}`, and :math:`f^{'}` is the energy dependent real part.
    Both scattering strengths are relative to the Thompson scattering of a free electron.
    
    See Also
    --------
    kkcalc.stoich.relativistic_correction_eq : The relativistic correction for a elemental composition.
    """

for i, dtype in enumerate(KK_Datatype):
    name = dtype.name.upper()
    if name in KK_DATATYPE_DOCS:
        dtype.__doc__ = KK_DATATYPE_DOCS[name]
    
class asf_abstract(atomic_scattering_abstract, metaclass=abc.ABCMeta):
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
    def can_calc_beta(self) -> bool:
        """
        Returns whether the object can calculate Beta values.
        
        Returns
        -------
        bool
            Whether the object can calculate Beta values.
        """
        return (self.number_density is not None
                #Formula mass property uses stoichiometry if not provided.
                or (self.density is not None and self.formula_mass is not None)) 
                                                   
    @property
    def betas(self) -> np.ndarray:
        """
        Converts object atomic scattering factors and energies to Beta values (index of refraction).
        
        The Beta value is the imaginary part of the index of refraction, representing absorption.
        Requires some form of material density information to convert to ASF.
        This can either be:
        - `number_density` in atoms per millilitre (cm^3),
        - `density` in grams per millilitre (cm^3), and
            - `formula_mass` (molecular mass), or
            - `stoichiometry` as a list of elemental symbol, number pairs or string of a formula.
        
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
        if self.number_density is not None:
            return conversions.ASF_to_betas(
            energies=self.energies, 
            factors=self.factors,
            number_density=self.number_density,
        )
        elif self.density is not None:
            if self.formula_mass is not None:
                return conversions.ASF_to_betas(
                    energies=self.energies, 
                    factors=self.factors,
                    density=self.density,
                    formula_mass=self.formula_mass,
                )
            elif self.stoichiometry is not None:
                return conversions.ASF_to_betas(
                    energies=self.energies, 
                    factors=self.factors,
                    density=self.density,
                    stoichiometry=self.stoichiometry,
                )
        raise ValueError("Material density information is required to convert to Beta values.")
 
    def to_betas(self,
                number_density: float | None = None,
                density: float | None = None, 
                formula_mass: float | None = None, 
                stoichiometry: kk_stoichiometry | str | None = None,
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
        if number_density is None and density is None and formula_mass is None and stoichiometry is None:
            # Attempt to use the object's density information to convert to Beta values.
            number_density = self.number_density
            density = self.density
            formula_mass = self.formula_mass
            stoichiometry = self.stoichiometry
        else:
            if stoichiometry is not None and isinstance(stoichiometry, str):
                stoichiometry = kk_stoichiometry(stoichiometry)
            
        # Attempt to use available density information to convert to Beta values.
        if number_density is not None:
            return self.energies, conversions.ASF_to_betas(energies=self.energies,
                                                            factors=self.factors,
                                                            number_density=number_density)
        elif density is not None:
            if formula_mass is not None:
                return self.energies, conversions.ASF_to_betas(energies=self.energies,
                                                                factors=self.factors,
                                                                density=density,
                                                                formula_mass=formula_mass)
            elif stoichiometry is not None:
                return self.energies, conversions.ASF_to_betas(energies=self.energies,
                                                                factors=self.factors,
                                                                density=density,
                                                                stoichiometry=stoichiometry)
        raise ValueError("Material density information is required to convert to Beta values.")

    
    @classmethod
    @abc.abstractmethod
    def from_betas(cls: Self,
                   energies: npt.NDArray,
                   beta: npt.NDArray,
                   number_density:float=None,
                   density:float=None, 
                   formula_mass:float=None, 
                   stoichiometry:kk_stoichiometry|str=None,
                   scale_to_database: bool = False,):
        """Converts Beta values (index of refraction) to atomic scattering factors (ASF)."""
        pass

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
    
    @classmethod
    @abc.abstractmethod
    def from_NEXAFS(cls: Self,
                    energies: npt.NDArray, 
                    NEXAFS: npt.NDArray,
                    scale_to_database: bool = False,
                    **kwargs) -> type[Self]:
        """
        Converts NEXAFS photoabsorption data to atomic scattering factors (ASF).
        """
        pass

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
    
    def dataframe(self) -> pd.DataFrame:
        """
        Generates a Pandas representation of the factors list, useful for display.
        """
        return pd.DataFrame(
            np.c_[self.energies, self.factors],
            columns=["Energy (eV)", "ASF"]
        )
        
    def __str__(self, **kwargs) -> str:
        """
        Creates a Pandas string representation of the factors list.
        
        Rows displayed are the first and last 5 if more than 10 rows.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the `pd.dataFrame.to_string` method.

        Returns
        -------
        str
            A string representation of the factors.
        """
        # Create a default max_rows if not provided.
        if "max_rows" not in kwargs:
            kwargs["max_rows"] = 10
        return self.dataframe().to_string(**kwargs)
    
    def __getitem__(self, key: int | slice) -> type[Self]:
        """
        Gets the atomic scattering factors at the specified index/slice.
        
        Parameters
        ----------
        key : int | slice
            Index or slice of the atomic scattering factors.
        
        Returns
        -------
        type[asf_abstract]
            Atomic scattering factors object at the specified index.
        """
        if not isinstance(key, (int, slice)):
            raise TypeError("Index must be an integer or slice.")
        common_kwargs = self._properties_dict
        return self.__class__(energies=self.energies[key], factors=self.factors[key], **common_kwargs)
    
    def __iter__(self) -> Iterator[tuple[float, float | complex]]:
        """
        Provides each energy and factor of the energy-dependent scattering amplitude.

        Yields
        ------
        energy : float
            The energy at which the scattering factor is defined.
        factor : float
            The atomic scattering factor.
        """
        for i in range(len(self.energies)):
            yield (self.energies[i], self.factors[i])
    
    @abc.abstractmethod
    def copy(self, **kwargs) -> type[Self]:
        """
        Generates a copy of the `asf` object.
        
        Parameters
        ----------
        **kwargs
            Any keyword arguments for the constructors to update the copy properties.

        Returns
        -------
        type[asf_abstract]
            A new `asf` object with the same atomic scattering factors,
            and properties, but unique memory allocation.
        """
        pass
    
class asf(asf_abstract, atomic_scattering):
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
    **kwargs
        Additional keyword arguments for the `kkcalc.models.common.atomic_scattering` such as:
        - `number_density` : float
        - `density` : float
        - `stoich` : stoichiometry
        - `formula_mass` : float
        - `name` : str
    
    See Also
    --------
    kkcalc.models.common.atomic_scattering : Common attributes between atomic scattering factor and polynomial models.
    """
    def __init__(self,
                 energies: npt.NDArray,
                 factors: npt.NDArray,
                 origin_dtype: KK_Datatype | None = None,
                 origin_data: npt.NDArray | None = None,
                 scale_to_database: bool = False,
                 **kwargs
                 ) -> None:
        atomic_scattering.__init__(self, **kwargs)
        self._energies = None
        self._factors = None
        
        self.energies = energies = np.asarray(energies)
        self.factors = factors = np.asarray(factors)
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
            self._origin_data = np.c_[energies, factors] # already creates copies
            
        if scale_to_database and self.__class__ == asf:
            # Do not allow the base class to scale, as no complexity designation.
            warnings.warn(f"Scaling to database is only available for real and imaginary components, not for {self}. Turning off scaling.")
            scale_to_database = False
            
        if scale_to_database:
            self.scale_to_database()
    
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
        if self.factors is not None and len(self._energies) != len(self.factors):
            warnings.warn("Length of energies does not match the length of factors. Factors have been discarded.")
            self._factors = None

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
        factors = np.asarray(factors)
        if len(factors) != len(self.energies):
            raise ValueError("Length of factors does not match the length of energies.")
        self._factors = factors
    
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
    def origin_data(self) -> np.ndarray | None:
        """
        Returns the original data provided for the atomic scattering factors.

        Returns
        -------
        np.ndarray | None
            Original data of the atomic scattering factors, matching
            the format described by the `origin_dtype` attribute.
            Returns a copy.
        """
        if self.origin_dtype == KK_Datatype.UNDEFINED:
            return None
        return self._origin_data.copy()
    
    def scale_to_database(self, **kwargs) -> None:
        """
        If the object contains a stoichiometry and has an identifiable complexity (imag, real),
        this method scales the atomic scattering factors to the database scale.
        
        Origin data is preserved.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the `atomic_scattering` constructor.
        
        Raises
        ------
        ValueError
            If the object does not contain a stoichiometry.
            If the object is not a real or imaginary designated complexity.
        """
        if self.stoichiometry is not None:
            if isinstance(self, asf_re):
                from kkcalc.asf_database.db_models import asp_db_re
                self.factors = asp_db_re.scale_data(self.energies, self.factors, self.stoichiometry)
                return
            elif isinstance(self, asf_im):
                from kkcalc.asf_database.db_models import asp_db_im
                self.factors = asp_db_im.scale_data(self.energies, self.factors, self.stoichiometry)
                return
            raise ValueError(f"Scaling to database is only available for real and imaginary components. {self} is {self.__class__}.")
        raise ValueError("Scaling to database requires a stoichiometry.")
    
    def to_atomic_scattering_polynomial(self, **kwargs) -> asp_type:
        """
        Converts asf object to asp object.
        
        Uses the `energies` and `factors` attributes (with length `N`) of the object
        to create an atomic scattering polynomial object of coefficients with length (N-1).
        
        For an array of polynomial coefficients, use the `atomic_scattering_polynomial` property.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the `asp` and `atomic_scattering` object.

        Returns
        -------
        asp
            Atomic scattering polynomial object.
            
        See Also
        --------
        kkcalc.models.polynomials.asp : Atomic scattering polynomial object.
        kkcalc.models.common.atomic_scattering : Common attributes between atomic scattering factor and polynomial models.
        """
        # Get the existing properties, update with kwargs
        common_kwargs = self._properties_dict
        common_kwargs.update(kwargs)
        # Return asp object
        return asp_type(
            energies=self.energies[:-1],
            coefs=self.atomic_scattering_polynomial,
            **common_kwargs
        )

    @doc_copy(to_atomic_scattering_polynomial)
    def to_ASP(self, **kwargs) -> asp_type:
        """
        Alias for `to_atomic_scattering_polynomial`.
        """
        return self.to_atomic_scattering_polynomial(**kwargs)
    
    @classmethod
    def from_NEXAFS(cls: Self,
                    energies: npt.NDArray, 
                    NEXAFS: npt.NDArray,
                    scale_to_database: bool = False,
                    **kwargs) -> type[Self]:
        """
        Converts NEXAFS photoabsorption data to atomic scattering factors (ASF).
        
        Parameters
        ----------
        energies : array_like
            Photon energies in eV.
        NEXAFS : array_like
            NEXAFS/XANES/photoabsorption data.
        scale_to_database : bool, optional
            Whether to scale the atomic scattering factors to the database scale.
            Requires a stoichiometry and a designated complexity (i.e. asf_im or asf_re).
        **kwargs
            Additional keyword arguments for the `atomic_scattering` object.
            
        Returns
        -------
        type[asf]
            Atomic scattering factors object.
        
        See Also
        --------
        kkcalc.models.common.atomic_scattering : Common attributes between atomic scattering factor and polynomial models.
        """
        return cls(energies=energies,
                   factors=conversions.NEXAFS_to_ASF(energies, NEXAFS),
                   origin_dtype=KK_Datatype.NEXAFS, 
                   origin_data=np.c_[energies, NEXAFS],
                   scale_to_database=scale_to_database,
                   **kwargs)
    
    @classmethod
    def from_betas(cls: Self,
                   energies: npt.NDArray,
                   beta: npt.NDArray,
                   number_density:float=None,
                   density:float=None, 
                   formula_mass:float=None, 
                   stoichiometry:kk_stoichiometry|str=None,
                   scale_to_database: bool = False,
                   **kwargs
                   ) -> Union[Self, "asf_abstract"]:
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
            Real/imaginary index of refraction values.
        number_density : float, optional
            Material density in atoms per millilitre (cm^3).
        density : float
            Material density in grams per millilitre (cm^3).
        formula_mass : float
            Atomic mass sum of the materials chemical formula (molecular mass).
            Equivalent to providing a `stoichiometry`.
        stoichiometry : stoichiometry | str
            Description of the combination of elements composing the material.
        scale_to_database : bool, optional
            Whether to scale the atomic scattering factors to the database scale.
            Requires a stoichiometry and a designated complexity (i.e. asf_im or asf_re).
        **kwargs
            Additional keyword arguments for the `atomic_scattering` object.
            
        See Also
        --------
        kkcalc.models.common.atomic_scattering : Common attributes between atomic scattering factor and polynomial models.
        """
        # Convert energy and beta data to numpy arrays.
        energies = np.asarray(energies)
        beta = np.asarray(beta)
        # Perform conversion
        factors = conversions.betas_to_ASF(energies, beta, number_density, density, formula_mass, stoichiometry)
        # Accumulate keyword arguments
        kwargs.update({
            "number_density": number_density,
            "density": density,
            "formula_mass": formula_mass,
            "stoichiometry": stoichiometry
        })
        # Return asf instance
        return cls(energies=energies, 
                   factors=factors, 
                   origin_dtype=KK_Datatype.BETA,
                   origin_data=np.c_[energies, beta],
                   scale_to_database=scale_to_database,
                   **kwargs)
    
    def copy(self, **kwargs) -> type["asf"]:
        """
        Generates a copy of the `asp` object.
        
        Parameters
        ----------
        **kwargs
            Any keyword arguments for the constructors to update the copy properties.

        Returns
        -------
        type[asp]
            A new `asp` object with the same polynomial coefficients, 
            and properties, but unique memory allocation.
        """
        # Copy the object properties
        common_kwargs = self._properties_dict
        for key in common_kwargs:
            if hasattr(common_kwargs[key], "copy"):
                common_kwargs[key] = common_kwargs[key].copy()
        # Update with kwargs
        common_kwargs.update(kwargs)
        # Create a new object
        return self.__class__(energies=self.energies.copy(),
                              factors=self.factors.copy(),
                              origin_dtype=self.origin_dtype,
                              origin_data=self.origin_data.copy(),
                              **common_kwargs)
    
class asf_re(asf):
    """
    Identical to `asf`, but reserved for real component factors.
    """
    
    @staticmethod
    def from_asf(asf: asf, **kwargs) -> Self:
        """
        Converts an `asf` object to an `asf_re` object.
        
        Parameters
        ----------
        asf : asf
            Atomic scattering factors object.
        **kwargs
            Additional keyword arguments for the `asf_re` and `atomic_scattering` object.

        Returns
        -------
        asf_re
            Atomic scattering factors object designated as a real component.
            
        See Also
        --------
        kkcalc.models.common.atomic_scattering : Common attributes between atomic scattering factor and polynomial models.
        """
        # Get the existing properties, update with kwargs
        common_kwargs = asf._properties_dict
        common_kwargs.update(kwargs)
        # Return asf object
        return asf_re(asf.energies, 
                      asf.factors, 
                      asf.origin_dtype, 
                      asf.origin_data,
                      **common_kwargs)
        
    def to_atomic_scattering_polynomial(self, **kwargs) -> asp_re:
        """
        Converts `asf_re` object to `asp_re` object.
        
        Uses the `energies` and `factors` attributes (with length `N`) of the object
        to create an atomic scattering polynomial object of coefficients with length (N-1).
        
        For an array of polynomial coefficients, use the `atomic_scattering_polynomial` property.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the `asp_re` and `atomic_scattering` object.

        Returns
        -------
        asp
            Atomic scattering polynomial object.
            
        See Also
        --------
        kkcalc.models.polynomials.asp : Atomic scattering polynomial object.
        kkcalc.models.common.atomic_scattering : Common attributes between atomic scattering factor and polynomial models.
        """
        # Get the existing properties, update with kwargs
        common_kwargs = self._properties_dict
        common_kwargs.update(kwargs)
        # Return asp object
        return asp_re(
            energies=self.energies,
            coefs=self.atomic_scattering_polynomial,
            **common_kwargs
        )

    @doc_copy(to_atomic_scattering_polynomial)
    def to_ASP(self, **kwargs) -> asp_re:
        """
        Alias for `to_atomic_scattering_polynomial`.
        """
        return self.to_atomic_scattering_polynomial(**kwargs)
    
    def kk_transform_inv(self,
                         target_energies: npt.NDArray | None = None,
                         improve_accuracy: bool = True,
                         stoichiometry: kk_stoichiometry | None = None,
                         relativistic_correction: float | None = None,
                         tolerance: float | None = DEF_TOL,
                         max_iter: int | None = DEF_ITER,
                         ) -> "asf_im":
        """
        Inverse Kramers-Kronig transform for the real part of the atomic scattering factors.
        
        Converts `asf_re` to `asp_re` and uses the `KK_PP` method to calculate the inverse transform.
       
        Parameters
        ----------
        target_energies : npt.NDArray
            The energies at which to calculate the inverse transform.
        improve_accuracy : bool, optional
            Whether to add extra data points to increase resolution.
        stoichiometry : stoichiometry, optional
            The stoichiometry of the material.
        relativistic_correction : float, optional
            The relativistic correction to apply.
            
        Returns
        -------
        asf_im
            Atomic scattering factors object with imaginary components.
        """
        # Convert
        asp_re = self.to_atomic_scattering_polynomial()
        return asp_re.kk_transform_inv(target_energies=target_energies,
                                       improve_accuracy=improve_accuracy,
                                       stoichiometry=stoichiometry,
                                       relativistic_correction=relativistic_correction,
                                       max_iter=max_iter,
                                       tolerance=tolerance)
    
    def calculate_complex_polynomial(self, 
                                     target_energies: npt.NDArray | None = None,
                                     improve_accuracy: bool = True,
                                     stoichiometry: kk_stoichiometry | None = None,
                                     relativistic_correction: float | None = None,
                                     tolerance: float | None = DEF_TOL,
                                     max_iter: int | None = DEF_ITER,
                                     **kwargs) -> "asp_complex":
        """
        Converts the imaginary part of the atomic scattering factors to real factors, and then uses both 
        to form a complex polynomial representation.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the `asp_complex` and `atomic_scattering` classes.
        
        Returns
        -------
        asp_complex
            An atomic scattering polynomial object.
            
        See Also
        --------
        kkcalc.models.polynomials.asp_complex : Complex atomic scattering polynomial class.
        kkcalc.models.common.atomic_scattering : Common attributes between atomic scattering factor and polynomial models.
        """
        # Return asp object
        re_asp = self.to_atomic_scattering_polynomial()
        return re_asp.calculate_complex_polynomial(target_energies=target_energies,
                                                   improve_accuracy=improve_accuracy,
                                                   stoichiometry=stoichiometry,
                                                   relativistic_correction=relativistic_correction,
                                                   tolerance=tolerance,
                                                   max_iter=max_iter,
                                                   **kwargs)
        
    def calculate_complex_factors(self, 
                                  target_energies: npt.NDArray | None = None,
                                  improve_accuracy: bool = True,
                                  stoichiometry: kk_stoichiometry | None = None,
                                  relativistic_correction: float | None = None,
                                  tolerance: float | None = DEF_TOL,
                                  max_iter: int | None = DEF_ITER,
                                  **kwargs) -> "asf_complex":
        """
        Converts the real part of the atomic scattering factors to imaginary factors, and then uses both 
        to form a complex representation.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the `asf_complex` and `atomic_scattering` classes.
        
        Returns
        -------
        asf_complex
            A complex atomic scattering factor object.
            
        See Also
        --------
        kkcalc.models.common.atomic_scattering : Common attributes between atomic scattering factor and polynomial models.
        """
        re: asp_re = self.to_atomic_scattering_polynomial()
        im: asf_im = re.kk_transform_inv(target_energies=target_energies,
                                         improve_accuracy=improve_accuracy,
                                         stoichiometry=stoichiometry,
                                         relativistic_correction=relativistic_correction,
                                         tolerance=tolerance,
                                         max_iter=max_iter)
        re_extended: asf_re = re.extend_energies(im.energies)
        # Return asf object
        common_kwargs = self._properties_dict
        common_kwargs.update(kwargs)
        return asf_complex(re=re_extended.to_atomic_scattering_polynomial(),
                           im=im,
                           **common_kwargs) # asf_complex already pulls properties from components.
        
class asf_im(asf):
    """
    Identical to `asf`, but reserved for imaginary component factors.
    """
    
    @staticmethod
    def from_asf(asf: asf, **kwargs) -> Self:
        """
        Converts an `asf` object to an `asf_im` object.
        
        Parameters
        ----------
        asf : asf
            Atomic scattering factors object.
        """
        common_kwargs = asf._properties_dict
        common_kwargs.update(kwargs)
        return asf_im(asf.energies, 
                      asf.factors, 
                      asf.origin_dtype, 
                      asf.origin_data,
                      **common_kwargs)
        
    def to_atomic_scattering_polynomial(self, **kwargs) -> asp_im:
        """
        Converts `asf_im` object to `asp_im` object.
        
        Uses the `energies` and `factors` attributes (with length `N`) of the object
        to create an atomic scattering polynomial object of coefficients with length (N-1).
        
        For an array of polynomial coefficients, use the `atomic_scattering_polynomial` property.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the `asp_im` and `atomic_scattering` object.

        Returns
        -------
        asp
            Atomic scattering polynomial object.
        """
        common_kwargs = self._properties_dict
        common_kwargs.update(kwargs)
        return asp_im(
            energies=self.energies,
            coefs=self.atomic_scattering_polynomial,
            **common_kwargs
        )

    @doc_copy(to_atomic_scattering_polynomial)
    def to_ASP(self) -> asp_im:
        """
        Alias for `to_atomic_scattering_polynomial`.
        """
        return self.to_atomic_scattering_polynomial()
    
    def kk_transform(self,
                     target_energies: npt.NDArray | None = None,
                     improve_accuracy: bool = True,
                     stoichiometry: kk_stoichiometry | None = None,
                     relativistic_correction: float | None = None,
                     tolerance: float | None = DEF_TOL,
                     max_iter: int | None = DEF_ITER,
                     ) -> "asf_re":
        """
        Kramers-Kronig transform for the imaginary part of the atomic scattering factors.

        Converts `asf_im` to `asp_im` and uses the `KK_PP` method to calculate the transform.
        
        Parameters
        ----------
        target_energies : npt.NDArray, optional
            The energies at which to calculate the transform.
        improve_accuracy : bool, optional
            Whether to add extra data points to increase resolution.
        stoichiometry : stoichiometry, optional
            The stoichiometry of the material.
        relativistic_correction : float, optional
            The relativistic correction to apply.
        tolerance : float, optional
            Improvements below this magnitude are considered negligible.
        max_iter : int, optional
            The maximum number of iterations to perform.

        Returns
        -------
        asf_re
            Atomic scattering factors object with real components.
        """ 
        asp_im = self.to_atomic_scattering_polynomial()
        return asp_im.kk_transform(target_energies=target_energies,
                                   improve_accuracy=improve_accuracy,
                                   stoichiometry=stoichiometry,
                                   relativistic_correction=relativistic_correction,
                                   tolerance=tolerance,
                                   max_iter=max_iter)
        
    def calculate_complex_polynomial(self, 
                                     target_energies: npt.NDArray | None = None,
                                     improve_accuracy: bool = True,
                                     stoichiometry: kk_stoichiometry | None = None,
                                     relativistic_correction: float | None = None,
                                     tolerance: float | None = DEF_TOL,
                                     max_iter: int | None = DEF_ITER,
                                     **kwargs) -> "asp_complex":
        """
        Converts the imaginary part of the atomic scattering factors to real factors, and then uses both 
        to form a complex polynomial representation.
        
        Parameters
        ----------
        target_energies : npt.NDArray, optional
            The energies at which to calculate the transform.
        improve_accuracy : bool, optional
            Whether to add extra data points to increase resolution.
        stoichiometry : stoichiometry, optional
            The stoichiometry of the material.
        relativistic_correction : float, optional
            The relativistic correction to apply.
        tolerance : float, optional
            Improvements below this magnitude are considered negligible.
        max_iter : int, optional
            The maximum number of iterations to perform.
        **kwargs
            Additional keyword arguments for the `asp_complex` and `atomic_scattering` classes.
        
        Returns
        -------
        asp_complex
            An atomic scattering polynomial object.
        """
        im_asp = self.to_atomic_scattering_polynomial()
        return im_asp.calculate_complex_polynomial(target_energies=target_energies,
                                                   improve_accuracy=improve_accuracy,
                                                   stoichiometry=stoichiometry,
                                                   relativistic_correction=relativistic_correction,
                                                   tolerance=tolerance,
                                                   max_iter=max_iter
                                                   **kwargs)
        
    def calculate_complex_factors(self,
                                  target_energies: npt.NDArray | None = None,
                                  improve_accuracy: bool = True,
                                  stoichiometry: kk_stoichiometry | None = None,
                                  relativistic_correction: float | None = None,
                                  tolerance: float | None = DEF_TOL,
                                  max_iter: int | None = DEF_ITER,
                                  **kwargs) -> "asf_complex":
        """
        Converts the real part of the atomic scattering factors to imaginary factors, and then uses both 
        to form a complex representation.
        
        Parameters
        ----------
        target_energies : npt.NDArray, optional
            The energies at which to calculate the transform.
        improve_accuracy : bool, optional
            Whether to add extra data points to increase resolution.
        stoichiometry : stoichiometry, optional
            The stoichiometry of the material.
        relativistic_correction : float, optional
            The relativistic correction to apply.
        tolerance : float, optional
            Improvements below this magnitude are considered negligible.
        max_iter : int, optional
            The maximum number of iterations to perform.
        **kwargs
            Additional keyword arguments for the `asf_complex` and `atomic_scattering` classes.
        
        Returns
        -------
        asf_complex
            A complex atomic scattering factor object.
        """
        im: asp_im = self.to_atomic_scattering_polynomial()
        re: asf_re = im.kk_transform(target_energies=target_energies,
                                     improve_accuracy=improve_accuracy,
                                     stoichiometry=stoichiometry,
                                     relativistic_correction=relativistic_correction,
                                     tolerance=tolerance,
                                     max_iter=max_iter)
        if im.energies.shape != re.energies.shape or np.any(im.energies != re.energies):
            # Extend the energies of the imaginary part to match the real part
            im_extended = im.extend_energies(re.energies)
        else:
            # No extension required
            im_extended = im
        # Convert to complex object
        common_kwargs = self._properties_dict
        common_kwargs.update(kwargs)
        return asf_complex(re=re,
                           im=im_extended.to_atomic_scattering_factors(),
                           **common_kwargs)
    
class asf_complex(asf_abstract, atomic_scattering):
    """
    Container for a pair of atomic scattering factors, reflecting
    the real and imaginary parts.

    Parameters
    ----------
    re : asf_re | asf
        Real part of the atomic scattering factors.
    im : asf_im | asf
        Imaginary part of the atomic scattering factors.
    **kwargs
        Additional keyword arguments for the `atomic_scattering` subclass.
        Does not effect the real or imaginary parts.
    
    See Also
    --------
    kkcalc.models.common.atomic_scattering : Common attributes between atomic scattering factor and polynomial
    """
    def __init__(self,
                 re: asf_re | asf,
                 im: asf_im | asf,
                 **kwargs
                 ):
        if not np.all(re.energies == im.energies):
            raise ValueError("Real and imaginary parts must have the same energy intervals.")
        if not isinstance(re, asf) or not isinstance(im, asf):
            raise TypeError(f"Real and imaginary parts must be subclasses of {asf}.")
        
        # Use the real then imaginary part properties to update None values
        common_kwargs = re._properties_dict
                
        # Check properties are the same
        for key in im._properties_dict:
            if key not in common_kwargs or common_kwargs[key] is None:
                common_kwargs[key] = im._properties_dict[key]
            elif common_kwargs[key] != im._properties_dict[key]:
                warnings.warn(f"Property {key} is different between real {re._properties_dict[key]}" 
                              + f" and imaginary parts {im._properties_dict[key]} for {self}.")
            else:
                # Ignore if the properties are the same
                pass
        
        # Update properties with kwargs
        common_kwargs.update(kwargs)
        
        # Convert to appropriate instance objects
        if isinstance(re, asf):
            re = asf_re.from_asf(re)
        if isinstance(im, asf):
            im = asf_im.from_asf(im)
            
        # Store attributes
        self._re : asf_re = re
        self._im : asf_im = im
        
        # Initialise atomic scattering object
        atomic_scattering.__init__(self, **common_kwargs)
    
    @property
    def energies(self) -> npt.NDArray:
        return self._re.energies
    
    @energies.setter
    def energies(self, energies: npt.NDArray) -> None:
        self._re.energies = energies
        self._im.energies = energies
    
    @property
    def factors(self) -> npt.NDArray[np.complex_]:
        return self._re.factors +  1j*self._im.factors
    
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
    def abs(self) -> npt.NDArray:
        """
        Absolute values of the atomic scattering factors.
        
        Returns
        -------
        np.ndarray
            Absolute values of the atomic scattering factors.
        """
        return np.abs(self.factors)
    
    @property
    def phase(self) -> npt.NDArray:
        """
        Phase of the atomic scattering factors.
        
        Returns
        -------
        np.ndarray
            Phase of the atomic scattering factors.
        """
        return np.angle(self.factors)
    
    @property
    def re(self) -> "asf_re":
        """
        Real part object of the atomic scattering factors.
        
        Returns
        -------
        asf_re
            Real part of the atomic scattering factors.
        """
        return self._re

    @property
    def im(self) -> "asf_im":
        """
        Imaginary part object of the atomic scattering factors.
        
        Returns
        -------
        asf_im
            Imaginary part of the atomic scattering factors.
        """
        return self._im
    
    def to_atomic_scattering_polynomial(self, **kwargs) -> "asp_complex":
        """
        Converts `asf_complex` object to `asp_complex` object.
        
        Uses the `energies` and `factors` attributes (with length `N`) of the object
        to create an atomic scattering polynomial object of coefficients with length (N-1).
        
        For an array of polynomial coefficients, use the `atomic_scattering_polynomial` property.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the `asp_complex` and `atomic_scattering` object.

        Returns
        -------
        asp
            Atomic scattering polynomial object.
            
        See Also
        --------
        kkcalc.models.polynomials.asp_complex : Complex atomic scattering polynomial object.
        """
        # Get the existing properties, update with kwargs
        common_kwargs = self._properties_dict
        common_kwargs.update(kwargs)
        # Return asp object
        return asp_complex(
            re=self._re.to_atomic_scattering_polynomial(),
            im=self._im.to_atomic_scattering_polynomial(),
            **common_kwargs
        )

    @doc_copy(to_atomic_scattering_polynomial)
    def to_ASP(self, **kwargs) -> asp_complex:
        """
        Alias for `to_atomic_scattering_polynomial`.
        """
        return self.to_atomic_scattering_polynomial(**kwargs)

    def contrast(self, other: "asf_complex") -> tuple[npt.NDArray, npt.NDArray]:
        r"""
        The energy-dependent contrast magnitude of the atomic scattering factors,
        which is the difference in imaginary and real parts squared.
        
        If the objects have different energy domains, only the common energies are considered,
        and a warning is issued. 
        
        .. math::
            contrast ~ \Delta(\delta)^2 + \Delta\beta^2
        
        Returns
        -------
        energies : np.ndarray
            Array of energy values defined for each contrast value.
        contrast : np.ndarray
            The contrast between two atomic scattering polynomials.
        """
        if self.can_calc_beta and other.can_calc_beta:
            if (self.energies.shape == other.energies.shape) and np.all(self.energies == other.energies):
                re_diff = self.re.betas - other.re.betas
                im_diff = self.im.betas - other.im.betas
                return self.energies, np.abs(re_diff)**2 + np.abs(im_diff)**2
            else:
                warnings.warn("Energy domains do not match. Only common energies are considered.")
                energy_subset = np.intersect1d(self.energies, other.energies)
                self_ind = np.searchsorted(self.energies, energy_subset)
                other_ind = np.searchsorted(other.energies, energy_subset)
                re_diff = self.re.betas[self_ind] - other.re.betas[other_ind]
                im_diff = self.im.betas[self_ind] - other.im.betas[other_ind]
                return energy_subset, np.abs(re_diff)**2 + np.abs(im_diff)**2
        else:
            raise ValueError("Both objects must have beta values to calculate contrast.")
        

    @classmethod
    def from_NEXAFS(cls: Self,
                    energies: npt.NDArray, 
                    NEXAFS: npt.NDArray[np.complex_],
                    scale_to_database: bool = False,
                    **kwargs) -> type[Self]:
        """
        Converts NEXAFS photoabsorption data to atomic scattering factors (ASF).
        
        Parameters
        ----------
        energies : array_like
            Photon energies in eV.
        NEXAFS : array_like
            NEXAFS/XANES/photoabsorption data. Infers real and imaginary parts.
        scale_to_database : bool, optional
            Whether to scale the atomic scattering factors to the database scale.
            Requires a stoichiometry and a designated complexity (i.e. asf_im or asf_re).
        **kwargs
            Additional keyword arguments for the `atomic_scattering` object.
            
        Returns
        -------
        type[asf]
            Atomic scattering factors object.
        
        See Also
        --------
        kkcalc.models.common.atomic_scattering : Common attributes between atomic scattering factor and polynomial models.
        """
        energies = np.asarray(energies)
        if NEXAFS.dtype != np.complex_:
            NEXAFS = NEXAFS.astype(np.complex_)
            warnings.warn("NEXAFS data was not complex. Assuming data is real-component only.")
        
        re = asf_re.from_NEXAFS(energies=energies, 
                                NEXAFS=NEXAFS.real,
                                scale_to_database=scale_to_database)
        im = asf_im.from_NEXAFS(energies=energies,
                                NEXAFS=NEXAFS.imag,
                                scale_to_database=scale_to_database)
        return cls(re=re, im=im, **kwargs)
    
    @classmethod
    def from_betas(cls: Self,
                   energies: npt.NDArray,
                   beta: npt.NDArray[np.complex_],
                   number_density:float=None,
                   density:float=None, 
                   formula_mass:float=None, 
                   stoichiometry:kk_stoichiometry|str=None,
                   scale_to_database: bool = False,
                   **kwargs
                   ) -> Union[Self, "asf_abstract"]:
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
            Infers real and imaginary parts of the index of refraction.
        number_density : float, optional
            Material density in atoms per millilitre (cm^3).
        density : float
            Material density in grams per millilitre (cm^3).
        formula_mass : float
            Atomic mass sum of the materials chemical formula (molecular mass).
            Equivalent to providing a `stoichiometry`.
        stoichiometry : stoichiometry | str
            Description of the combination of elements composing the material.
        scale_to_database : bool, optional
            Whether to scale the atomic scattering factors to the database scale.
            Requires a stoichiometry and a designated complexity (i.e. asf_im or asf_re).
        **kwargs
            Additional keyword arguments for the `atomic_scattering` object.
            
        See Also
        --------
        kkcalc.models.common.atomic_scattering : Common attributes between atomic scattering factor and polynomial models.
        """
        # Convert energy and beta data to numpy arrays.
        energies = np.asarray(energies)
        
        if beta.dtype != np.complex_:
            beta = beta.astype(np.complex_)
            warnings.warn("Beta data was not complex. Assuming data is real-component only.")
        
        # Accumulate keyword arguments
        kwargs.update(dict(
            number_density=number_density,
            density=density,
            formula_mass=formula_mass,
            stoichiometry=stoichiometry,
            origin_dtype=KK_Datatype.BETA,
            origin_data=np.c_[energies, beta]
        ))
        # Return asf instances
        re = asf_re.from_betas(energies=energies,
                               beta=beta.real,
                               scale_to_database=scale_to_database,
                               **kwargs)
        im = asf_im.from_betas(energies=energies,
                               beta=beta.imag,
                               scale_to_database=scale_to_database,
                               **kwargs)
        # Create complex class.
        return cls(re=re, im=im, **kwargs)

    def copy(self, **kwargs) -> type["asf_complex"]:
        """
        Generates a copy of the `asf` object.

        Parameters
        ----------
        **kwargs
            Any keyword arguments for the constructors to update the copy properties

        Returns
        -------
        type[asf_complex]
            A new `asf_complex` object with the same polynomial coefficients, 
            and properties, but unique memory allocation.
        """
        # Copy the object properties
        common_kwargs = self._properties_dict
        for key in common_kwargs:
            if hasattr(common_kwargs[key], "copy"):
                common_kwargs[key] = common_kwargs[key].copy()
        # Update with kwargs
        common_kwargs.update(kwargs)
        # Create a new object
        return self.__class__(re=self.re.copy(),
                              im=self.im.copy(),
                              **common_kwargs)