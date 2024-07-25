from kkcalc.stoich import stoichiometry as kk_stoichiometry

import numpy as np
import numpy.typing as npt
from scipy.constants import (
    Avogadro as N_A, 
    speed_of_light as c, 
    Planck as h,
    elementary_charge as e,
    pi,
    epsilon_0,
    electron_mass as m_e
)
E_RADIUS: float = 1/(4 * pi * epsilon_0) * e**2 / (m_e * c**2) #classical electron radius

class conversions:
    """
    Container for conversion methods between atomic scattering factors and various other formats.
    
    Namely, between:
    - Photoabsorption data,
    - NEXAFS data,
    - XANES data,
    - Beta data (index of refraction)
    - Atomic scattering polynomial (ASP) coefficients
    """
    
    @staticmethod
    def betas_to_ASF(
        energies: npt.NDArray,
        betas: npt.NDArray,
        number_density: float,
        density: float = None,
        formula_mass: float = None,
        stoichiometry: kk_stoichiometry | str = None,
        reverse : bool = False
    ) -> npt.NDArray:
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
        reverse : bool
            Flag to indicate the reverse conversion.
        """
        
        # Get number density from material density information.
        if number_density:
            pass
        elif density:
            fm = None
            if formula_mass:
                fm = formula_mass
            elif stoichiometry:
                stoichiometry = kk_stoichiometry(stoichiometry) if isinstance(stoichiometry, str) else stoichiometry
                if isinstance(stoichiometry, stoichiometry):
                    fm = stoichiometry.formula_mass
                else:
                    raise ValueError("Invalid stoichiometry provided.")
            else:
                raise ValueError("Material `formula_mass` or `stoichiometry` required with `density` required to convert Beta to ASF.")
            number_density = density * N_A / fm
        else:
            raise ValueError("No material density information provided to convert Beta to ASF.")
        
        prefactor = (
            1e6 # Convert from m^3 to cm^3
            *2*pi*energies**2
            /(number_density*E_RADIUS*(h*c)**2)
        )
        if not reverse:
            # Generate factors from Beta data.
            factors = prefactor * betas
            return factors
        else:
            factors = betas #relabel betas as factors.
            # Generate betas from factor data.
            betas_reverse = factors / prefactor
            return betas_reverse
    
    @staticmethod
    def ASF_to_betas(
        energies: npt.NDArray,
        factors: npt.NDArray,
        number_density: float,
        density: float = None,
        formula_mass: float = None,
        stoichiometry: kk_stoichiometry | str = None
    ) -> npt.NDArray:
        """
        Converts atomic scattering factors (ASF) to Beta values (index of refraction).
        
        Uses `betas_to_asf` with the `reverse` flag set to `True`.
        
        Parameters
        ----------
        energies : array_like
            Photon energies in eV.
        factors : array_like
            Atomic scattering factors.
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
        return conversions.betas_to_ASF(
            energies=energies,
            factors=factors,
            number_density=number_density,
            density=density,
            formula_mass=formula_mass,
            stoichiometry=stoichiometry,
            reverse=True
        )
    
    @staticmethod 
    def NEXAFS_to_ASF(
        energies: npt.NDArray,
        NEXAFS: npt.NDArray,
        reverse : bool = False
    ) -> npt.NDArray:
        """Convert NEXAFS photoabsorption data to atomic scattering factors (ASF).

        Parameters
        ----------
        raw_data : two-dimensional `numpy.array` of `float`
            The array consists of two columns: Energy and magnitude.
        reverse : boolean
            flag to indicate the reverse conversion

        Returns
        -------
        The function returns a `numpy.array` of atomic scattering factors.
        They are made up of the energy and the magnitude of the imaginary
        part of the atomic scattering factors.

        """
        prefactor = (2*E_RADIUS*h*c)
        if not reverse:
            # Convert from NEXAFS to ASF.
            factors = prefactor*NEXAFS/energies
            return factors
        else:
            factors = NEXAFS #relabel NEXAFS as factors.
            # Convert from ASF to NEXAFS.
            nexafs_reverse = factors*energies/prefactor
            return nexafs_reverse
    
    @staticmethod
    def ASF_to_NEXAFS(
        energies: npt.NDArray,
        factors: npt.NDArray,
    ) -> npt.NDArray:
        """Convert atomic scattering factors (ASF) to NEXAFS photoabsorption data.
        
        Uses `NEXAFS_to_ASF` with the `reverse` flag set to `True`.

        Parameters
        ----------
        energies : array_like
            Photon energies in eV.
        factors : array_like
            Atomic scattering factors.

        Returns
        -------
        np.ndarray
            NEXAFS photoabsorption data.
        """
        return conversions.NEXAFS_to_ASF(energies, factors, reverse=True)
    
    @staticmethod
    def betas_to_NEXAFS(
        energies: npt.NDArray,
        betas: npt.NDArray,
        number_density: float,
        density: float = None,
        formula_mass: float = None,
        stoichiometry: kk_stoichiometry | str = None,
    ) -> npt.NDArray:
        """
        Converts Beta values (index of refraction) to NEXAFS photoabsorption data.
        
        Uses `betas_to_asf` and `ASF_to_NEXAFS` to perform the conversion.
        
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
        factors = conversions.betas_to_ASF(energies, betas, number_density, density, formula_mass, stoichiometry)
        return conversions.ASF_to_NEXAFS(energies, factors)
    
    def to_atomic_scattering_poylnomial_coefficients(self):
        return
    
    @staticmethod
    def NEXAFS_to_betas(
        energies: npt.NDArray,
        NEXAFS: npt.NDArray,
        number_density: float,
        density: float = None,
        formula_mass: float = None,
        stoichiometry: kk_stoichiometry | str = None,
    ) -> npt.NDArray:
        """
        Converts NEXAFS photoabsorption data to Beta values (index of refraction).
        
        Uses `NEXAFS_to_ASF` and `ASF_to_betas` to perform the conversion.
        
        Parameters
        ----------
        energies : array_like
            Photon energies in eV.
        NEXAFS : array_like
            NEXAFS photoabsorption data.
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
        factors = conversions.NEXAFS_to_ASF(energies, NEXAFS)
        return conversions.ASF_to_betas(energies, factors, number_density, density, formula_mass, stoichiometry)
    
    @staticmethod
    def ASF_to_ASP(
        energies: npt.NDArray,
        factors: npt.NDArray,
        N: int = 5
    ) -> npt.NDArray:
        """
        Converts atomic scattering factors (ASF) to atomic scattering polynomial (ASP) coefficients.
        
        Calculates `N` polynomial coefficients for the spans between `factors` defined at `energies`.
        Currently only the first two coefficients are calculated (linear interpolation).
        
        Parameters
        ----------
        energies : array_like
            An array of `N` photon energies in eV.
        factors : array_like
            An array of `N` atomic scattering factors.
        N : int
            The number of coefficients to calculate. Default is 5.
            
        Returns
        -------
        npt.NDArray
            An array of `N-1` Atomic scattering polynomial coefficients.
        """
        # Ensure no duplicate energies and ordered.
        diffs = np.diff(energies)
        monotonic = np.all(diffs > 0)
        if not monotonic:
            raise ValueError(
                "Energies must be ordered and unique."
                + f" Negative differences: {np.where(diffs <= 0)}."
                )
        
        # Calculate the coefficients: setup array.
        coefs = np.zeros((len(energies)-1, N))
        # Calculate coefficient #0
        coefs[:, 0] = (factors[1:] - factors[:-1]) / (energies[1:] - energies[:-1])
        # Calculate coefficient #1
        coefs[:, 1] = factors[:-1] - coefs[:, 0] * energies[:-1]
        return coefs
    
    @staticmethod
    def ASP_to_ASF(
        energies: npt.NDArray,
        coefs: npt.NDArray,
        orders: npt.NDArray | None = None
    ) -> npt.NDArray:
        """
        Converts atomic scattering polynomial (ASP) coefficients to atomic scattering factors (ASF).
        
        Parameters
        ----------
        energies : array_like
            An array of `N` or `N+1` photon energies in eV.
            using the starting energy of each interval.
        coefs : array_like
            An array with dimension (`N`, `M`), with `N` sets of `M` atomic
            scattering polynomial coefficients.
        orders : array_like, optional
            An array of `M` integers defining the polynomial orders for each coefficient set.
            Each integer corresponds to the power of the energy term multipled by 
            the corresponding coefficient in the  polynomial, before summation to factors.
            If None, assumes coefficients in sequential order decreasing from 1.
            i.e: [1, 0, -1, -2, ...] etc.
            
        Returns
        -------
        npt.NDArray
            An array of `N` or `N+1` atomic scattering factors, matching the input `energies` length.
            If `energies` has length `N+1`, the last ASF value will be calculated using the last ASP coefficient.
        """
        energies = np.asarray(energies, dtype=float)
        coefs = np.asarray(coefs, dtype=float)
        # Check dimensions:
        if energies.ndim == 0:
            # Boost to 1D
            energies = np.array([energies])
        if coefs.ndim == 1: 
            # Boost to 2D
            coefs = np.array([coefs])
        
        # Check shapes:
        if (coefs.shape[0] != energies.shape[0]-1) and (coefs.shape[0] != energies.shape[0]):
            raise ValueError(
                f"Number of coefficients sets ({len(coefs)}) "
                + f"does not match the number of energies ({len(energies)-1} or {len(energies)}).")
        
        # Create an array of energy powers for each coefficient.
        if orders is not None:
            if orders.ndim ==1 and orders.shape[0] == coefs.shape[1]:
                powers = np.c_[*[energies**i for i in orders]]
            else:
                raise ValueError(
                    f"Number of orders ({orders.shape[0]}) "
                    + f"does not match the number of coefficients ({coefs.shape[1]}).")
        else:    
            powers = np.c_[*[energies**(1-i) for i in range(5)]]
        # Do energies match the number of coefficient sets?
        if energies.shape[0] == coefs.shape[0]+1:
            # Duplicate the final polynomial to define the final boundary.
            coefs = np.r_[coefs, coefs[-1:,:]] # Duplicate the last row.
        return np.sum(coefs * powers, axis=1)