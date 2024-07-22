"""
'Piecewise polynomial' representation models of scattering factors.
"""

import numpy as np
import numpy.typing as npt
from collections.abc import Iterator
from typing import TYPE_CHECKING

from kkcalc.util import doc_copy
from kkcalc.models.conversions import conversions
if TYPE_CHECKING:
    from kkcalc.models.factors import asf as asf_type

class asp:
    """
    A generic container for a piecewise polynomial representation of scattering factors
    (atomic scattering polynomial). 
    
    Allows the evaluation of the scattering factors at specified energies, by calling 
    the object or using the `evaluate_energies` method.
    
    Attributes
    """
    def __init__(self, energies, coefs):
        # Convert inputs to numpy arrays is not already
        energies = np.asarray(energies)
        coefs = np.asarray(coefs)
        if energies.ndim != 1:
            raise ValueError("Energies must be a 1D array.")
        if coefs.ndim != 2:
            raise ValueError("Coefficients must be a 2D array.")
        
        # Check energies are monotonic
        diff_sign = np.diff(energies) > 0 #True = Positive, False = Negative.
        if not np.all(diff_sign):
            raise ValueError("Energies must be in increasing order. Indexes of non-monotonic values: ", np.where(~diff_sign)[0])
        
        # Check input dimensions match
        if len(energies) != len(coefs) + 1:
            raise ValueError(
                f"Pairs of energies define the intervals for each set of polynomial coefficients." +
                f"Number of coefficients ({len(coefs)}) does not match the number of energies ({len(energies)})+1.")
            
        # Set properties
        self._energies = energies
        self._coefs = coefs
        
    @property
    def energies(self) -> npt.NDArray:
        """
        Returns the interval energy values, between which the `coefs` are defined.

        Returns
        -------
        npt.NDArray
            An array of energy values with length N+1, where N is the number of segments.
        """
        return self._energies
    
    @property
    def coefs(self) -> npt.NDArray:
        """
        Returns the polynomial coefficients for the scattering factor, defined on the intervals of `energies`.

        Returns
        -------
        npt.NDArray
            A 2D array, where rows correspond to the segments defined by `energies`, and columns are the polynomial coefficients.
        """
        return self._coefs
    
    @staticmethod
    @doc_copy(conversions.ASP_to_ASF)
    def coefs_to_atomic_scattering_factors(
        energies:npt.NDArray, 
        coefs:npt.NDArray) -> npt.NDArray:
        r"""
        Alias for `conversions.ASP_to_ASF` to calculate the atomic scattering factors from
        polynomial `coefs` defined between `energies`.
        """
        return conversions.ASP_to_ASF(energies, coefs)
    
    @property
    def atomic_scattering_factors(self) -> npt.NDArray:
        """
        Returns the atomic scattering factors calculated from the piecewise polynomial coefficients.
        
        Returns
        -------
        npt.NDArray
            The atomic scattering factors calculated from the polynomial coefficients.
        """
        return self.coefs_to_atomic_scattering_factors(self.energies, self.coefs)
    
    @property
    @doc_copy(atomic_scattering_factors)
    def asf(self) -> npt.NDArray:
        """
        Alias for `atomic_scattering_factors`.
        """
        return self.atomic_scattering_factors
    
    def to_atomic_scattering_factors(self) -> "asf_type":
        """
        Converts the piecewise polynomial representation to an atomic scattering factor object.
        
        Returns
        -------
        asf
            An atomic scattering factor object with the same polynomial coefficients as the piecewise polynomial.
        """
        if not "asf" in locals():
            from kkcalc.models.factors import asf as asf_type
        return asf_type(energies=self.energies,
                   factors=self.atomic_scattering_factors)
        
    @doc_copy(to_atomic_scattering_factors)
    def to_asf(self) -> "asf_type":
        """
        Alias for `to_atomic_scattering_factors`.
        """
        return self.to_atomic_scattering_factors()
    
    def evaluate_energies(self, 
            target_energies:npt.NDArray | float | None = None
            ) -> npt.NDArray:
        r"""
        Calculate Henke scattering factors from object polynomial coefficients at desired `energies`.
        
        Uses `coefs_to_atomic_scattering_factors` to calculate the ASF values after matching energies to segments.

        Parameters
        ----------
        energies : array_like | float, optional
            1D array (or singular) of `M` energies in eV, if None then the energies defined in the object are used.

        Returns
        -------
        npt.NDArray
            The magnitude of the atomic scattering factors at energy (or energies) `energies`.
            Dimensions are `M` if `energies` is an array, or 1 if `energies` is a singular value.

        """
        # Type check
        if target_energies is not None:
            if not isinstance(target_energies, (int, float)):
                target_energies = np.asarray(target_energies)
            else:
                target_energies = np.array([target_energies])
        
        # If no energies or coefficients are provided, use the object's values to return the intrinsic ASF values.
        if target_energies is None:
            return self.evaluate_energies(self.energies, self.coefs)
        # If energies are provided, but not coefficients, use the object's coefficients to calculate the ASF values at the given energies.
        else: #target_energies is not None:
            # Find where the energies are located in the object's energies.
            indices = np.searchsorted(self.energies, target_energies) - 1 # subtract to transfer from spans (N+1) to coefficients (N).
            if -1 in indices or len(self.energies)-1 in indices:
                # Check if all searchsorted invalid indexes are defined.
                invalid = np.where((indices < 0) | (indices == len(self.energies)-1))
                inval_energies = target_energies[invalid]
                for inv_e in inval_energies:
                    if inv_e == target_energies[0]:
                        indices[invalid] = 0 # First defined polynomial.
                    elif inv_e == target_energies[-1]:
                        indices[invalid] = len(self.energies)-2 # Last defined polynomial.
                    else:
                        raise ValueError(
                            f"Some energies {target_energies[invalid]}"
                            + f"are outside the defined energy range ({self.energies.min()}, {self.energies.max()})."
                        )
                
            # Collate coefficients corresponding to the energies.
            target_coefs = self.coefs[indices]
            # Calculate the ASF values at the given energies.
            return self.coefs_to_atomic_scattering_factors(target_energies, target_coefs)

    @doc_copy(evaluate_energies)
    def __call__(self, target_energies:npt.NDArray | float | None = None) -> npt.NDArray:
        """
        Callable alias for `evaluate_energies`.
        """
        return self.evaluate_energies(target_energies)
    
    def __iter__(self) -> Iterator[tuple[tuple[float, float], np.ndarray]]:
        """
        Provides each segment and piecewise polynomial coefficients of the energy-dependent scattering amplitude.

        Yields
        ------
        segment : tuple[float, float]
            The energy interval for which the polynomial coefficients are valid.
        poly_coefs : np.ndarray
            The polynomial coefficients for the scattering factor in the given energy interval.
        """
        for i in range(len(self._energies)-1):
            yield (self.energies[i], self.energies[i+1]), self.coefs[i]
    
    def __getitem__(self, i: int) -> tuple[tuple[float, float], np.ndarray]:
        """
        Returns segment energies and polynomial coefficients at the indexed energy interval.

        Parameters
        ----------
        index : int
            The index of the segment to return.

        Returns
        -------
        interval : tuple[float, float],
            The energy interval for which the polynomial coefficients are valid.
        coefficients : np.ndarray
            The polynomial coefficients for the scattering factor in the given energy interval.
        """
        if i < 0 or i >= len(self._energies)-2:
            raise IndexError(f"Index {i} out of range for {len(self._energies)-1} segments.")
        return (self.energies[i], self.energies[i+1]), self.coefs[i]
    
    def __str__(self) -> str:
        return f"Scattering factors with {len(self._energies)-1} segments"
    
    def __repr__(self) -> str:
        return f"piecewise_polynomail({self._energies}, {self._coefs})"
    
    def __len__(self) -> int:
        return len(self._coefs)
