"""
This module contains the Kramers-Kronig methods.
"""
import math
import numpy as np
import numpy.typing as npt
import warnings
from typing import Any
from .stoich import stoichiometry
from enum import Enum
from .models.factors import KK_Datatype, asf
from kkcalc.asf_database.asf_polynomials import asp_db, asp_db_extended
        
class kk_algorithms:
    """
    This class contains the static Kramers-Kronig algorithms.
    """
    
    @staticmethod
    def KK_General_PP(eval_energies: npt.NDArray, 
                    energies: npt.NDArray, 
                    imaginary_spectrum: npt.NDArray, 
                    orders : npt.NDArray, 
                    relativistic_correction : float = 0
        ) -> npt.NDArray:
        """
        Calculate Kramers-Kronig transform with 'Piecewise Polynomial'
        algorithm plus the Biggs and Lighthill extended data.

        .. math:: 
            equation

        Parameters
        ----------
        Eval_Energy : array_like
            Set of photon energies describing points at which to evaluate the real spectrum
        Energy : array_like
            Set of photon energies describing intervals for which each row of `imaginary_spectrum` is valid.
        imaginary_spectrum : array_like
            A 2D array, consisting of columns of polynomial coefficients belonging to the power terms indicated by 'order'.
        orders : array_like
            The vector represents the polynomial indices corresponding to the columns of imaginary_spectrum
        relativistic_correction : float
            The relativistic correction to the Kramers-Kronig transform.
            You can calculate the value using the `calc_relativistic_correction` function.

        Returns
        -------
        array_like
                This function returns the real part of the scattering factors evaluated at photon energies specified by Eval_Energy.
        """
        # Convert inputs to numpy arrays is not already
        eval_energies = np.asarray(eval_energies)
        energies = np.asarray(energies)
        imaginary_spectrum = np.asarray(imaginary_spectrum)
        orders = np.asarray(orders)

        # Check input dimensions match
        if imaginary_spectrum.shape[0] != energies.shape[0]-1:
                raise ValueError(f"First axis of imaginary_spectrum must have one less row ({imaginary_spectrum.shape[0]}) than energies ({energies.shape[0]})")
        if imaginary_spectrum.shape[1] != orders.shape[0]:
                raise ValueError(f"Second axis of imaginary_spectrum must have the same number of columns ({imaginary_spectrum.shape[1]}) as orders ({orders.shape[0]})")

        # Need to build arrays with dimensions X-E-n [Energies, Evaluation Energies, Orders]
        C = np.tile(imaginary_spectrum[:,np.newaxis,:],         (1, len(eval_energies), 1)) # 2D to 3D
        N = np.tile(orders[np.newaxis,np.newaxis,:],            (len(energies) - 1, len(eval_energies),1))
        X = np.tile(energies[:,np.newaxis,np.newaxis],          (1, len(eval_energies), len(orders))) # 1D to 3D
        E = np.tile(eval_energies[np.newaxis, :, np.newaxis],   (len(energies) - 1, 1, len(orders))) #1d TO 3d
        # Calculate when evaluating energies are equal to the data energies, in a boolean array (1 or 0)
        poles = np.equal(X, 
            np.tile(eval_energies[np.newaxis, :, np.newaxis],   (len(energies), 1, len(orders))
        ))    

        # Basic integral, the resulting shape matches the evaluation energies
        Integral: npt.NDArray = np.sum(
            -C*(-E)**N * np.log(np.abs(
                (X[1:,:,:] + E) / (X[:-1,:,:] + E) # X_{i+1} / X_i
                - (C*E**N*(1-poles[1:,:,:]) # If a pole, then zero contribution
                   *np.log(np.abs(
                    # (X-E)_{i+1} / (X-E)_i
                    (X[1:,:,:] - E + poles[1:,:,:]) # Non-zero if a pole, just to avoid log(0)
                    / ( # Divide by X-E, unless a pole
                        X[:-1,:,:] - E + poles[:-1,:,:] # Add 1 for poles.
                        # Original 
                        # X[:-1,:,:] * (1-poles[:-1,:,:]) # If a pole, then zero
                        # + poles[:-1,:,:] * X[[0, *range(energies.shape[0]-2)],:,:] #
                        # - E
                    )
                )))
            ))
            ,axis=(0,2)
        )
        
        ### Calculate the Kramers-Kronig integral additional terms
        if np.any(orders<=-2): # N<=-2, ln(x) terms
            i = [slice(None,None,None),slice(None,None,None),orders<=-2]
            Integral += np.sum(
                C[i] * ((-E[i])**N[i]+E[i]**N[i]) * np.log(
                    np.absolute(
                        (X[1:,:,orders<=-2])/(X[:-1,:,orders<=-2])
                    )    
                ),axis=(0,2))

        if np.any(orders>=0): # N>=0,  x^k terms
            for ni in np.where(orders>=0)[0]:
                i = [slice(None,None,None),slice(None,None,None),ni]
                n = orders[ni]
                for k in range(n,0,-2):
                    Integral += np.sum(
                        C[i] / float(-k)*2*E[i]**(n-k)*(
                            X[1:,:,ni]**k - X[:-1,:,ni]**k
                        ),axis=0)

        if np.any(orders <=-3): # N<=-3, x^k terms
            for ni in np.where(orders<=-3)[0]:
                i = [slice(None,None,None),slice(None,None,None),ni]
                n = orders[ni]
                for k in range(n+2,0,2):
                    Integral += np.sum(
                        C[i] / float(k)*((-1)**(n-k)+1)*E[i]**(n-k)*(
                            X[1:,:,ni]**k - X[:-1,:,ni]**k
                        ),axis=0)
        return Integral / math.pi + relativistic_correction
        

    def KK_PP(eval_energies, energies, imaginary_spectrum, relativistic_correction):
        """Calculate Kramers-Kronig transform with "Piecewise Polynomial"
        algorithm plus the Biggs and Lighthill extended data.

        Parameters
        ----------
        Eval_Energy : numpy vector of `float`
            Set of photon energies describing points at which to evaluate the real spectrum
        Energy : numpy vector of `float`
            Set of photon energies describing intervals for which each row of `imaginary_spectrum` is valid
        imaginary_spectrum : two-dimensional `numpy.array` of `float`
            The array consists of five columns of polynomial coefficients: A_1, A_0, A_-1, A_-2, A_-3
        relativistic_correction : float
            The relativistic correction to the Kramers-Kronig transform.
            You can calculate the value using the `calc_relativistic_correction` function.

        Returns
        -------
        This function returns the real part of the scattering factors evaluated at photon energies specified by Eval_Energy.

        """
        X1 = energies[0:-1]
        X2 = energies[1:]
        E = np.tile(eval_energies, (len(energies)-1, 1)).T
        Full_coeffs = imaginary_spectrum.T
        Symb_1 = (( Full_coeffs[0, :]*E+Full_coeffs[1, :])*(X2-X1)+0.5*Full_coeffs[0, :]*(X2**2-X1**2)-(Full_coeffs[3, :]/E+Full_coeffs[4, :]*E**-2)*numpy.log(numpy.absolute(X2/X1))+Full_coeffs[4, :]/E*(X2**-1-X1**-1))
        Symb_2 = ((-Full_coeffs[0, :]*E+Full_coeffs[1, :])*(X2-X1)+0.5*Full_coeffs[0, :]*(X2**2-X1**2)+(Full_coeffs[3, :]/E-Full_coeffs[4, :]*E**-2)*numpy.log(numpy.absolute(X2/X1))-Full_coeffs[4, :]/E*(X2**-1-X1**-1))+(Full_coeffs[0, :]*E**2-Full_coeffs[1, :]*E+Full_coeffs[2, :]-Full_coeffs[3, :]*E**-1+Full_coeffs[4, :]*E**-2)*numpy.log(numpy.absolute((X2+E)/(X1+E)))
        Symb_3 = (1-1*((X2==E)|(X1==E)))*(Full_coeffs[0, :]*E**2+Full_coeffs[1, :]*E+Full_coeffs[2, :]+Full_coeffs[3, :]*E**-1+Full_coeffs[4, :]*E**-2)*numpy.log(numpy.absolute((X2-E+1*(X2==E))/(X1-E+1*(X1==E))))
        Symb_B = np.sum(Symb_2 - Symb_1 - Symb_3, axis=1)  # Sum areas for approximate integral
        # Patch singularities
        hits = energies[1:-1]==E[:,0:-1]
        E_hits = numpy.append(numpy.insert(numpy.any(hits, axis=0),[0,0],False),[False,False])
        Eval_hits = numpy.any(hits, axis=1)
        X1 = energies[E_hits[2:]]
        XE = energies[E_hits[1:-1]]
        X2 = energies[E_hits[:-2]]
        C1 = Full_coeffs[:, E_hits[2:-1]]
        C2 = Full_coeffs[:, E_hits[1:-2]]
        Symb_singularities = numpy.zeros(len(eval_energies))
        Symb_singularities[Eval_hits] = (C2[0, :]*XE**2+C2[1, :]*XE+C2[2, :]+C2[3, :]*XE**-1+C2[4, :]*XE**-2)*numpy.log(numpy.absolute((X2-XE)/(X1-XE)))
        # Finish things off
        KK_Re = (Symb_B-Symb_singularities) / (math.pi*eval_energies) + relativistic_correction
        logger.debug("Done!")
        return KK_Re
        
def kk_calculate_real(
    energies: npt.NDArray,
    intensities: npt.NDArray,
    formula : stoichiometry | str, 
    load_options=None,
    input_data_type: KK_Datatype = None,
    merge_domain:tuple[float, float]=None,
    fix_distortions:bool=False,
    curve_tolerance=None,
    curve_recursion=50):
    """
    """
    # Prepare data
    #     data = None
    #     if NEXAFS_data is not None:
    #         data = NEXAFS_data
    #         if energies:
    #             raise warnings.warn("`NEXAFS_data` and `energies` are both provided. Only `NEXAFS_data` will be used.")
    #         if intensities:
    #             raise warnings.warn("`NEXAFS_data` and `intensities` are both provided. Only `NEXAFS_data` will be used.")
    #     elif energies is not None and intensities is not None and len(energies) == len(intensities):
    # data = np.c_[energies, intensities]
    
    # Verify shapes
    if energies.shape != intensities.shape:
        raise ValueError(f"Shape mismatch between energies ({energies.shape}) and intensities ({intensities.shape})")
    
    # Use the stoichiometry to get the relativistic correction and database atomic scattering polynomial
    stoich = formula if isinstance(formula, stoichiometry) else stoichiometry.from_chemical_formula(formula)
    rc = stoich.relativistic_correction
    db_poly: asp_db = stoich.asp_im() #database
    
    # Load the NEXAFS data.
    data_asf = asf(energies, intensities)
    
    # Combine the data with the database.
    merge_data_asp = asp_db_extended(
            data_asf=data_asf,
            asp_db = db_poly,
            merge_domain=merge_domain,
            fix_distortions=fix_distortions
    )
    
    
    
    if curve_tolerance is not None and False:
        # output_data = 
        pass    
    
    Relativistic_Correction = calc_relativistic_correction(Stoichiometry)
    Full_E, Imaginary_Spectrum = data.calculate_asf(Stoichiometry)
    if NearEdgeDataFile is not None:
            NearEdge_Data = data.convert_data(data.load_data(NearEdgeDataFile, load_options),FromType=input_data_type,ToType='asf')
            Full_E, Imaginary_Spectrum = data.merge_spectra(NearEdge_Data, Full_E, Imaginary_Spectrum, merge_points=merge_domain, add_background=add_background, fix_distortions=fix_distortions)
    Real_Spectrum = KK_PP(Full_E, Full_E, Imaginary_Spectrum, Relativistic_Correction)
    if curve_tolerance is not None:
            output_data = improve_accuracy(Full_E,Real_Spectrum,Imaginary_Spectrum, Relativistic_Correction, curve_tolerance, curve_recursion)
    else:
            Imaginary_Spectrum_Values = data.coeffs_to_ASF(Full_E, numpy.vstack((Imaginary_Spectrum,Imaginary_Spectrum[-1])))
            output_data = numpy.vstack((Full_E,Real_Spectrum,Imaginary_Spectrum_Values)).T
    return output_data

        
# def kk_calculate_real(NearEdgeDataFile, ChemicalFormula, load_options=None, input_data_type=None, merge_points=None, add_background=False, fix_distortions=False, curve_tolerance=None, curve_recursion=50):
#         """Do all data loading and processing and then calculate the kramers-Kronig transform.
#         Parameters
#         ----------
#         NearEdgeDataFile : string
#                 Path to file containg near-edge data
#         ChemicalFormula : string
#                 A standard chemical formula string consisting of element symbols, numbers and parentheses.
#         merge_points : list or tuple pair of `float` values, or None
#                 The photon energy values (low, high) at which the near-edge and scattering factor data values
#                 are set equal so as to ensure continuity of the merged data set.

#         Returns
#         -------
#         This function returns a numpy array with columns consisting of the photon energy, the real and the imaginary parts of the scattering factors.
#         """
#         Stoichiometry = data.ParseChemicalFormula(ChemicalFormula)
#         Relativistic_Correction = calc_relativistic_correction(Stoichiometry)
#         Full_E, Imaginary_Spectrum = data.calculate_asf(Stoichiometry)
#         if NearEdgeDataFile is not None:
#                 NearEdge_Data = data.convert_data(data.load_data(NearEdgeDataFile, load_options),FromType=input_data_type,ToType='asf')
#                 Full_E, Imaginary_Spectrum = data.merge_spectra(NearEdge_Data, Full_E, Imaginary_Spectrum, merge_points=merge_points, add_background=add_background, fix_distortions=fix_distortions)
#         Real_Spectrum = KK_PP(Full_E, Full_E, Imaginary_Spectrum, Relativistic_Correction)
#         if curve_tolerance is not None:
#                 output_data = improve_accuracy(Full_E,Real_Spectrum,Imaginary_Spectrum, Relativistic_Correction, curve_tolerance, curve_recursion)
#         else:
#                 Imaginary_Spectrum_Values = data.coeffs_to_ASF(Full_E, numpy.vstack((Imaginary_Spectrum,Imaginary_Spectrum[-1])))
#                 output_data = numpy.vstack((Full_E,Real_Spectrum,Imaginary_Spectrum_Values)).T
#         return output_data