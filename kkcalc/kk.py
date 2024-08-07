"""
This module contains the Kramers-Kronig methods.
"""
import math
import numpy as np
import numpy.typing as npt
import warnings
from kkcalc.stoich import stoichiometry
from kkcalc.asf_database.db_models import asp_db, asp_db_extended
from kkcalc.models.conversions import conversions
from kkcalc.models.factors import KK_Datatype, asf
from kkcalc.models.polynomials import asp
        
class kk_algorithms:
    """
    This class contains the static Kramers-Kronig algorithms.
    """
    
    @staticmethod
    def KK_General_PP(eval_energies: npt.NDArray, 
                    energies: npt.NDArray, 
                    imaginary_spectrum: npt.NDArray, 
                    orders : npt.NDArray | None = None,
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
        orders : array_like, optional
            The vector represents the polynomial indices corresponding to the columns of imaginary_spectrum.
            By default assumes [1, 0, -1, ...] for the columns of imaginary_spectrum.
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
        if orders is None:
            orders = np.linspace(1, 2-imaginary_spectrum.shape[1], imaginary_spectrum.shape[1], dtype=int)
        else:
            orders = np.asarray(orders)

        # Check input dimensions match
        if imaginary_spectrum.shape[0] != energies.shape[0]-1:
                raise ValueError(f"First axis of imaginary_spectrum must have one less row ({imaginary_spectrum.shape[0]}) than energies ({energies.shape[0]})")
        if imaginary_spectrum.shape[1] != orders.shape[0]:
                raise ValueError(f"Second axis of imaginary_spectrum must have the same number of columns ({imaginary_spectrum.shape[1]}) as orders ({orders.shape[0]})")

        # Need to build arrays with dimensions X-E-n [Energies, Evaluation Energies, Orders]
        C = np.tile(imaginary_spectrum[:,np.newaxis,:],         (1, len(eval_energies), 1)) # 2D to 3D
        N = np.tile(orders[np.newaxis,np.newaxis,:],            (len(energies) - 1, len(eval_energies),1)) #1D to 3D
        X = np.tile(energies[:,np.newaxis,np.newaxis],          (1, len(eval_energies), len(orders))) # 1D to 3D
        E = np.tile(eval_energies[np.newaxis, :, np.newaxis],   (len(energies) - 1, 1, len(orders))) #1D TO 3D
        # Calculate when evaluating energies are equal to the data energies, in a boolean array (1 or 0)
        poles = np.equal(X, 
            np.tile(eval_energies[np.newaxis, :, np.newaxis],   (len(energies), 1, len(orders))
        ))

        # Basic integral, the resulting shape matches the evaluation energies
        integral: npt.NDArray = np.sum(
            -C*(-E)**N * np.log(np.abs(
                (X[1:,:,:] + E) / (X[:-1,:,:] + E) # X_{i+1} / X_i
            ))

            - C*(E)**N * (1-poles[1:,:,:])  # If a pole in numerator or denominator, then zero contribution 
            # Need to multiply by something to prevent contribution at i=0. 
            # * np.array([0] + [1] * (poles.shape[0]-2))[:, np.newaxis, np.newaxis] #* (1-poles[:-1,:,:])
            * np.log(np.abs(
                # `(X_{i+1}-E) / (X_i-E)` if not a pole, else `(X_{i+1}-E) / (X_{i-1}-E)`
                (X[1:,:,:] - E + poles[1:,:,:]) # Non-zero if a pole, just to avoid log(0)
                / (
                    # Alternative 
                    X[:-1,:,:] * (1-poles[:-1,:,:]) # If not a pole, then use X
                    + poles[:-1,:,:] * np.r_[X[np.newaxis,0], X[0:-2]] #If a pole, then use X-1 value prior (unless at X[0]).
                    - E
                    # + np.r_[(poles[0] * poles[1])[np.newaxis,:], np.zeros(poles[3:,].shape)] # If poles[0] or poles[1] is a pole, then add small amount.
                    # Original
                    # X[:-1,:,:] * (1-poles[:-1,:,:]) # If not a pole, then use X
                    # + poles[:-1,:,:] * X[[0, *range(energies.shape[0]-2)],:,:] #If a pole, then use X-1 value prior (unless at X[0]).
                    # - E
                    # TODO Prevent divide by zero at i = 0
                    # + np.array([1] + [0] * (poles.shape[0]-2))[:, np.newaxis, np.newaxis] # Prevent divide by zero at pole at i = 0
                )
            )) * (1 - np.r_[(poles[0] * poles[1])[np.newaxis,:], np.zeros(poles[2:,].shape)]) # If poles[0] or poles[1] is a pole, zero value.
            ,axis=(0,2)
        )
        
        ### Calculate the Kramers-Kronig integral additional terms
        if np.any(orders<=-2): # N<=-2, ln(x) terms
            i = (slice(None,None,None),slice(None,None,None), orders<=-2)
            integral += np.sum(
                C[*i] * ((-E[*i])**N[*i]+E[*i]**N[*i]) * np.log(
                    np.absolute(
                        (X[1:,:,orders<=-2])/(X[:-1,:,orders<=-2])
                    )    
                ),axis=(0,2))

        if np.any(orders>=0): # N>=0,  x^k terms
            for ni in np.where(orders>=0)[0]:
                i = [slice(None,None,None),slice(None,None,None),ni]
                n = orders[ni]
                for k in range(n,0,-2):
                    integral += np.sum(
                        C[*i] / float(-k)*2*E[*i]**(n-k)*(
                            X[1:,:,ni]**k - X[:-1,:,ni]**k
                        ),axis=0)

        if np.any(orders <=-3): # N<=-3, x^k terms
            for ni in np.where(orders<=-3)[0]:
                i = [slice(None,None,None),slice(None,None,None),ni]
                n = orders[ni]
                for k in range(n+2,0,2):
                    n = n.astype(float)
                    integral += np.sum(
                        C[*i] / float(k)
                        *((-1)**(n-k)+1)*E[*i]**(n-k)*(
                            X[1:,:,ni]**k - X[:-1,:,ni]**k
                        ),axis=0)
        return integral / math.pi + relativistic_correction
        

    def KK_PP(
        target_energies: npt.NDArray,
        energies: npt.NDArray,
        imag_coefs: npt.NDArray,
        relativistic_correction: float
        ) -> npt.NDArray:
        """
        Calculate Kramers-Kronig transform from imaginary coefficients at evaluation energies, 
        with "Piecewise Polynomial" algorithm plus the Biggs and Lighthill extended data.

        Parameters
        ----------
        target_energies : np.ndarray
            Set of photon energies describing points at which to evaluate the real spectrum with shape (N,)
        energies : np.ndarray
            Set of photon energies with shape (`M+1`,) describing intervals for which each row of `imag_coefs` is valid 
        imag_coefs : np.ndarray
            The 2D array with shape (`M`, 5) consists of columns of polynomial coefficients: A_1, A_0, A_-1, A_-2, A_-3
            defined between the `M+1` energies in `energies`.
        relativistic_correction : float
            The relativistic correction to the Kramers-Kronig transform.
            This is calcuable using `kkcalc.stoich.relativistic_correction_eq`.

        Returns
        -------
        This function returns the real part of the scattering factors evaluated at photon energies specified by Eval_Energy.

        """
        # if np.all(target_energies == energies):
        #     # If every target energy is already in the energy list, ... TODO
        #     raise NotImplementedError(
        #         "This function is not yet implemented for the case where every target energy is in the energy list."
        #     )
        # else:
        # M is the number of polynomial energy spans, N is the number of target energies.
        X1 = energies[0:-1] # M 
        X2 = energies[1:] # M
        E = np.tile(target_energies, (len(energies)-1, 1)).T # Results in a 2D of shape (N, M)
        coefs_T = imag_coefs.T
        # 
        Symb_1 = (
            (coefs_T[0, :]*E + coefs_T[1, :])*(X2-X1)
            + 0.5*coefs_T[0, :]*(X2**2-X1**2)
            - (coefs_T[3, :]/E + coefs_T[4, :] * E**-2)*np.log(np.abs(X2/X1))
            + coefs_T[4, :]/E*(X2**-1-X1**-1))
        #
        Symb_2 = (
                (-coefs_T[0, :]*E + coefs_T[1, :])*(X2-X1)
                + 0.5*coefs_T[0, :]*(X2**2-X1**2)
                + (coefs_T[3, :]/E - coefs_T[4, :] * E**-2)*np.log(np.abs(X2/X1))
                - coefs_T[4, :]/E*(X2**(-1)-X1**(-1))
            ) + (
                coefs_T[0, :]*E**2
                - coefs_T[1, :]*E
                + coefs_T[2, :]
                - coefs_T[3, :]*E**-1
                + coefs_T[4, :]*E**-2
            ) * np.log(np.abs((X2+E)/(X1+E)))
        #
        Symb_3 = (
            (1-1*((X2==E)|(X1==E)))
            * (coefs_T[0, :]*E**2 
            + coefs_T[1, :]*E 
            + coefs_T[2, :] 
            + coefs_T[3, :]*E**-1
            + coefs_T[4, :]*E**-2)
            * np.log(np.abs(
                (X2-E+1*(X2==E)) / (X1-E+1*(X1==E))
                )
            )
        )
        # Sum areas for approximate integral
        Symb_B = np.sum(Symb_2 - Symb_1 - Symb_3, axis=1)
        
        # Patch singularities
        singularities = energies[1:-1]==E[:,0:-1]
        E_sing = np.append(np.insert(np.any(singularities, axis=0),[0,0],False),[False,False])
        Eval_sing = np.any(singularities, axis=1)
        X1 = energies[E_sing[2:]]
        XE = energies[E_sing[1:-1]]
        X2 = energies[E_sing[:-2]]
        # C1 = Full_coeffs[:, E_sing[2:-1]] # Not used... why?
        C2 = coefs_T[:, E_sing[1:-2]]
        Symb_singularities = np.zeros(len(target_energies))
        Symb_singularities[Eval_sing] = (
            C2[0, :]*XE**2
            + C2[1, :]*XE
            + C2[2, :]
            + C2[3, :]*XE**-1
            + C2[4, :]*XE**-2
        )*np.log(np.abs((X2-XE)/(X1-XE)))
        # Finish things off
        KK_Re = (Symb_B-Symb_singularities) / (math.pi*target_energies) + relativistic_correction
        return KK_Re
    
    def KK_PP_inv(
        target_energies: npt.NDArray,
        energies : npt.NDArray,
        real_spectrum: npt.NDArray):
        """Calculate inverse Kramers-Kronig transform with "Piecewise Polynomial"
        algorithm plus the Biggs and Lighthill extended data.

        Parameters
        ----------
        target_energies : array_like
            1D Array of photon energies with shape (M,) describing points at which 
            to evaluate the imaginary spectrum.
        energies : array_like
            1D Array of photon energies with shape (N+1,) describing intervals for
            which each row of `real_spectrum` coefficients are defined.
        real_spectrum : array_like
            2D Array of polynomial coefficients with shape (N, 5) for the real part of the
            scattering factors. Coefficients correspond to powers [1, 0, -1, -2, -3].

        Returns
        -------
        imag_spectrum : array_like
            1D Array of imaginary scattering factors with shape (M,) evaluated at
            photon energies specified by `target_energies`.
        """
        ## Inverse KK is only a minor modification of the forward algorithm
        return -target_energies*kk_algorithms.KK_PP(
            target_energies = target_energies,
            energies = energies,
            imag_coefs = np.roll(real_spectrum,1,axis=1),
            relativistic_correction = 0
        )
        
    @staticmethod
    def improve_accuracy(energies: npt.NDArray,
                         real: npt.NDArray,
                         imag_coefs: npt.NDArray,
                         relativistic_correction: float,
                         tolerance: float,
                         max_iter: int = 50
                         ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Calculates extra data points so that the Kramers-Kronig transform is more accurate.
        
        Parameters
        ----------
        energies : npt.NDArray
            The photon energies.
        real : npt.NDArray
            The real part of the scattering factors.
        imag_coefs : npt.NDArray
            Polynomial coefficients representing the imaginary part of the scattering factors.
        relativistic_correction : float
            The relativistic correction to the Kramers-Kronig transform.
        tolerance : float
            The level of error allowed in the linear extrapolation.
        max_iter : int
            The maximum number of iterations an energy interval can be split.
            
        Returns
        -------
        energies : npt.NDArray
            The photon energies of length N.
        real : npt.NDArray
            The real part of the scattering factors, of length N.
        imag : npt.NDArray  
            Imaginary coefficients representing the scattering factors, of length N-1 for each energy segment.
        """
        
        # An array of indexes 1 to N-1 (every single midpoint). Midpoints calculated as x[i] + x[i-1] / 2.
        idx_extra = np.linspace(1, energies.shape[0]-1, energies.shape[0]-1, dtype=int)
        # data
        imag = conversions.ASP_to_ASF(energies, imag_coefs)
        
        # Iterate until the error is below the tolerance, or the maximum number of iterations is reached
        for i in range(max_iter):
            # Get energy midpoints
            en_mid = (energies[idx_extra] + energies[idx_extra-1]) / 2
            # Calculate new midpoint values
            re_mid = kk_algorithms.KK_PP(
                eval_energies=en_mid,
                energies=energies,
                imaginary_spectrum=imag_coefs,
                relativistic_correction=relativistic_correction
            )
            im_mid = asp.evaluate_energies_on_coefs(
                target_energies=en_mid, 
                energies=energies,
                coefs=imag_coefs
            )
            
            # Evaluate new values to the average of the old values. If coefs are linear, this will be zero. 
            # Difference from linear is the error, bigger is better for finding new corrections.
            im_err = np.abs(im_mid - (imag[idx_extra] + imag[idx_extra-1]) / 2)
            re_err = np.abs(re_mid - (real[idx_extra] + real[idx_extra-1]) / 2)
            
            # Boolean for improvement - newly evaluated points have a change greater than the tolerance
            improved = (im_err > tolerance) | (re_err > tolerance)
            
            # Insert new points and values where errors are big.
            if i==0:
                print(f"New energies: {en_mid[improved]}")
                print(f"Existing energies: {energies}")
            energies = np.insert(energies, idx_extra[improved], en_mid[improved])
            if i==0:
                print(f"Updated energies: {energies}")
            imag = np.insert(imag, idx_extra[improved], im_mid[improved])
            real = np.insert(real, idx_extra[improved], re_mid[improved])
            
            # Check if at satisfactory level
            if np.sum(improved) == 0:
                # Return values if no improvements are made
                return energies, real, imag
            else:
                ## Setup new indexes for the next iteration, use indexes where improvement was made
                ## Requires midpoint calculation either-side of improved midpoint
                # 1. Duplicate coefficients so that energy segments are the same length as the energies - 1
                imag_coefs = np.insert(imag_coefs, idx_extra[improved], imag_coefs[idx_extra[improved]], axis=0)
                # 2. Create new indexes, where to perform midpoint calcs
                new_midpoint_locs = np.insert(
                    arr=    np.zeros(idx_extra.shape[0], dtype=bool), # Copy existing midpoint list
                    obj=    idx_extra[improved], # Add the locations where improvements were made 
                    values= True # Insert True
                )
                new_midpoint_locs = new_midpoint_locs | np.roll(new_midpoint_locs, 1) # Add the point after the new points
                idx_extra = np.where(new_midpoint_locs) # don't need to create condition, already boolean array.

        # Return values if the maximum number of iterations is reached
        return energies, real, imag
        
def kk_calculate_real(
    energies: npt.NDArray,
    intensities: npt.NDArray,
    formula : stoichiometry | str,
    input_data_type : KK_Datatype = None,
    merge_domain:tuple[float, float]=None,
    fix_distortions:bool=False,
    tolerance=None,
    max_iter=50) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Legacy function to calculate the real part of the scattering factors from the imaginary part.
    
    Instead use `kkcalc.models.factors.asp_im` object and the method `asf.kk_calculate_real`.
    
    Parameters
    ----------
    energies : npt.NDArray
        The photon energies, with shape (N,)
    intensities : npt.NDArray
        The imaginary component intensities, with shape (N,)
    formula : stoichiometry | str
        The chemical formula of the material.
    input_data_type : KK_Datatype
        The type of data provided. Default is KK_Datatype.UNDEFINED.
        
    """
    
    # Verify shapes
    if energies.shape != intensities.shape:
        raise ValueError(f"Shape mismatch between energies ({energies.shape}) and intensities ({intensities.shape})")
    
    # Use the stoichiometry to get the relativistic correction and database atomic scattering polynomial
    stoich = formula if isinstance(formula, stoichiometry) else stoichiometry.from_chemical_formula(formula)
    rc = stoich.relativistic_correction
    db_poly: asp_db = stoich.asp_im() #database
    
    # Load the NEXAFS data.
    match input_data_type:
        case KK_Datatype.UNDEFINED:
            warnings.warn("No data type provided. Defaulting to NEXAFS.")
            data_asf = asf.from_NEXAFS(energies, intensities)
        case KK_Datatype.NEXAFS | KK_Datatype.XANES | KK_Datatype.PHOTOABSORPTION:
            data_asf = asf.from_NEXAFS(energies, intensities)
        case KK_Datatype.ASF:
            data_asf = asf(energies, intensities)
        case KK_Datatype.BETA:
            data_asf = asf.from_betas(energies, intensities)
        # case KK_Datatype.
        case _:
            raise ValueError(f"Invalid data type: {input_data_type}")
    
    # Combine the data with the database.
    merge_data_asp = asp_db_extended(
            data_asf=data_asf,
            asp_db = db_poly,
            merge_domain=merge_domain,
            fix_distortions=fix_distortions
    )
    # Calculate the real spectrum:
    real = kk_algorithms.KK_PP(
        target_energies=merge_data_asp.energies,
        energies=merge_data_asp.energies,
        imag_coefs=merge_data_asp.coefs,
        relativistic_correction=rc
    )
    
    if tolerance is not None:
        energies, real, imag = kk_algorithms.improve_accuracy(
            energies=merge_data_asp.energies,
            real=real,
            imag_coefs=merge_data_asp.coefs,
            relativistic_correction=rc,
            tolerance=tolerance,
            max_iter=max_iter
        )
    else:
        imag = merge_data_asp.atomic_scattering_factors
    
    return energies, real, imag


if __name__ == "__main__":
    ## Test various formulas
    # Setup graph
    import matplotlib.pyplot as plt
    plots = plt.subplots(1,2, figsize=(10,4))
    fig: plt.Figure = plots[0]
    ax: plt.Axes = plots[1][0]
    ax2: plt.Axes = plots[1][1]
    
    # Create a merge of physical data and database data
    POLYSTYRENE = "CH"
    PS_NAME = "Polystyrene"
    ps_stoich = stoichiometry(POLYSTYRENE)
    asp_db_PS = asp_db(ps_stoich)
    
    # Import Data
    import os
    data_dir = os.path.join(os.path.dirname(__file__),
                            "../examples/data")
    data_file = os.path.normpath(os.path.join(data_dir, "PS_004_-dc.txt"))
    data_PS = np.genfromtxt(data_file, skip_header=4)
    assert data_PS.shape[1] == 2, "Data file must have two columns"
    
    # Create the atomic scattering factors from NEXAFS data
    asf_PS = asf.from_NEXAFS(energies = data_PS[:,0],
                             NEXAFS = data_PS[:,1])
    
    # Combine the data with the database
    asp_db_PS_extended = asp_db_extended(
        data_asf=asf_PS,
        db_asp=asp_db_PS,
        merge_domain=(280, 320),
        # fix_distortions=False
    )
    
    asp_db_PS_extended_fixed = asp_db_extended(
        data_asf=asf_PS,
        db_asp=asp_db_PS,
        merge_domain=(280, 320),
        fix_distortions=True
    )
    
    extended_asf = asp_db_PS_extended.to_atomic_scattering_factors()
    extended_asf_fixed = asp_db_PS_extended_fixed.to_atomic_scattering_factors()
    ax.plot(extended_asf.energies, extended_asf.factors, label=f"{PS_NAME} Extended ASF")
    ax.plot(extended_asf_fixed.energies, extended_asf_fixed.factors, label=f"{PS_NAME} Extended ASF Fixed")
    db_asf = asp_db_PS.to_asf()
    ax.plot(asp_db_PS.energies, db_asf.factors, label=f"{PS_NAME} DB ASF")
    ax.set_xlim(270, 330)
    ax2.set_xlim(ax.get_xlim())
    # ax2.set_xscale("log")
    # ax2.set_ylim(450, 900)
    
    ax.set_title(PS_NAME)
    ax.legend()
    
    # Use KK algorithms
    gen_pp_real = kk_algorithms.KK_General_PP(
        eval_energies=asp_db_PS_extended.energies,
        energies=asp_db_PS_extended.energies,
        imaginary_spectrum=asp_db_PS_extended.coefs,
        orders=np.array([1,0,-1,-2,-3]),
        relativistic_correction=ps_stoich.relativistic_correction
    )
    ax2.plot(asp_db_PS_extended.energies, gen_pp_real + 0.1, label=f"{PS_NAME} General PP")
    ax2.set_ylim(-10,10)
    from kkcalc_old.kk import KK_General_PP as KK_General_PP_old
    
    gen_pp_real_old = KK_General_PP_old(
        Eval_Energy=asp_db_PS_extended.energies,
        Energy=asp_db_PS_extended.energies,
        imaginary_spectrum=asp_db_PS_extended.coefs,
        orders=np.array([1,0,-1,-2,-3]),
        relativistic_correction=ps_stoich.relativistic_correction
    )
    ax2.plot(asp_db_PS_extended.energies, gen_pp_real_old + 0.2, label=f"{PS_NAME} General PP old")
    
    pp_real = kk_algorithms.KK_PP(
        target_energies=asp_db_PS_extended.energies,
        energies=asp_db_PS_extended.energies,
        imag_coefs=asp_db_PS_extended.coefs,
        relativistic_correction=ps_stoich.relativistic_correction
    )
    ax2.plot(asp_db_PS_extended.energies, pp_real, label=f"{PS_NAME} PP")
    
    pp_real = kk_algorithms.KK_PP(
        target_energies=asp_db_PS_extended_fixed.energies,
        energies=asp_db_PS_extended_fixed.energies,
        imag_coefs=asp_db_PS_extended_fixed.coefs,
        relativistic_correction=ps_stoich.relativistic_correction
    )
    ax2.plot(asp_db_PS_extended.energies, pp_real, label=f"{PS_NAME} PP Fixed")
    
    ax2.set_title(PS_NAME)
    ax2.legend()
    
    plt.show()
    