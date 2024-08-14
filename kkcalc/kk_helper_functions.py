import numpy as np, numpy.typing as npt
import warnings

from kkcalc import kk_transforms
from kkcalc.stoich import kk_stoichiometry
from kkcalc.stoich import kk_stoichiometry
from kkcalc.asf_database.db_models import asp_db, asp_db_extended
from kkcalc.models.factors import KK_Datatype, asf, asf_im

def calc_real(
    energies: npt.NDArray,
    intensities: npt.NDArray,
    formula : kk_stoichiometry | str,
    input_data_type : KK_Datatype = None,
    merge_domain:tuple[float, float]=None,
    fix_distortions:bool=False,
    tolerance=None,
    max_iter=50) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Legacy function to calculate the real part of the scattering factors from the imaginary part.
    
    Calculates the real part through the Kramers-Kronig transform, throught the following procedure:
    1. Converts the input data to an atomic scattering factor `asf_im` object.
    2. Merges the data with the polynomial database, into an `asp_db_extended` object, 
        using the merge_domain to truncate the data.
    3. Calculates the real part of the scattering factors using the `KK_PP` method.
    4. Using the `improve_accuracy` method, adding extra data points to increase resolution.
    5. Returns the resulting energies, real and imaginary parts of the scattering factors.
    
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
        
    Returns
    -------
    energies : npt.NDArray
        The photon energies, with shape (N,)
    real : npt.NDArray
        The real part of the scattering factors, with shape (N,)
    imag : npt.NDArray
        The imaginary part of the scattering factors, with shape (N,)    
    """
    
    # Verify shapes
    if energies.shape != intensities.shape:
        raise ValueError(f"Shape mismatch between energies ({energies.shape}) and intensities ({intensities.shape})")
    
    # Use the stoichiometry to get the relativistic correction and database atomic scattering polynomial
    stoich = formula if isinstance(formula, kk_stoichiometry) else kk_stoichiometry.from_chemical_formula(formula)
    rc = stoich.relativistic_correction
    db_poly: asp_db = stoich.asp_im() #database
    
    # Load the NEXAFS data.
    match input_data_type:
        case KK_Datatype.UNDEFINED:
            warnings.warn("No data type provided. Defaulting to NEXAFS.")
            data_asf = asf_im.from_NEXAFS(energies, intensities)
        case KK_Datatype.NEXAFS | KK_Datatype.XANES | KK_Datatype.PHOTOABSORPTION:
            data_asf = asf_im.from_NEXAFS(energies, intensities)
        case KK_Datatype.ASF:
            data_asf = asf_im(energies, intensities)
        case KK_Datatype.BETA:
            data_asf = asf_im.from_betas(energies, intensities)
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
    real = kk_transforms.KK_PP(
        target_energies=merge_data_asp.energies,
        energies=merge_data_asp.energies,
        imag_coefs=merge_data_asp.coefs,
        relativistic_correction=rc
    )
    
    if tolerance is not None:
        energies, real, imag = kk_transforms.improve_accuracy(
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
    ps_stoich = kk_stoichiometry(POLYSTYRENE)
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

    # Combine the data with the database into polynomials
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

    # Convert to scattering factors
    extended_asf = asp_db_PS_extended.to_atomic_scattering_factors()
    extended_asf_fixed = asp_db_PS_extended_fixed.to_atomic_scattering_factors()
    db_asf = asp_db_PS.to_asf()
    
    # Plot the imaginary parts
    ax.plot(extended_asf.energies, extended_asf.factors, label=f"{PS_NAME} Extended ASF")
    ax.plot(extended_asf_fixed.energies, extended_asf_fixed.factors, label=f"{PS_NAME} Extended ASF Fixed")
    ax.plot(asp_db_PS.energies, db_asf.factors, label=f"{PS_NAME} DB ASF")
    ax.set_xlim(270, 330)
    ax2.set_xlim(ax.get_xlim())
    # ax2.set_xscale("log")
    # ax2.set_ylim(450, 900)

    ax.set_title(PS_NAME)
    ax.legend()

    ## Use KK algorithms
    #1.
    gen_pp_real = kk_transforms.KK_General_PP(
        target_energies=asp_db_PS_extended.energies,
        energies=asp_db_PS_extended.energies,
        imag_coefs=asp_db_PS_extended.coefs,
        orders=np.array([1,0,-1,-2,-3]),
        relativistic_correction=ps_stoich.relativistic_correction
    )
    #2. 
    asf_real = asp_db_PS_extended.kk_transform(
        stoichiometry=ps_stoich,
    )
    
    # Plot KK Results
    ax2.plot(asp_db_PS_extended.energies, gen_pp_real + 0.1, label=f"{PS_NAME} General PP")
    ax2.plot(asf_real.energies, asf_real.factors + 0.2, label=f"{PS_NAME} Conversion")
    ax2.set_ylim(-10,10)

    # from kkcalc_old.kk import KK_General_PP as KK_General_PP_old
    # gen_pp_real_old = KK_General_PP_old(
    #     Eval_Energy=asp_db_PS_extended.energies,
    #     Energy=asp_db_PS_extended.energies,
    #     imaginary_spectrum=asp_db_PS_extended.coefs,
    #     orders=np.array([1,0,-1,-2,-3]),
    #     relativistic_correction=ps_stoich.relativistic_correction
    # )
    # ax2.plot(asp_db_PS_extended.energies, gen_pp_real_old + 0.2, label=f"{PS_NAME} General PP old")

    pp_real = kk_transforms.KK_PP(
        target_energies=asp_db_PS_extended.energies,
        energies=asp_db_PS_extended.energies,
        imag_coefs=asp_db_PS_extended.coefs,
        relativistic_correction=ps_stoich.relativistic_correction
    )
    ax2.plot(asp_db_PS_extended.energies, pp_real, label=f"{PS_NAME} PP")

    pp_real = kk_transforms.KK_PP(
        target_energies=asp_db_PS_extended_fixed.energies,
        energies=asp_db_PS_extended_fixed.energies,
        imag_coefs=asp_db_PS_extended_fixed.coefs,
        relativistic_correction=ps_stoich.relativistic_correction
    )
    ax2.plot(asp_db_PS_extended.energies, pp_real, label=f"{PS_NAME} PP Fixed")

    ax2.set_title(PS_NAME)
    ax2.legend()

    plt.show()
