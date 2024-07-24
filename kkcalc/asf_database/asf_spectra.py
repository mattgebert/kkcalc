import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
import warnings

from kkcalc.stoich import stoichiometry
from kkcalc.models.poly import asp
from kkcalc.models.factors import asf
from kkcalc.models.conversions import conversions

# Load the real/imag scattering factors as they vary with energy
from kkcalc.asf_database import ASF_DATABASE

class asp_db(asp):
    """
    Uses stochiometry to calculate an imaginary-component piecewise polynomial representation from Henke, Briggs and Lighthill data.
    
    Summation of scattering factor data given the chemical stoichiometry.
    
    Parameters
    ----------
    stoich : stoichiometry
        The stoichiometry of the compound, i.e. the elemental composition.
    
    Attributes
    -------
    energies : numpy.ndarray
        An N+1 length array listing the starting photon energies of the segments that the spectrum is broken up into.
    coefs : numpy.ndarray
        An 2D numpy array with dimensions (N, 5) in which each row lists the polynomial coefficients describing the shape of the imaginary spectrum in that segment.
        
    See Also
    --------
    asf_db : The atomic scattering factor module for KK calc, where data is sourced from Briggs and Lighthill, and Henke et al.
    """
    def __init__(self, stoich: stoichiometry):
        # Store the stoichiometry
        self.stoichiometry: stoichiometry = stoich
        """The reference stoichiometry used to compose the summed atomic scattering factors."""
        
        # Get composition
        comp = stoich.composition
        
        # Get unique energy points for all elements
        energies = np.unique(np.r_[*[ASF_DATABASE[z]['E'] for z,_ in comp]])
        
        # Add weighted asf data sets for KK calculation
        im_coefs = np.zeros((len(energies)-1, 5)) # Stores summations of imaginary coefficients at each energy
        
        # Stores the current energy index for each element, defining coefficients at intermediate energies.
        counters = np.zeros(len(comp), dtype=int) 
        # Iterate over the unique energies
        for i, energy in enumerate(energies[1:]): # iterate over the energies of the ASF_DATABASE
            sum_im = 0 # Sum of the imaginary coefficients at the current energy
            # Sum the imaginary coefficients at each energy
            for j, (z, n) in enumerate(comp):
                # Imaginary coefs at current energy
                db_im_coefs = ASF_DATABASE[z]['Im'][counters[j]] # the imaginary piecewise polynomial coefficients
                sum_im += n * db_im_coefs  # Multiply by stoichiometry n
                # Check if the next energy matches the currently used elemental energy, i.e. end of the valid interval.
                if ASF_DATABASE[z]['E'][counters[j]+1] == energy:
                    counters[j] += 1 # Increment counter[j] by 1 if the energy matches, to move to the next energy window
            # Store the sum of the imaginary coefficients at the current energy
            im_coefs[i,:] = sum_im
                        
        # Setup properties
        super().__init__(energies, im_coefs)
        
class asp_db_extended(asp):
    """
    Class for extending an `asp` object with database scattering factor data.
    
    Merges scattering factor polynomials with the user-provided near-edge data.
    
    Attributes
    ----------
    dataset_asf : asf
        The original `asf` (atomic scattering factor) object containing the user data,
        used to generate the extended `asp` object.
    dataset_asp : asp_db
        The original `asp_db` (atomic scattering polynomial) object containing the database data,
        used to extend the data contained in the `asf` object.

    Parameters
    ----------
    data_asf : asf
        The atomic scattering factor object.
    db_asp : asp_db
        The atomic scattering potential object, 
        generated for a given material stoichiometry.
    merge_domain : tuple[float, float] | None
        The range of energies to merge the user data_asf with the db_asp data.
    fix_distortions : bool
        Flag to fix distortions in the user data_asf.
    """
    def __init__(self, 
                 data_asf: asf,
                 db_asp: asp_db,
                 merge_domain: tuple[float, float] | None = None,
                 fix_distortions: bool = False,
                 ) -> None:
        # Check sorted
        if not np.all(np.diff(data_asf.energies) > 0):
            raise ValueError("Data energies must be sorted")
        
        ### 1. Alignment of Energy Values:
        # Get the data pointers from the asf object
        data_e, data_y = data_asf.data
        # Get the data pointers from the asp object
        asp_e, asp_coefs = db_asp.energies, db_asp.coefs
        
        merge_e, merge_coefs = self.extend_data_with_db(
            data_e=data_e,
            data_y=data_y,
            db_e=asp_e,
            db_coefs=asp_coefs,
            merge_domain=merge_domain,
            fix_distortions=fix_distortions
        )
        
        # Set the asp object attributes.
        super().__init__(energies = merge_e, coefs = merge_coefs)
        
        # Store the data_asf object for reference
        self.dataset_asf: asf = data_asf
        """
        The original `asf` (atomic scattering factor) object containing the user data,
        used to generate the extended `asp` object.
        """
        
        self.dataset_asp: asp = db_asp
        """
        The original `asp` (atomic scattering polynomial) object containing the database data,
        used to extend the `asf` object.
        """
        return
    
    @staticmethod
    def extend_data_with_db(
        data_e: npt.NDArray, 
        data_y: npt.NDArray,
        db_e: npt.NDArray,
        db_coefs: npt.NDArray,
        merge_domain: tuple[float, float] | None = None,
        fix_distortions: bool = False
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Merge the user data (factors) with the database data (polynomial coefs).

        Parameters
        ----------
        data_e : npt.NDArray
            The energy values of the user data.
        data_y : npt.NDArray
            The atomic scattering factor values of the user data.
        db_e : npt.NDArray
            The energy values of the database data.
        db_coefs : npt.NDArray
            The atomic scattering factor polynomial coefficients of the database data.
        merge_domain : tuple[float, float] | None, optional
            The energy domain within with to merge the data.
            By default None, using full data domain.
        fix_distortions : bool, optional
            Use a fit to correct distortions in the user data, by default False

        Returns
        -------
        npt.NDArray
            The merged energy values.
        npt.NDArray
            The merged atomic scattering factor polynomial coefficients.

        Raises
        ------
        ValueError
            Merge domain must be in increasing order.
        ValueError
            Data within the provided energy domain must contain more than one datapoint.
        """
        
        # Check if merge points are defined:
        if merge_domain is None:
            merge_domain = data_e[[0, -1]] # full range of the data_asf energies
            data_merge_lb_idx = 0
            data_merge_ub_idx = -1
        else:
            if merge_domain[0] >= merge_domain[1]:
                raise ValueError("Merge domain must be in increasing order")
            # Find the indices and values of the data_asf energies that are within the range of the db_asp energies
            data_merge_lb_idx = np.argmax(data_e > merge_domain[0])
            data_merge_ub_idx = np.argmax(data_e > merge_domain[1]) - 1
            if data_merge_lb_idx == data_merge_ub_idx:
                raise ValueError(f"Data within domain {merge_domain} must contain more than one energy")
            
        # Use linear interpolation to find corresponding values of the merge domain.
        data_merge_range = np.interp(merge_domain, data_e, data_y)
        
        # Find the indices of the spans where the db_asp energies are within the range of the data_asf energies.
        first_domain_idx = np.argmax(db_e > merge_domain[0])
        db_asp_merge_lb_idx = first_domain_idx - 1 if first_domain_idx > 0 else 0 # Find value before merge/data edge 
        db_asp_merge_ub_idx = np.argmax(db_e > merge_domain[1]) # Find value after merge/data edge
        
        # Calculate the corresponding y values using the polynomial coefs
        db_asp_merge_range = asp.evaluate_energies_on_coefs(
            target_energies=merge_domain,
            energies=db_e,
            coefs=db_coefs)
        
        ### Calculate the scale difference between the data_asf and db_asp 
        # Range(db_asp) / Range(data_asf)
        scale = (db_asp_merge_range[1] - db_asp_merge_range[0]) / (data_merge_range[1] - data_merge_range[0]) 
        scaled_data_y = (data_y-data_merge_range[0]) * scale + db_asp_merge_range[0]
        
        if fix_distortions:
            # TODO: Fix final value correlation
            warnings.warn("Distortion correction is experimental and may not work as expected.")
            # Perform a fit along the domain
            db_y = asp_db.evaluate_energies_on_coefs(target_energies=data_e[data_merge_lb_idx:data_merge_ub_idx],
                                                     energies=db_e,
                                                     coefs=db_coefs) # Find equivalent values of the db_asp energies to the data energies 
            guess_grad = - (data_merge_range[1] - data_merge_range[0]) / (db_asp_merge_range[1] - db_asp_merge_range[0]) / data_y[-1]
            fit_func = asp_db_extended.grad_min
            fit_x = data_e[data_merge_lb_idx:data_merge_ub_idx] # essential to only use domain data to perform fit.
            fit_y = scaled_data_y[data_merge_lb_idx:data_merge_ub_idx] # essential to only use domain data to perform fit.
            (grad, ), _ = opt.leastsq(
                func=fit_func,
                x0=guess_grad,
                args=(fit_x, fit_y, db_asp_merge_range, db_y)
            )
            
            # Reassign the scaled data
            merge_data_e = fit_x
            merge_data_y = (
                db_asp_merge_range[0]
                + asp_db_extended.grad_min(grad,
                                           fit_x,
                                           fit_y,
                                           db_asp_merge_range, 0)
            )
            print(merge_data_e.shape, merge_data_y.shape)
        else:
            # Construct the merge data to use
            merge_data_e = data_e[data_merge_lb_idx:data_merge_ub_idx]
            merge_data_y = scaled_data_y[data_merge_lb_idx:data_merge_ub_idx]   
        
        # Add merge domain to the merge data if not already present
        if merge_domain[0] != merge_data_e[0]:
            merge_data_e = np.r_[merge_domain[0], merge_data_e]
            merge_data_y = np.r_[db_asp_merge_range[0], merge_data_y]
        if merge_domain[1] != merge_data_e[-1]:
            merge_data_e = np.r_[merge_data_e, merge_domain[1]]
            merge_data_y = np.r_[merge_data_y, db_asp_merge_range[1]]
        
        # Convert factors to coefficients
        merge_data_coefs = conversions.ASF_to_ASP(
            energies=merge_data_e,
            factors=merge_data_y
        )
        # Add the db sections to the merge data
        merge_e = merge_data_e
        merge_coefs = merge_data_coefs
        # Boundary already added, so finish at idx-1.
        if db_asp_merge_lb_idx > 0:
            merge_e = np.r_[db_e[0:db_asp_merge_lb_idx-1], merge_e]
            merge_coefs = np.r_[db_coefs[0:db_asp_merge_lb_idx-1], merge_coefs]
        # Boundary already added, so start at idx+1 for energies, and idx for coefs.
        if db_asp_merge_ub_idx < len(db_e):
            merge_e = np.r_[merge_e, db_e[db_asp_merge_ub_idx+1:]]
            merge_coefs = np.r_[merge_coefs, db_coefs[db_asp_merge_ub_idx:]]
        
        return merge_e, merge_coefs
    
    @staticmethod
    def grad_min(grad, x, y, db_merge_range, db_y):
        r"""
        Minimum function to fit a general gradient of the data to the database.
        
        
        
        Parameters
        ----------
        grad : float
            The gradient to fit.
        x : numpy.ndarray
            The x data values to fit.
        y : numpy.ndarray
            The y data values to fit.
        db_merge_range : tuple[float, float]
            The range of the db_asp energies to fit.
        db_y : numpy.ndarray
            The database atomic scattering factor values (not coefs) to fit.
        """
        data_grad_diff = (y - y[0]) - grad*(x - x[0]) # 0 to some number
        data_grad_diff_total = (y[-1] - y[0]) - grad*(x[-1] - x[0]) # some number
        norm_grad_diff = data_grad_diff / data_grad_diff_total # Evolves from 0 to 1
        db_range = db_merge_range[1] - db_merge_range[0] # Range of the database values
        # Difference between the gradient data scaled to the database range, and the database values.
        return norm_grad_diff * db_range - db_y
    
    
if __name__ == "__main__":
    ## Test various formulas
    # Setup graph
    import matplotlib.pyplot as plt
    plots = plt.subplots(1,2)
    fig: plt.Figure = plots[0]
    ax: plt.Axes = plots[1][0]
    ax2: plt.Axes = plots[1][1]
    
    P3MEET = "C9H12O6S2" #C9H11O3S
    CARBON = "C"
    ANTIMONY = "Sb"
    BISMUTH = "Bi"
    TELLURIUM = "Te"
    SELINIUM = "Se"
    SULFUR = "S"
    
    # compounds = [P3MEET, CARBON, SULFUR, ANTIMONY, BISMUTH, TELLURIUM, SELINIUM]
    # compounds = [P3MEET, CARBON, SULFUR]
    compounds = [ANTIMONY, BISMUTH, TELLURIUM, SELINIUM]
    
    for compound in compounds:
        stoich = stoichiometry(compound)
        stoich_asp = stoich.asp_im()
        
        # Convert all energies to asf:
        energies = stoich_asp.energies
        stoich_asf = stoich_asp.to_asf()
        
        # Graph the asf
        scat = ax.scatter(energies, stoich_asf.factors, s=1, alpha=0.5, label=f"{compound} ASF")
        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("ASF Data")
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        # Plot the polynomials
        for i, e1 in enumerate(energies[:-1]):
            e2 = energies[i+1]
            x = np.linspace(e1, e2, 100)
            x_asf = stoich_asp.evaluate_energies(target_energies=x)
            ax.plot(x, x_asf, linewidth=0.5, c=scat.get_edgecolor(), label=f"'{compound}' Polynomial" if i == 0 else None)
    ax.set_title("Atomic Scattering Factors of Elements and Compounds")
    ax.legend()
    
    
    # Create a merge of physical data and database data
    POLYSTYRENE = "CH"
    PS_NAME = "Polystyrene"
    ps_stoich = stoichiometry(POLYSTYRENE)
    asp_db_PS = asp_db(ps_stoich)
    
    # Import Data
    import os
    data_dir = os.path.join(os.path.dirname(__file__),
                            "../../examples/data")
    data_file = os.path.normpath(os.path.join(data_dir, "PS_004_-dc.txt"))
    data_PS = np.genfromtxt(data_file, skip_header=4)
    assert data_PS.shape[1] == 2, "Data file must have two columns"
    asf_PS = asf(energies = data_PS[:,0], 
                 factors = data_PS[:,1])
    
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
    ax2.plot(extended_asf.energies, extended_asf.factors, label=f"{PS_NAME} Extended ASF")
    ax2.plot(extended_asf_fixed.energies, extended_asf_fixed.factors, label=f"{PS_NAME} Extended ASF Fixed")
    db_asf = asp_db_PS.to_asf()
    ax2.plot(asp_db_PS.energies, db_asf.factors, label=f"{PS_NAME} DB ASF")
    ax2.set_xlim(270, 330)
    # ax2.set_xscale("log")
    # ax2.set_ylim(450, 900)
    
    ax2.set_title(PS_NAME)
    ax2.legend()
    
    plt.show()