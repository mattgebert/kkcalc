import numpy as np
from kkcalc.stoich import stoichiometry
from kkcalc.models.poly import asp
from kkcalc.models.factors import asf

# Load the real/imag scattering factors as they vary with energy
from kkcalc.asf_db import ASF_DATABASE

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
    asf : The atomic scattering factor module for KK calc, where data is sourced from Briggs and Lighthill, and Henke et al.
    """
    def __init__(self, stoich: stoichiometry):
        
        # Get stoichiometry
        comp = stoich.composition
        
        # Get unique energy points for all elements
        energies = np.unique(np.r_[*[ASF_DATABASE[z]['E'] for z,_ in comp]])
        
        # Add weighted asf data sets for KK calculation
        im_coefs = np.zeros((len(energies)-1, 5)) # Stores summations of imaginary coefficients at each energy
        # Iterate over the unique energies
        for i, energy in enumerate(energies[1:]): # iterate over the energies of the ASF_DATABASE
            sum_im = 0 # Sum of the imaginary coefficients at the current energy
            index = 0 # Index of the current energy in the ASF_DATABASE
            # Sum the imaginary coefficients at each energy
            for (z, n) in comp:
                # Imaginary coefs at current energy
                db_im_coefs = ASF_DATABASE[z]['Im'][index,:] # the imaginary piecewise polynomial coefficients
                sum_im += n * db_im_coefs  # Multiply by stoichiometry n
                
                # Check if the next energy matches the currently used elemental energy, i.e. end of the valid interval.
                if ASF_DATABASE[z]['E'][index+1] == energy:
                    index += 1 # Increment counter[j] by 1 if the energy matches, to move to the next energy window
            # Store the sum of the imaginary coefficients at the current energy
            im_coefs[i,:] = sum_im
        
        # Setup properties
        super().__init__(energies, im_coefs)
        
class asp_db_extended(asf):
    """
    Class for extending an `asf` object with database scattering factor data.
    
    Merges scattering factor polynomials with the user-provided near-edge data.

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
        ### 1. Alignment of Energy Values:
        
        # Get the data pointers from the asf object
        data_e, data_y = data_asf.data
        # Get the data pointers from the asp object
        asp_e, asp_y = db_asp.energies, db_asp.coefs
        
        # Check if merge points are defined:
        if merge_domain is None:
            merge_domain = data_e[[0, -1]] # full range of the data_asf energies
            data_merge_lb_idx = 0
            data_merge_ub_idx = -1
            data_merge_lb = merge_domain[0]
            data_merge_ub = merge_domain[1]
        else:
            # Find the indices and values of the data_asf energies that are within the range of the db_asp energies
            data_merge_lb_idx = np.argmax(data_e > merge_domain[0])
            data_merge_ub_idx = np.argmax(data_e < merge_domain[1])
            data_merge_lb = data_e[data_merge_lb_idx]
            data_merge_ub = data_e[data_merge_ub_idx]
            
        # Use linear interpolation to find corresponding values of the merge domain.
        data_merge_range = np.interp(merge_domain, data_e, data_y)
        
        # Find the indices of the spans where the db_asp energies are within the range of the data_asf energies.
        db_asp_merge_lb_idx = np.where(asp_e < merge_domain[0]) # Find values before merge/data edge 
        db_asp_merge_ub_idx = np.where(asp_e > merge_domain[1]) # Find values after merge/data edge
        # Find the nearest values to the edge
        db_asp_merge_lb = asp_e[db_asp_merge_lb_idx][-1] # Use the last value before merge/data edge
        db_asp_merge_ub = asp_e[db_asp_merge_ub_idx][0] # Use first value after merge/data edge
        # Calculate the corresponding y values using the polynomial coefs
        db_asp_merge_range = db_asp.to_asf(energies=merge_domain)
        
        ### Calculate the scale difference between the data_asf and db_asp 
        # Range(db_asp) / Range(data_asf)
        scale = (db_asp_merge_range[1] - db_asp_merge_range[0]) / (data_merge_range[1] - data_merge_range[0]) 
        scaled_data_y = (data_y-data_merge_range[1]) * scale + db_asp_merge_range[0]
        
        if fix_distortions:
            # TODO: Implement later
            pass
        
        # Merge the data
        merge_data_e = np.r_[merge_domain[0], 
                             data_e[data_merge_lb_idx:data_merge_ub_idx], 
                             merge_domain[1]]
        merge_data_y = np.r_[db_asp_merge_range[0], 
                             scaled_data_y[data_merge_lb_idx:data_merge_ub_idx], 
                             db_asp_merge_range[1]]
        
        # Convert factors to coefficients
        
        
        
        pass
        return
    
        numpy = np
        # logger.info("Merge near-edge data with wide-range scattering factor data")
        if merge_points is None:
            merge_points = NearEdge_Data[[0,-1],0]
        NearEdge_merge_ind = [numpy.where(NearEdge_Data[:,0] > merge_points[0])[0][0], numpy.where(NearEdge_Data[:,0] < merge_points[1])[0][-1]]
        NearEdge_merge_values = numpy.interp(merge_points, NearEdge_Data[:,0], NearEdge_Data[:,1])
        if ASF_Data is not None:
            asf_merge_ind = [numpy.where(ASF_E > merge_points[0])[0][0]-1, numpy.where(ASF_E > merge_points[1])[0][0]-1]
            asf_merge_values = [coeffs_to_ASF(merge_points[0], ASF_Data[asf_merge_ind[0],:]), coeffs_to_ASF(merge_points[1], ASF_Data[asf_merge_ind[1],:])]
            if add_background:
                logger.info("Add background")
                logger.error("Not implemented!")
                #get pre-edge region
                #extrapolate background
                scale = (asf_merge_values[1]-asf_merge_values[0])/(NearEdge_merge_values[1]-NearEdge_merge_values[0])
                scaled_NearEdge_Data = numpy.vstack((NearEdge_Data[:,0],((NearEdge_Data[:, 1]-NearEdge_merge_values[0])*scale)+asf_merge_values[0])).T
            else:# don't add background
                scale = (asf_merge_values[1]-asf_merge_values[0])/(NearEdge_merge_values[1]-NearEdge_merge_values[0])
                scaled_NearEdge_Data = numpy.vstack((NearEdge_Data[:,0],((NearEdge_Data[:, 1]-NearEdge_merge_values[0])*scale)+asf_merge_values[0])).T
            try:
                import scipy.optimize
                SCIPY_FLAG = True
            except ImportError:
                SCIPY_FLAG = False
                logger.info('Failed to import the scipy.optimize module - disabling the \'fix distortions\' option.')
            if SCIPY_FLAG and fix_distortions:
                logger.info("Fix distortions")
                import scipy.optimize
                ASF_fitY = 0.0*NearEdge_Data[:, 0]
                for i, E in enumerate(NearEdge_Data[:,0]):
                    ASF_fitY[i] = coeffs_to_ASF(E, ASF_Data[numpy.where(ASF_E > E)[0][0]-1])
                fitfunc = lambda p, x, y, asf_mv, asf: ((y-p*x)-(y[0]-p*x[0]))/((y[-1]-p*x[-1])-(y[0]-p*x[0]))*(asf_mv[1]-asf_mv[0])+asf_mv[0] - asf
                p0 = -(NearEdge_merge_values[1]-NearEdge_merge_values[0])/((asf_merge_values[1]-asf_merge_values[0])*NearEdge_Data[0,0])
                logger.debug("Fix distortions - start fit with p0 ="+str(p0))
                p1, success = scipy.optimize.leastsq(fitfunc, p0, args=(NearEdge_Data[:, 0], NearEdge_Data[:, 1], asf_merge_values, ASF_fitY))
                logger.debug("Fix distortions - complete fit with p1 ="+str(p1[0]))
                NearEdge_fitY = asf_merge_values[0]+((NearEdge_Data[:,1]-p1[0]*NearEdge_Data[:,0])-(NearEdge_Data[0,1]-p1[0]*NearEdge_Data[0,0]))*(asf_merge_values[1]-asf_merge_values[0])/((NearEdge_Data[-1,1]-p1[0]*NearEdge_Data[-1,0])-(NearEdge_Data[0,1]-p1[0]*NearEdge_Data[0,0]))
                scaled_NearEdge_Data = numpy.vstack((NearEdge_Data[:,0],NearEdge_fitY)).T
        else:
            scaled_NearEdge_Data = NearEdge_Data.copy()
            asf_merge_values = NearEdge_merge_values.copy()
        plot_scaled_NearEdge_Data = scaled_NearEdge_Data.copy()
        scaled_NearEdge_Data = numpy.vstack(((merge_points[0],asf_merge_values[0]),scaled_NearEdge_Data[NearEdge_merge_ind[0]:NearEdge_merge_ind[1]+1,:],(merge_points[1],asf_merge_values[1])))

        #Now convert point values to coefficients
        NearEdge_Coeffs = numpy.zeros((len(scaled_NearEdge_Data),5))
        M = (scaled_NearEdge_Data[1:,1]-scaled_NearEdge_Data[:-1,1])/(scaled_NearEdge_Data[1:,0]-scaled_NearEdge_Data[:-1,0])
        NearEdge_Coeffs[:-1,0] = M
        NearEdge_Coeffs[:-1,1] = scaled_NearEdge_Data[:-1,1]-M*scaled_NearEdge_Data[:-1,0]
        if ASF_Data is None:
            NearEdge_Coeffs = NearEdge_Coeffs[0:-1,:]
            #NearEdge_Coeffs[-1,:] = NearEdge_Coeffs[-2,:]
            #Paste data sections together
            Full_E = scaled_NearEdge_Data[:,0]
            Full_Coeffs = NearEdge_Coeffs
            splice_points = [0,len(Full_E)-1]
        else:
            NearEdge_Coeffs[-1,:] = ASF_Data[asf_merge_ind[1],:]
            #Paste data sections together
            Full_E = numpy.hstack((ASF_E[0:asf_merge_ind[0]+1],scaled_NearEdge_Data[:,0],ASF_E[asf_merge_ind[1]+1:]))
            Full_Coeffs = numpy.vstack((ASF_Data[0:asf_merge_ind[0]+1,:],NearEdge_Coeffs,ASF_Data[asf_merge_ind[1]+1:,:]))
            splice_points = [asf_merge_ind[0]+1, asf_merge_ind[0]+len(scaled_NearEdge_Data[:,0])]
        if plotting_extras:
            return Full_E, Full_Coeffs, plot_scaled_NearEdge_Data, splice_points
        else:
            return Full_E, Full_Coeffs
        
        
        
        # """Normalise the user-provided, near-edge data to the scattering factor data and combine them.
        
        # Parameters
        # ----------
        # NearEdge_Data : a numpy array with two columns: Photon energy and absorption data values.
        # ASF_E : 1D numpy array listing the starting photon energies for each spectrum segment.
        # ASF_Data: nx5 numpy array in which each row lists the polynomial coefficients describing the shape of the spectrum in that segment.
        # merge_points : a pair of photon energies indicating the data range of the NearEdge_Data to be used.
        # add_background : boolean flag for adding a background function to the provided near-edge data.
        # fix_distortions : boolean flag for removing erroneous slope from the provided near-edge data.
        # plotting_extras : boolean flag for providing plottable feedback data that isn't required for actual calculations.

        # Returns
        # -------
        # Full_E : 1D numpy array listing the updated starting photon energies for each spectrum segment.
        # Full_Coeffs : nx5 numpy array in which each row lists the polynomial coefficients describing the shape of the spectrum in that segment.
        # plot_scaled_NearEdge_Data (optional) : an updated verion of NearEdge_Data that has been scaled to match the scattering factor data.
        # splice_points (optional) : a list of two pairs of photon energy-magnitude values to represent the merge_points in a plot.
        # """
        # logger.info("Merge near-edge data with wide-range scattering factor data")
        
    
if __name__ == "__main__":
    ## Test various formulas
    # Setup graph
    import matplotlib.pyplot as plt
    plots = plt.subplots(1,1)
    fig: plt.Figure = plots[0]
    ax: plt.Axes = plots[1]
    
    P3MEET = "C9H12O6S2" #C9H11O3S
    CARBON = "C1"
    ANTIMONY = "Sb1"
    SULFUR = "S1"
    
    compounds = [P3MEET, CARBON, SULFUR, ANTIMONY]
    
    for compound in compounds:
        stoich = stoichiometry(compound)
        print(stoich)
        asp = stoich.asp_im()
        
        # Convert all energies to asf:
        energies = asp.energies
        asf = asp.to_asf()
        
        # Graph the asf
        ax.scatter(energies, asf, s=1, alpha=0.5, label=f"{compound} ASF")
        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("ASF Data")
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        # Plot the polynomials
        for i, e1 in enumerate(energies[:-1]):
            e2 = energies[i+1]
            x = np.linspace(e1, e2, 100)
            asf = asp.to_asf(energies=x)
            ax.plot(x, asf, linewidth=0.5, c="teal", label=f"{compound} Polynomial" if i == 0 else None)
            if i==0:
                print("ASFE", e1)
                print("ASF", asf[0])
                print("ASF2", asp.to_asf(energies=e1))
    
    ax.set_title("Atomic Scattering Factors of Elements and Compounds")
    ax.legend()
    plt.show()