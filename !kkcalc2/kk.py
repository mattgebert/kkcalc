import math
import numpy as np
import numpy.typing as npt
from typing import Any

class kramersKronig:
    
    @staticmethod
    def KK_General_PP(eval_energies: npt.NDArray, 
                      energies: npt.NDArray, 
                      imaginary_spectrum: npt.NDArray, 
                      orders : npt.NDArray, 
                      relativistic_correction : float = 0
                      ) -> npt.NDArray:
        """Calculate Kramers-Kronig transform with "Piecewise Polynomial"
        algorithm plus the Biggs and Lighthill extended data.

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
        
        # Need to build x-E-n arrays
        X = np.tile(energies[:,np.newaxis,np.newaxis],(1,len(eval_energies),len(orders))) # 1D to 3D
        E = np.tile(eval_energies[np.newaxis,:,np.newaxis],(len(energies)-1,1,len(orders))) #1d TO 3d
        C = np.tile(imaginary_spectrum[:,np.newaxis,:],(1,len(eval_energies),1)) # 2D to 3D
        N = np.tile(orders[np.newaxis,np.newaxis,:],(len(energies)-1,len(eval_energies),1))
        
        # Calculate when evaluating energies are equal to the data energies, in a boolean array (1 or 0)
        poles = np.tile(np.equal(eval_energies[:,np.newaxis],energies[np.newaxis,:]),(1,1,len(orders)))
        # poles = np.equal(X,np.tile(eval_energies[np.newaxis,:,np.newaxis],(len(energies),1,len(orders))))
        
        # Calculate 
        
        ### Calculate the Kramers-Kronig integral
        
        
        # all N, ln(x+E) and ln(x-E) terms and pole
        Integral = np.sum(
            -C*(-E)**N * np.log(np.abs((X[1:,:,:]+E)/(X[:-1,:,:]+E)))
            -C*E**N*(1-poles[1:,:,:])*np.log(np.abs(
                (
                    X[1:,:,:]-E+poles[1:,:,:]
                )/(
                    (1-poles[:-1,:,:])*X[:-1,:,:]+poles[:-1,:,:]*X[[0]+list(range(len(energies)-2)),:,:]-E
                )
            ))
        ,axis=(0,2))
        
        if np.any(orders<=-2): # N<=-2, ln(x) terms
            i = [slice(None,None,None),slice(None,None,None),orders<=-2]
            Integral += np.sum(C[i]*((-E[i])**N[i]+E[i]**N[i])*np.log(np.absolute((X[1:,:,orders<=-2])/(X[:-1,:,orders<=-2]))),axis=(0,2))
        
        if np.any(orders>=0): # N>=0,  x^k terms
            for ni in np.where(orders>=0)[0]:
                i = [slice(None,None,None),slice(None,None,None),ni]
                n = orders[ni]
                for k in range(n,0,-2):
                    Integral += np.sum(C[i]/float(-k)*2*E[i]**(n-k)*(X[1:,:,ni]**k-X[:-1,:,ni]**k),axis=0)
        
        if np.any(orders <=-3): # N<=-3, x^k terms
            for ni in np.where(orders<=-3)[0]:
                i = [slice(None,None,None),slice(None,None,None),ni]
                n = orders[ni]
                for k in range(n+2,0,2):
                    Integral += np.sum(C[i]/float(k)*((-1)**(n-k)+1)*E[i]**(n-k)*(X[1:,:,ni]**k-X[:-1,:,ni]**k),axis=0)
        
        return Integral / math.pi + relativistic_correction