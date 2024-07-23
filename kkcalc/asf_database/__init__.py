"""
This module is the atomic scattering factor module for KK calc. 

It's primarily used to 

Data is sourced from 
- Briggs and Lighthill, 1976, J. Phys. Chem. Ref. Data, 5, 581-837,
- Henke et al., 1993, At. Data Nucl. Data Tables, 54, 181-342.

The atomic scattering factors found in `ASF.json` are calculated using `asf_generator_script.py`.
These scattering factors can be accessed via the `ASF_DATABASE` variable in this module, 
or can be loaded using the `load_asf_database` function.

Each element in the database is a dictionary consisting of the following keys:
- 'E': A numpy array of `N+1` photon energies corresponding to intervals of the scattering factor data.
- 'Re': A numpy array of `N-3` real coefficient of the scattering factor. TODO: Why is this N-3?
- 'Im': A 2D numpy array of dimensions `N, 5`, with values of `5` piecewise polynomial coefficients
        for the imaginary part of the scattering factors, corresponding to the energies intervals.
"""

from kkcalc.asf_db.asf_loader import load_asf_database
# from kkcalc.asf_db.asf_spectra import asp_db
ASF_DATABASE = load_asf_database() #spectral data, plus atomic masses

# Example usage
if __name__ == '__main__':
    for z, data in enumerate(ASF_DATABASE):
        print(data["name"], data["E"].shape, data["Re"].shape, data["Im"].shape)