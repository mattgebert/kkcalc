import os, json
import numpy as np
import numpy.typing as npt
from typing import Literal, Union, TypedDict

class ASFElement(TypedDict):
	E: npt.NDArray
	Im: npt.NDArray
	Re: np.array
	name: str
	symbol: str
	mass: float

# dict[Literal["E", "Im", "Re"], np.ndarray]
# | dict[Literal["name", "symbol"], str]
# | dict[Literal["mass"], float]
def load_asf_database() -> dict[int, ASFElement]:
	"""
	Loads atomic scattering factor database from a json file.	
	The database has been previously created by PackRawData.py
	
	Parameters
	----------
	None
	
	Returns
	-------
	The function returns a dictionary of elements, each consisting of a dictionary
	of data types
	"""
	asf_database = {}
	with open(os.path.join(os.path.dirname(__file__),'ASF.json'),'r') as f:
		# Load all information. This inclues E, Im, and Re but also name and atomic masses.
		json_database = json.load(f)
		# Convert lists to numpy arrays and convert dictionary keys to integers
		for Z in json_database.keys():
			try:
				intZ = int(Z)
				# Use the same values but with integer keys
				asf_database[intZ] = json_database[Z]
				asf_database[intZ]['E'] = np.array(json_database[Z]['E'])
				asf_database[intZ]['Im'] = np.array(json_database[Z]['Im'])
				asf_database[intZ]['Re'] = np.array(json_database[Z]['Re'])
			except ValueError:
				continue
	return asf_database