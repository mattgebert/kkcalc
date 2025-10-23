"""
The loading module for the atomic scattering factor database.

Loads the database from a json file created by the db_generator_script.py script.
By default, the database file is expected to be in the same directory as this module.
Can load a compressed gzip version of the database for smaller distribution size.
"""

import os, json
import numpy as np
import numpy.typing as npt
from typing import Literal, Union, TypedDict
import gzip


class ASFElement(TypedDict):
    """
    Atomic Scattering Factor data for a single element.
    """

    E: npt.NDArray
    Im: npt.NDArray
    Re: npt.NDArray
    name: str
    symbol: str
    mass: float


def load_asf_database() -> dict[int, ASFElement]:
    """
    Load the atomic scattering factor database from a json file.

    The database has been previously created by the db_generator_script.py script.

    Returns
    -------
    dict[int, ASFElement]
            A dictionary of elements atomic number keys, with each value
            consisting of a dictionary of values (see `ASFElement`).
    """
    path_json = os.path.join(os.path.dirname(__file__), "ASF.json")
    path_gzip_json = os.path.join(os.path.dirname(__file__), "ASF.json.gz")

    # Load all information. This inclues E, Im, and Re but also name and atomic masses.
    if os.path.exists(path_gzip_json):
        with gzip.open(path_gzip_json, "rt") as f:
            json_database = json.load(f)
    elif os.path.exists(path_json):
        with open(path_json, "r") as f:
            json_database = json.load(f)
    else:
        raise FileNotFoundError("ASF database file not found.")

    # Convert lists to numpy arrays and convert dictionary keys to integers
    asf_database = {}
    for Z in json_database.keys():
        try:
            intZ = int(Z)
            # Use the same values but with integer keys
            asf_database[intZ] = json_database[Z]
            asf_database[intZ]["E"] = np.array(json_database[Z]["E"])
            asf_database[intZ]["Im"] = np.array(json_database[Z]["Im"])
            asf_database[intZ]["Re"] = np.array(json_database[Z]["Re"])
        except ValueError:
            continue

    return asf_database
