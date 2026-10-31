"""
Session save/load and CSV export utilities for the KKCalc2 GUI.

A "session" is the full set of atomic scattering factor/polynomial objects currently loaded
into the GUI's object list. Sessions are serialized to a JSON file, recording each object's
class, material properties (name, stoichiometry, density, etc.), and underlying numeric data
(energies and factors/coefficients), so that a session can be closed and reopened later.

Only the "concrete" representation classes are supported: `asf_re`, `asf_im`, `asf_complex`,
`asp_re`, `asp_im`, `asp_complex`. Database-extended objects (e.g. `asp_db_im_extended`) are
subclasses of these and are saved/restored as their flattened (non-extended) base representation
-- the resulting numeric data is preserved and fully usable, but extension-specific metadata
(e.g. the merge domain or original database reference) is not.
"""

import csv
import json
import os
from typing import Any

import numpy as np
import numpy.typing as npt

from kkcalc2.models import (
    asf_abstract,
    asf_complex,
    asf_im,
    asf_re,
    asp_abstract,
    asp_complex,
    asp_im,
    asp_re,
)

#: The current session file format version.
SESSION_FORMAT_VERSION = 1


def _serialize_properties(obj: asf_abstract | asp_abstract) -> dict[str, Any]:
    """
    Collect an object's material properties into a JSON-serializable dictionary.

    Parameters
    ----------
    obj : asf_abstract | asp_abstract
        The object to collect properties from.

    Returns
    -------
    dict[str, Any]
        A dictionary of `name`/`stoichiometry`/`density`/`number_density`/`formula_mass`/
        `is_extended` values, with `stoichiometry` converted to its string representation.
    """
    props = obj._properties_dict
    stoich = props["stoichiometry"]
    return {
        "name": props["name"],
        "stoichiometry": str(stoich) if stoich is not None else None,
        "density": props["density"],
        "number_density": props["number_density"],
        "formula_mass": props["formula_mass"],
        "is_extended": props["is_extended"],
    }


def _serialize_object(obj: asf_abstract | asp_abstract) -> dict[str, Any]:
    """
    Serialize a single `asf`/`asp` object to a JSON-serializable dictionary.

    Parameters
    ----------
    obj : asf_abstract | asp_abstract
        The object to serialize. Must be an instance of `asf_re`, `asf_im`, `asf_complex`,
        `asp_re`, `asp_im`, or `asp_complex` (or a subclass thereof).

    Returns
    -------
    dict[str, Any]
        A JSON-serializable representation of the object.

    Raises
    ------
    TypeError
        If `obj` is not an instance of a supported class.

    See Also
    --------
    _deserialize_object : The inverse operation.
    """
    if not isinstance(obj, (asf_abstract, asp_abstract)):
        raise TypeError(
            f"Cannot save object of type {type(obj).__name__!r} to a session file. "
            "Only asf_re/asf_im/asf_complex/asp_re/asp_im/asp_complex (or subclasses) are supported."
        )
    properties = _serialize_properties(obj)

    if isinstance(obj, asf_complex):
        return {
            "class": "asf_complex",
            "properties": properties,
            "energies": obj.energies.tolist(),
            "factors_real": obj.factors.real.tolist(),
            "factors_imag": obj.factors.imag.tolist(),
        }
    if isinstance(obj, asp_complex):
        return {
            "class": "asp_complex",
            "properties": properties,
            "energies": obj.energies.tolist(),
            "coefs_real": obj.coefs.real.tolist(),
            "coefs_imag": obj.coefs.imag.tolist(),
            "orders": obj.orders.tolist() if obj.orders is not None else None,
        }
    if isinstance(obj, (asf_re, asf_im)):
        return {
            "class": "asf_re" if isinstance(obj, asf_re) else "asf_im",
            "properties": properties,
            "energies": obj.energies.tolist(),
            "factors": obj.factors.tolist(),
        }
    if isinstance(obj, (asp_re, asp_im)):
        return {
            "class": "asp_re" if isinstance(obj, asp_re) else "asp_im",
            "properties": properties,
            "energies": obj.energies.tolist(),
            "coefs": obj.coefs.tolist(),
            "orders": obj.orders.tolist() if obj.orders is not None else None,
        }
    raise TypeError(
        f"Cannot save object of type {type(obj).__name__!r} to a session file. "
        "Only asf_re/asf_im/asf_complex/asp_re/asp_im/asp_complex (or subclasses) are supported."
    )


def _deserialize_object(data: dict[str, Any]) -> asf_abstract | asp_abstract:
    """
    Reconstruct a single `asf`/`asp` object from its serialized dictionary representation.

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary previously produced by `_serialize_object`.

    Returns
    -------
    asf_abstract | asp_abstract
        The reconstructed object.

    Raises
    ------
    ValueError
        If `data["class"]` is not a recognised class name.

    See Also
    --------
    _serialize_object : The inverse operation.
    """
    cls_name = data["class"]
    props: dict[str, Any] = dict(data["properties"])
    energies = np.asarray(data["energies"], dtype=float)

    if cls_name == "asf_complex":
        factors = np.asarray(data["factors_real"], dtype=float) + 1j * np.asarray(
            data["factors_imag"], dtype=float
        )
        return asf_complex.from_asf(energies, factors, **props)
    if cls_name == "asp_complex":
        orders = _load_orders(data)
        coefs_real = np.asarray(data["coefs_real"], dtype=float)
        coefs_imag = np.asarray(data["coefs_imag"], dtype=float)
        re_obj = asp_re(energies=energies, coefs=coefs_real, orders=orders, **props)
        im_obj = asp_im(energies=energies, coefs=coefs_imag, orders=orders, **props)
        return asp_complex(re=re_obj, im=im_obj, **props)
    if cls_name == "asf_re":
        factors = np.asarray(data["factors"], dtype=float)
        return asf_re(energies=energies, factors=factors, **props)
    if cls_name == "asf_im":
        factors = np.asarray(data["factors"], dtype=float)
        return asf_im(energies=energies, factors=factors, **props)
    if cls_name == "asp_re":
        orders = _load_orders(data)
        coefs = np.asarray(data["coefs"], dtype=float)
        return asp_re(energies=energies, coefs=coefs, orders=orders, **props)
    if cls_name == "asp_im":
        orders = _load_orders(data)
        coefs = np.asarray(data["coefs"], dtype=float)
        return asp_im(energies=energies, coefs=coefs, orders=orders, **props)
    raise ValueError(f"Unknown class {cls_name!r} in session file.")


def _load_orders(data: dict[str, Any]) -> npt.NDArray[np.integer] | None:
    """
    Load the (optional) `orders` array from a serialized `asp`-like dictionary.

    Parameters
    ----------
    data : dict[str, Any]
        A dictionary previously produced by `_serialize_object` for an `asp`-like object.

    Returns
    -------
    npt.NDArray[np.integer] | None
        The loaded `orders` array, or `None` if it is not present.
    """
    orders = data.get("orders")
    return np.asarray(orders, dtype=int) if orders is not None else None


def save_session(
    objs: list[asf_abstract | asp_abstract], filepath: str | os.PathLike
) -> None:
    """
    Save a list of loaded objects to a session JSON file.

    Parameters
    ----------
    objs : list[asf_abstract | asp_abstract]
        The objects to save (e.g. `kk_object_list.all_objects` from the GUI).
    filepath : str | os.PathLike
        Path to the session file to write.

    Raises
    ------
    TypeError
        If any object in `objs` is not a supported class (see `_serialize_object`).

    See Also
    --------
    load_session : Load a previously saved session file.
    """
    payload = {
        "version": SESSION_FORMAT_VERSION,
        "objects": [_serialize_object(obj) for obj in objs],
    }
    # Write to a temp file first so a failed save can't corrupt an existing session.
    tmp_path = f"{os.fspath(filepath)}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, filepath)


def load_session(filepath: str | os.PathLike) -> list[asf_abstract | asp_abstract]:
    """
    Load a list of objects from a previously saved session JSON file.

    Parameters
    ----------
    filepath : str | os.PathLike
        Path to the session file to read.

    Returns
    -------
    list[asf_abstract | asp_abstract]
        The reconstructed objects, in the order they were saved.

    Raises
    ------
    ValueError
        If the file's format version is not supported, or an object has an unrecognised class.

    See Also
    --------
    save_session : Save a list of objects to a session file.
    """
    with open(filepath, "r") as f:
        payload = json.load(f)

    version = payload.get("version")
    if version != SESSION_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported session file version: {version!r}. "
            f"Expected {SESSION_FORMAT_VERSION!r}."
        )
    return [_deserialize_object(entry) for entry in payload["objects"]]


def export_objects_to_csv(
    objs: list[asf_abstract | asp_abstract], directory: str | os.PathLike
) -> list[str]:
    """
    Export each object's energy/factor data to its own CSV file within `directory`.

    Real-valued objects (`asf_re`/`asf_im`/`asp_re`/`asp_im`) are written with columns
    `Energy (eV)`/`Factor`. Complex-valued objects (`asf_complex`/`asp_complex`) are written
    with columns `Energy (eV)`/`Factor (Real)`/`Factor (Imag)`.

    Parameters
    ----------
    objs : list[asf_abstract | asp_abstract]
        The objects to export.
    directory : str | os.PathLike
        The directory to write the CSV files into (created if it doesn't already exist).
        One file is written per object, named `<object name>.csv` (sanitized and
        de-duplicated if names clash).

    Returns
    -------
    list[str]
        The paths of the CSV files written, in the same order as `objs`.
    """
    os.makedirs(directory, exist_ok=True)
    written_paths: list[str] = []
    used_names: set[str] = set()

    for obj in objs:
        safe_name = (
            "".join(
                c if (c.isalnum() or c in " _-.") else "_"
                for c in (obj.name or "object")
            ).strip()
            or "object"
        )
        candidate = safe_name
        i = 1
        while candidate in used_names:
            candidate = f"{safe_name}_{i}"
            i += 1
        used_names.add(candidate)

        path = os.path.join(str(directory), f"{candidate}.csv")
        energies = obj.energies
        factors = obj.asf if isinstance(obj, asp_abstract) else obj.factors

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            if np.iscomplexobj(factors):
                writer.writerow(["Energy (eV)", "Factor (Real)", "Factor (Imag)"])
                for e, val in zip(energies, factors):
                    writer.writerow([e, val.real, val.imag])
            else:
                writer.writerow(["Energy (eV)", "Factor"])
                for e, val in zip(energies, factors):
                    writer.writerow([e, val])
        written_paths.append(path)

    return written_paths
