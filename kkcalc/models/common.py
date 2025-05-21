"""
Classes for common attributes between atomic scattering factor and polynomial models.
"""

import abc
import warnings
from kkcalc.stoich import stoichiometry as kk_stoichiometry, CompositionAlias
from scipy.constants import N_A
from typing import Literal, Self, TypedDict


class PROPERTIES_DICT(TypedDict, total=False):
    """Properties of the common atomic scattering classes."""

    name: str | None
    stoichiometry: kk_stoichiometry | str | None
    density: float | None
    number_density: float | None
    formula_mass: float | None
    is_extended: bool


class atomic_scattering_abstract(metaclass=abc.ABCMeta):
    """
    Interface for common attributes between atomic scattering factor and polynomial models.

    Attributes
    ----------
    name : str
        Material/sample name.
    stoichiometry : stoichiometry | None
        Stoichiometry of the material.
    density : float | None
        Material density in grams per millilitre (cm^3).
    number_density : float | None
        Material number density in atoms per millilitre (cm^3).
    formula_mass : float | None
        Atomic mass sum of the materials chemical formula (molecular mass).
    """

    @property
    @abc.abstractmethod
    def stoichiometry(self) -> kk_stoichiometry | None:
        """
        The `stoichiometry` of the material associated with the scattering factors.

        Returns `None` if no stoichiometry has been provided.

        Returns
        -------
        stoichiometry | None
            Stoichiometry of the material.
        """
        pass

    @property
    @abc.abstractmethod
    def density(self) -> float | None:
        """
        The material density in grams per millilitre (cm^3).

        Returns
        -------
        float | None
            Material density.
        """
        pass

    @property
    @abc.abstractmethod
    def number_density(self) -> float | None:
        """
        The material number density in atoms per millilitre (cm^3).

        Returns
        -------
        float | None
            Material number density.
        """
        pass

    @property
    @abc.abstractmethod
    def formula_mass(self) -> float | None:
        """
        The atomic mass sum of the materials chemical formula (molecular mass).

        In units of atomic mass units (amu).

        Returns
        -------
        float | None
            Atomic mass sum of the materials chemical formula.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        The name of the material/sample associated with the scattering factors.

        Returns
        -------
        str
            Material/sample name.
        """
        pass

    @property
    @abc.abstractmethod
    def is_extended(self) -> bool:
        """
        If the material has been extended by the KKCalc database.

        Returns
        -------
        bool
            `True` if the material has been extended by the KKCalc database.
        """
        pass

    @property
    def _properties_dict(self) -> PROPERTIES_DICT:
        """
        A dictionary of the material class properties.

        Returns
        -------
        dict
            Dictionary of class properties.
        """
        return {
            "name": self.name,
            "stoichiometry": self.stoichiometry,
            "density": self.density,
            "number_density": self.number_density,
            "formula_mass": self.formula_mass,
            "is_extended": self.is_extended,
        }

    @property
    def can_calc_beta(self) -> bool:
        r"""
        Whether the object can calculate $\delta$/$\beta$ values given enough density information.

        Returns
        -------
        bool
            If the object can calculate $\delta$/$\beta$ values.
        """
        return (
            self.number_density is not None
            # Formula mass property uses stoichiometry if not provided.
            or (self.density is not None and self.formula_mass is not None)
        )

    @abc.abstractmethod
    def copy(self) -> Self:
        """
        A copy of the object instance.

        Returns
        -------
        atomic_scattering_abstract
            Copy instance of the object.
        """
        pass


class atomic_scattering(atomic_scattering_abstract):
    """
    Class for common attributes between atomic scattering factor and polynomial models.

    Creates internal attributes for `number_density`, `density`, `stoichiometry`, and `formula_mass`.

    Parameters
    ----------
    name : str | None, optional
        Material/sample name. By default `None`.
    number_density : float | None, optional
        Material number density in atoms per millilitre (cm^3). By default `None`.
        Equivalent to providing a density, with a stoichiometry or formula mass.
    density : float | None, optional
        Material density in grams per millilitre (cm^3).
    stoichiometry : stoichiometry | CompositionAlias | str | None, optional
        Stoichiometry of the material. By default `None`.
    formula_mass : float | None, optional
        Atomic mass sum of the materials chemical formula (molecular mass).
        Equivalent to providing a stoichiometry. By default `None`.
    is_extended : bool, optional
        `True` if the material has been extended by the KKCalc database.
        By default `False`.
    """

    def __init__(
        self,
        name: str | None = None,
        number_density: float | None = None,
        density: float | None = None,
        stoichiometry: kk_stoichiometry | CompositionAlias | None = None,
        formula_mass: float | None = None,
        is_extended: bool = False,
    ) -> None:  # numpydoc ignore=GL08
        super().__init__()
        # Initialize internal attributes.
        self._name = name
        self._number_density = None
        self._density = None
        self._stoichiometry = None
        self._formula_mass = None
        self._is_extended = False

        # Raise a warning if competing information is provided.
        if (
            number_density is not None
            and density is not None
            and stoichiometry is not None
        ):
            warnings.warn(
                "Competing information provided for `number density` and `density` given a `stoichiometry`. "
                "`Number density` information precedes `density`.",
                UserWarning,
            )

        # Assign in reverse order of importance.
        self.stoichiometry = stoichiometry  # can infer a formula mass
        self.formula_mass = formula_mass
        self.density = density
        self.number_density = number_density

        # Finally assign if the material has been extended by the KKCalc database.
        self._is_extended = is_extended  # has to be done after the other assignments, otherwise cannot set stoichiometry.
        return

    @atomic_scattering_abstract.name.getter
    def name(self) -> str | None:  # numpydoc ignore=PR02
        """
        The name of the material/sample associated with the scattering factors.

        Parameters
        ----------
        name : str
            Material/sample name.

        Returns
        -------
        str
            The Material/sample name. If no name but a `stoichiometry` is provided,
            returns the stoichiometry string. If no `s`toichiometry` either, then returns `None`.
        """
        if self._name is None:
            if self.stoichiometry is not None:
                return str(self.stoichiometry)
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:  # numpydoc ignore=GL08
        self._name = name
        return

    @property
    def number_density(self) -> float | None:  # numpydoc ignore=PR02
        """
        The material number density in atoms per millilitre (cm^3).

        When setting a value,
        - Generates/Updates `density` if the `formula_mass` or `stoichiometry` is defined.

        Parameters
        ----------
        number_density : float | None
            Material number density in atoms per millilitre (cm^3).

        Returns
        -------
        float | None
            Material number density in atoms per millilitre (cm^3).
        """
        return self._number_density

    @number_density.setter
    def number_density(
        self, number_density: float | None
    ) -> None:  # numpydoc ignore=GL08
        self._number_density = number_density
        if number_density is not None:
            if self.formula_mass is not None:
                # Generate a density from the formula mass and number density.
                self._density = number_density * self.formula_mass / N_A
        return

    @property
    def density(self) -> float | None:  # numpydoc ignore=PR02
        """
        The material density in grams per millilitre (cm^3).

        When setting a value that is not `None`,
        - Generates/Updates `number_density` if the `stoichiometry` or `formula_mass` is defined.
        - Generates `formula_mass` if None, and updates `number_density`, if the `stoichiometry` is defined.
        - Generates `formula_mass` if None, if the `number_density` is defined.

        Parameters
        ----------
        density : float | None
            Material density in grams per millilitre (cm^3).

        Returns
        -------
        float | None
            Material density in grams per millilitre (cm^3).
        """
        return self._density

    @density.setter
    def density(self, density: float | None) -> None:  # numpydoc ignore=GL08
        self._density = density
        if density is not None:
            if self.formula_mass is not None:
                # Update / generate a number density from the formula mass and density.
                self._number_density = density * N_A / self.formula_mass
            elif self.stoichiometry is not None:
                # Generate a formula mass from the stoichiometry and density.
                fm = self.stoichiometry.formula_mass
                self._formula_mass = fm
                self._number_density = density * N_A / fm
            elif self.number_density is not None and self.stoichiometry is None:
                # Generate a formula mass from the number density and density.
                self._formula_mass = density * N_A / self.number_density
        return

    @property
    def formula_mass(self) -> float | None:  # numpydoc ignore=PR02
        """
        The atomic mass sum of the materials chemical formula (or molecular mass).

        In units of atomic mass units (amu).

        When getting a value that is not `None`,
        - Returns the `stoichiometry.formula_mass` if `stoichiometry` is defined.
        - Alternatively returns the `formula_mass` if it has been defined.

        When setting a value,
        - Generates/Updates `number_density` if the `density` is defined.
        - Generates `density` if `None`, if the `number_density` is defined.

        Parameters
        ----------
        formula_mass : float | None
            Atomic mass sum of the materials chemical formula (molecular mass).

        Returns
        -------
        float | None
            Atomic mass sum of the materials chemical formula.
        """
        if self.stoichiometry is not None:
            return self.stoichiometry.formula_mass
        return self._formula_mass

    @formula_mass.setter
    def formula_mass(self, formula_mass: float | None) -> None:  # numpydoc ignore=GL08
        self._formula_mass = formula_mass
        if self.stoichiometry is not None:
            warnings.warn(
                "Setting a formula mass will not be internally used when a `stoichiometry` has been assigned.",
                UserWarning,
            )
        else:
            if formula_mass is not None:
                # Update / generate a number density from the formula mass and density.
                if self.density is not None:
                    self._number_density = self.density * N_A / formula_mass
                # Generate a density from the formula mass and number density.
                elif self.number_density is not None:
                    self._density = self.number_density * formula_mass / N_A
        return

    @property
    def stoichiometry(self) -> kk_stoichiometry | None:  # numpydoc ignore=PR02
        """
        The `stoichiometry` of the material associated with the scattering factors.

        Cannot be set when `is_extended` is `True`; implies the use of the existing `stoichiometry` to create the scattering data, and therefore is immutable.

        When setting a value that is not `None`,
        - Generates/Updates `formula_mass` when set, using the stoichiometry.
        - Generates/Updates `number_density` if the `density` is defined.
        - Generates `density` if `None`, if the `number_density` is defined.

        Parameters
        ----------
        stoich : stoichiometry | str | None
            Stoichiometry of the material.

        Returns
        -------
        stoichiometry | None
            Stoichiometry of the material.
            `None` if no stoichiometry has been provided.

        Raises
        ------
        ValueError
            If the object property `is_extended` is True, due to the object
            being created from data in the KKCalc database.

        See Also
        --------
        kk_stoichiometry : Stoichiometry class.
        """
        return self._stoichiometry

    @stoichiometry.setter
    def stoichiometry(
        self, stoich: kk_stoichiometry | str | None
    ) -> None:  # numpydoc ignore=GL08
        # Convert a string to a stoichiometry object.
        if isinstance(stoich, str):
            stoich = kk_stoichiometry(stoich)

        if not self.is_extended:
            # Generate / update the formula mass and number density.
            if stoich is not None:
                # Generate a formula mass from the stoichiometry.
                # Modify private attribute before stoichiometry, to avoid immutable error.
                self.formula_mass = stoich.formula_mass
                if self.density is not None:
                    # Update / generate a number density from the stoichiometry
                    # and density, regardless of the current number density.
                    self._number_density = self.density * N_A / stoich.formula_mass
                elif self.number_density is not None:
                    # Generate a density from the stoichiometry and number density.
                    self._density = self.number_density * stoich.formula_mass / N_A

            # Set the stoichiometry attribute after the formula mass has been set.
            self._stoichiometry = stoich
        else:
            raise ValueError(
                "Stoichiometry is immutable on a dataset once extended by the KKCalc database."
            )
        return

    @property
    def is_extended(self) -> bool:  # numpydoc ignore=PR02
        """
        Property of the object if it has been extended by the KKCalc database.

        Parameters
        ----------
        is_extended : bool
            `True` if the material has been extended by the KKCalc database.

        Returns
        -------
        bool
            `True` if the material has been extended by the KKCalc database.
        """
        return self._is_extended

    @is_extended.setter
    def is_extended(self, is_extended: bool) -> None:  # numpydoc ignore=GL08
        self._is_extended = is_extended

    # Override the _properties_dict property, to properly document.
    @property
    def _properties_dict(self) -> PROPERTIES_DICT:  # numpydoc ignore=PR02
        """
        Property for the material class properties.

        Parameters
        ----------
        properties : PROPERTIES_DICT
            Keyword arguments for class properties. `None` values are ignored.

        Returns
        -------
        dict
            Dictionary of class material properties.
        """
        f = atomic_scattering_abstract._properties_dict.fget
        assert f is not None
        return f(self)

    @_properties_dict.setter
    def _properties_dict(
        self, properties: PROPERTIES_DICT
    ) -> None:  # numpydoc ignore=GL08
        for key in properties:
            match key:
                case "name":
                    self.name = properties[key]  # type: ignore - Bug in PyLance ## TODO: Submit an issue using match statement.
                case "stoichiometry":
                    self.stoichiometry = properties[key]  # type: ignore
                case "density":
                    self.density = properties[key]  # type: ignore
                case "number_density":
                    self.number_density = properties[key]  # type: ignore
                case "formula_mass":
                    self.formula_mass = properties[key]  # type: ignore
                case "is_extended":
                    self.is_extended = properties[key]  # type: ignore
                case _:
                    raise ValueError(f"Invalid property: {key}")
        return

    def copy(self) -> Self:
        """
        Return a copy of the class instance.

        Returns
        -------
        atomic_scattering
            Copy of the class instance.
        """
        cls = self.__class__
        return cls(
            name=self.name,
            number_density=self.number_density,
            density=self.density,
            stoichiometry=self.stoichiometry,
            formula_mass=self.formula_mass,
            is_extended=self.is_extended,
        )
