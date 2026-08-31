"""
Pytest fixtures providing access to the example NEXAFS raw data bundled in `kkcalc2/data`.

Each example material/edge is parametrized through the `nexafs_material` fixture, with
further fixtures building the data up into each representation:
- `nexafs_material`: NEXAFS (default, raw photoabsorption data) and material properties.
- `nexafs_asf`: Atomic scattering factors (ASF): real, imaginary and complex.
- `nexafs_refractive`: Refractive (delta/beta index of refraction) components: real, imaginary and complex.
- `nexafs_refractive_index`: Refractive index (n = 1 - delta + i*beta): real, imaginary and complex.
"""

import io
import pkgutil
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
import pytest

from kkcalc2.models.factors import asf_complex, asf_im, asf_re


@dataclass
class NexafsMaterial:
    """Metadata and raw NEXAFS data for a single example material/absorption edge."""

    key: str
    """Short identifier for the dataset."""
    name: str
    """Material/sample name."""
    stoichiometry: str
    """Chemical formula of the material."""
    density: float
    """Typical material density in g/cm^3."""
    data_path: str
    """Package-relative path (under `kkcalc2`) to the raw data file."""
    skip_header: int = 0
    """Number of header lines to skip when parsing the raw data file."""
    delimiter: str | None = None
    """Column delimiter for the raw data file. `None` for whitespace-delimited files."""

    @cached_property
    def _raw_data(self) -> npt.NDArray:
        raw = pkgutil.get_data("kkcalc2", self.data_path)
        assert raw is not None, f"Could not load packaged data file '{self.data_path}'."
        return np.genfromtxt(
            io.BytesIO(raw), skip_header=self.skip_header, delimiter=self.delimiter
        )

    @property
    def energies(self) -> npt.NDArray:
        """Photon energies (eV) of the raw NEXAFS data."""
        return self._raw_data[:, 0]

    @property
    def nexafs(self) -> npt.NDArray:
        """Default representation: raw NEXAFS photoabsorption data."""
        return self._raw_data[:, 1]


@dataclass
class NexafsAsf:
    """Atomic scattering factor (ASF) representations of an example NEXAFS dataset."""

    im: asf_im
    """Imaginary atomic scattering factors, derived directly from the NEXAFS data."""
    complex: asf_complex
    """Complex atomic scattering factors, KK transformed on the original energy grid."""

    @property
    def re(self) -> asf_re:
        """Real atomic scattering factors, from the KK transform of the imaginary part."""
        return self.complex.re


@dataclass
class NexafsRefractive:
    """Refractive index component (delta/beta) representations of an example NEXAFS dataset."""

    real: npt.NDArray
    """Real (dispersive, delta) refractive component."""
    imag: npt.NDArray
    """Imaginary (absorptive, beta) refractive component."""
    complex: npt.NDArray
    """Complex refractive value (delta + i*beta)."""


@dataclass
class NexafsRefractiveIndex:
    """Refractive index (n = 1 - delta + i*beta) representations of an example NEXAFS dataset."""

    real: npt.NDArray
    """Real refractive index component (n = 1 - delta)."""
    imag: npt.NDArray
    """Imaginary refractive index component (n = beta)."""
    complex: npt.NDArray
    """Complex refractive index (n = 1 - delta + i*beta)."""


# Example NEXAFS datasets bundled with kkcalc2, using a typical density for each material.
NEXAFS_MATERIALS: dict[str, NexafsMaterial] = {
    "P3MEEET_O": NexafsMaterial(
        key="P3MEEET_O",
        name="P3MEEET Oxygen K",
        stoichiometry="C11H16O3S",
        density=1.3,
        data_path="data/P3MEEET_Oxygen_K.csv",
        delimiter=",",
    ),
    "P3MEEET_S": NexafsMaterial(
        key="P3MEEET_S",
        name="P3MEEET Sulfur K",
        stoichiometry="C11H16O3S",
        density=1.3,
        data_path="data/P3MEEET_Sulfur_K.csv",
        delimiter=",",
    ),
    "PEDOT_O": NexafsMaterial(
        key="PEDOT_O",
        name="PEDOT-C6C8 Oxygen K",
        stoichiometry="C21H36O2S",
        density=1.2,
        data_path="data/PEDOTC6C8_Oxygen_K.csv",
        delimiter=",",
    ),
    "PEDOT_S": NexafsMaterial(
        key="PEDOT_S",
        name="PEDOT-C6C8 Sulfur K",
        stoichiometry="C21H36O2S",
        density=1.2,
        data_path="data/PEDOTC6C8_Sulfur_K.csv",
        delimiter=",",
    ),
    "PS_C": NexafsMaterial(
        key="PS_C",
        name="Polystyrene Carbon K",
        stoichiometry="C8H8",
        density=1.05,
        data_path="data/PS_004_-dc.txt",
        skip_header=4,
    ),
}


@pytest.fixture(
    params=list(NEXAFS_MATERIALS.values()),
    ids=list(NEXAFS_MATERIALS.keys()),
    scope="session",
)
def nexafs_material(request: pytest.FixtureRequest) -> NexafsMaterial:
    """Parametrized fixture providing the raw NEXAFS data (default representation) for each example material."""
    return request.param


@pytest.fixture(scope="session")
def nexafs_asf(nexafs_material: NexafsMaterial) -> NexafsAsf:
    """Atomic scattering factor (real, imaginary and complex) representation of the example material."""
    im = asf_im.from_NEXAFS(
        energies=nexafs_material.energies,
        NEXAFS=nexafs_material.nexafs,
        name=nexafs_material.name,
        stoichiometry=nexafs_material.stoichiometry,
        density=nexafs_material.density,
    )
    # Use improve_accuracy=False so the real & imaginary parts share the original energy grid.
    complex_ = im.calculate_complex_factors(improve_accuracy=False)
    return NexafsAsf(im=im, complex=complex_)


@pytest.fixture(scope="session")
def nexafs_refractive(nexafs_asf: NexafsAsf) -> NexafsRefractive:
    """Refractive (delta/beta) component (real, imaginary and complex) representation of the example material."""
    return NexafsRefractive(
        real=nexafs_asf.re.deltas,
        imag=nexafs_asf.im.betas,
        complex=nexafs_asf.complex.refractive,
    )


@pytest.fixture(scope="session")
def nexafs_refractive_index(
    nexafs_refractive: NexafsRefractive,
) -> NexafsRefractiveIndex:
    """Refractive index (n = 1 - delta + i*beta) (real, imaginary and complex) representation of the example material."""
    return NexafsRefractiveIndex(
        real=1 - nexafs_refractive.real,
        imag=nexafs_refractive.imag,
        complex=1 - nexafs_refractive.real + 1j * nexafs_refractive.imag,
    )
