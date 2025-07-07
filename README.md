# KKcalc
Kramers-Kronig Calculator and toolkit for material optical properties.

[![Coverage Status](https://coveralls.io/repos/github/xraysoftmat/kkcalc/badge.svg?branch=v2)](https://coveralls.io/github/xraysoftmat/kkcalc?branch=v2) [![Code style: black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/psf/black) [![Code doc: numpydoc](https://img.shields.io/badge/doc_style-numpydoc-blue.svg)](https://github.com/numpy/numpydoc) ![example workflow](https://github.com/xraysoftmat/kkcalc/actions/workflows/test.yml/badge.svg)

## Introduction
`kkcalc` is an open-source `python` package to calculate the (inverse) Kramers-Kronig transform of X-ray absorption (dispersion) data,

$$f_2 (E) = \frac{2}{\pi} P \int_{0}^{\infty}\frac{x f_1(x)}{x^2 - E^2} dx + \mathcal{Z}^\star$$

using a polynomial representation algorithm developed by Watts^[[1](#1)].

This package also provides an object oriented API, to evaluate optical constants (index of refraction, absorption and dispersion, etc.), extend measurement spectra.

`KKCalc` is usable via an object-oriented Python API, or through a PyQT6 GUI interface. Documentation can be found at [readthedocs](https://kkcalc.rtfd.org/), or can be [built](#docbuild) for offline.

Cite our previous work for this repository here[<sup>[1]</sup>](#1).

## Features
- **Extend and Scale**
   Scale and extend your NEXAFS datasets by the Henke^[[2](#2)] and Briggs/Lighthill^[[3](#3)] atomic scattering factor databases, for more accurate transforms. Stitch together multiple datasets.
- **Relativistic Correction**
   Use your material composition to automatically calculate the relativistic correction, $f^0$.
- **GUI Interface**
   No programming? No problem! Run the python module (executable coming soon).
- **Contrast Calculations**
   Can calculate the relative contrast between materials in a mix.

## Installation

##### PYPI Installation

[`KKcalc`](https://pypi.org/project/kkcalc/) is [registered](https://pypi.org/project/kkcalc/) on the [`PyPI`](https://pypi.org/) (Python Package Index) system, and can be installed using a cmd/bash terminal.

    pip install kkcalc

There are also optional dependency groups `dev`, `gui` and `docs` that can be installed as such (Note dependency groups are a [new](https://peps.python.org/pep-0735/) feature and requires `pip>=25.1`, you can upgrade with `python -m pip install --upgrade pip`):

    pip install kkcalc --group gui

The `gui` group is recommended to use the easy-access PyQt features.


Further details about `pip` usage can be found in the [PyPI installation tutorial](https://packaging.python.org/tutorials/installing-packages/).

##### Dependencies
Current dependencies can be found in the repository [`pyproject.toml`](pyproject.toml) file.

#### Usage

###### GUI

`KKCalc` has a modularised GUI interface that can be used without requiring the python API.

![Screenshot of the KKcalc GUI interface](docs/_static/kk_calc.jpg)

###### Python

`kkcalc` is an object-oriented, typed library. The class structure is depicted below. Of note are the `asf` (atomic scattering factor) and `asp_db_extended` (atomic scattering polynomial database) classes, from which data loading and most Kramers-Kronig operations can be performed.

![Architecture graphic of the class structure in KKCalc](https://docs.google.com/drawings/d/1Py6hnj8KXS7-dePQF1435XrBfsZQHaX1XQZ-ol2J7rA/export/png)

If you are keen to include `kkcalc` into your own `python` scripts, use a modern code editor that can intepret docstrings, such as VSCode or PyCharm to get hints as you code.

#### Contributing & Developement
Want to contribute? That's fantastic! Have a discussion with us first via Issues/Discussions to avoid wasted effort, and so we can make our visions align.

To install for development, create your own [fork/branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of `kkcalc` on `github` so you can save and contribute your own changes, and clone that to your local file system using `Github Desktop` or `Git` via Windows Terminal / CMD.

    git clone https://github.com/<myusername>/kkcalc

  Change directory to the repository, and install an editable version of the repository to your system or a [virtual environment](https://docs.python.org/3/library/venv.html), so modifications will be reflected upon `kkcalc` import.

    # Create a virtual environment to avoid conflict with other python packages
    python -m venv <name>         # Often just use 'venv' as the name.

    # [Activate the virtual environment](https://docs.python.org/3/library/venv.html#how-venvs-work)
    <name>\Scripts\activate.bat   # Windows
    source <name>/bin/activate    # Bash

    # Install the library from the source code
    pip install -e . --group dev

  `kkcalc` uses a combination of continuous integration (CI) tools to manage the code quality and documentation. These include
  - [`pre-commit`](https://pre-commit.com/): To fix consistency issues in code before pushing to the repository.
  - [`black`](https://github.com/psf/black): To standardize code formatting so the entire repository is consistent and readable.
  - [`numpydoc`](https://numpydoc.readthedocs.io/): Readable documentation to standard with other scientific python packages.
  <!-- - [`pytest`]() -->

  Install these hooks as follows:

    pre-commit install

#### Documentation <a id=docbuild></a>
To build the documentation, install the documentation dependencies from the package directory:

    pip install . --group docs

To build the documentation, run

    cd docs
    make html

which should create a new set of static `HTML` files in the `/docs/_build` directory.

To update the documentation after edits, run:

    sphinx-apidoc -o ./docs ./kkcalc

in the source directory, then rebuild as above.

## References

<a id=1><sup>[1]</sup></a> Benjamin Watts, "Calculation of the Kramers-Kronig transform of X-ray spectra by a piecewise Laurent polynomial method", *Opt. Express* **22**, (2014) 23628-23639. [`DOI:10.1364/OE.22.023628`](https://doi.org/10.1364/OE.22.023628)

<a id=1><sup>[2]</sup></a> B.L. Henke, E.M. Gullikson, and J.C. Davis, "X-ray interactions: photoabsorption, scattering, transmission, and reflection at E=50-30000 eV, Z=1-92", *Atomic Data and Nuclear Data Tables* **54** (2) (1993) 181-342 [`DOI:10.1006/adnd.1993.1013`](https://doi.org/10.1006/adnd.1993.1013).

<a id=1><sup>[3]</sup></a> F. Biggs, and R. Lighthill, "Analytical approximations for X-ray cross-sections III", *Sandia Report* SAND87-0070 UC-34 (1988). [`DOI:10.2172/7124946`](https://doi.org/10.2172/7124946)
