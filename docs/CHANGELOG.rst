.. _changelog:

=========
CHANGELOG
=========

..
    version list

.. _changelog-v1.1.0:

v1.1.0 (2026-09-03)
===================

Bug Fixes
---------

* **__init__.py**: Remove gui import into toplevel module (`649b305`_)

* **__main__.py**: Incorrect pathing to polystyrene data for GUI, updated for kkcalc2 (`49f6bf6`_)

* **adjust-instantiation-for-complex-classes**: Avoids a bug where classes attempted to use abstract
  asf class, rather than the im or re versions (`37f1586`_)

* **asp**: Fix asp.extend_energies bug (`d566225`_)

* **asp_db_im_extended**: Added shortcut method to input nexafs data to create an extended db object
  (`f05cfd6`_)

* **common.py**: Ensure fomula masses utilise stoichiometry first if available (`a1c906f`_)

* **common.py**: Remove variable propogation errors (`d844205`_)

* **dialogs.py**: Adjusts pandas to be the default loader, and fixes header default kwarg
  (`3ea8d11`_)

* **polynomials**: Add properties for refractive index values to ASP (`3319e73`_)

* **poylnomials.py**: Add none check before using np.asarray (`f027313`_)

* **transforms.py**: Adjust the inverse transform to subtract the relativistic correction (atomic
  number) before transforming (`dcbc768`_)

* **transforms.py**: Reduced ``DEF_ITER`` to 10 iterations, to avoid division beyond floating
  precision (`26c7d23`_)

* **update-loader-for-minor-bugs**: Firsttly prevents failure with None stoichiometry for combining
  multiple edges. secondly adjusts to use basename instead of full path for a new instance name
  (`1f49cfa`_)

Build System
------------

* **deps**: Update numpy requirement from ~=2.4.0 to >=2.4,<2.6 (`219489c`_)

* **deps**: Update scipy requirement from ~=1.17.1 to >=1.17.1,<1.19.0 (`f652f1a`_)

Documentation
-------------

* **index.rst, contributing.rst**: Fix broken links to github (`8d4f360`_)

* **install.rst**: Add short section for periodictable dependency (`b24a974`_)

* **install.rst, api.rst**: Minor doc fixes (`164b077`_)

* **README.md**: Add zenodo reference (`e750040`_)

Features
--------

* **add-periodictable-backend**: Add alternative implementation for periodictable calculations
  (`3a77da6`_)

* **asf_modifier.py**: Add new GUI implementation of fix distortions, that allow for the two methods
  to be selected (`2a1bb40`_)

* **db_models.py**: Add new datascaling method ``prepost_fit`` (`180fb3d`_)

.. _164b077: https://github.com/xraysoftmat/kkcalc/commit/164b0771762d9f2f2199d4f71dbe7ea09f337540
.. _180fb3d: https://github.com/xraysoftmat/kkcalc/commit/180fb3d679479ead6713d2b56c476e51ef8c0f73
.. _1f49cfa: https://github.com/xraysoftmat/kkcalc/commit/1f49cfa6d76af42fa3b9406b725f1baaf6b19062
.. _219489c: https://github.com/xraysoftmat/kkcalc/commit/219489cf91b37779cd6cee500f08c2374cb9f432
.. _26c7d23: https://github.com/xraysoftmat/kkcalc/commit/26c7d23c280c29f750fc64a35c9e8d684bd2da4c
.. _2a1bb40: https://github.com/xraysoftmat/kkcalc/commit/2a1bb405efea3c212f0f0adf5245fa6fbf12cdcf
.. _3319e73: https://github.com/xraysoftmat/kkcalc/commit/3319e73ae6a9b8bdc937af89d473f82f86b75412
.. _37f1586: https://github.com/xraysoftmat/kkcalc/commit/37f15863bc7c7dc30e0ceecbddfe21744354c0df
.. _3a77da6: https://github.com/xraysoftmat/kkcalc/commit/3a77da6c2f8ff55f428439caf876056b39858bcc
.. _3ea8d11: https://github.com/xraysoftmat/kkcalc/commit/3ea8d1104a836cebc65839a937c56af2b4021275
.. _49f6bf6: https://github.com/xraysoftmat/kkcalc/commit/49f6bf692606fe31f31b759d3526117cbfe9cce4
.. _649b305: https://github.com/xraysoftmat/kkcalc/commit/649b30551dca829e5d68e3cae141d188a74c81af
.. _8d4f360: https://github.com/xraysoftmat/kkcalc/commit/8d4f360fd95c6bbf7b85c84d6f934faa4304c488
.. _a1c906f: https://github.com/xraysoftmat/kkcalc/commit/a1c906fbb7814c4f9d6bd2eafe6be00cb4becf3d
.. _b24a974: https://github.com/xraysoftmat/kkcalc/commit/b24a9742d72def8b9ba9bc75d0d017d780c6c9e6
.. _d566225: https://github.com/xraysoftmat/kkcalc/commit/d566225b6c88a8fe72f460e13d35baaa5a4f2d26
.. _d844205: https://github.com/xraysoftmat/kkcalc/commit/d844205a8edf5a84a13cec860198067c85628645
.. _dcbc768: https://github.com/xraysoftmat/kkcalc/commit/dcbc768a950c8179b34ceb3ae40deb55988735eb
.. _e750040: https://github.com/xraysoftmat/kkcalc/commit/e750040f96fded5cf8044282d6db9a84e96d353c
.. _f027313: https://github.com/xraysoftmat/kkcalc/commit/f027313485b300a017dd746f65d7960ffc14cbf5
.. _f05cfd6: https://github.com/xraysoftmat/kkcalc/commit/f05cfd625de2f4d41424d4eaa4e9c222acd57d28
.. _f652f1a: https://github.com/xraysoftmat/kkcalc/commit/f652f1a1268550a3e32bc736caf8d117aa1703a3


.. _changelog-v1.0.0:

v1.0.0 (2026-03-17)
===================

Bug Fixes
---------

* **asf_database**: Adjust filepaths and data loading for asf database (`81b0925`_)

* **asf_database**: Reduce database size and document code (`6527361`_)

* **asf_viewer.py**: Added cases for NEXAFS type checking, and continue conditions to avoid UI
  crashes (`16f5c55`_)

* **asf_viewer.py**: Added data treatment for ``refractive`` KK_Datatype option in the GUI, instead
  of crashing (`4afbd99`_)

* **asp_complex**: Ensured common kwargs propogated into asf conversions (`45dc1f0`_)

* **common.py**: Make density information more robust to modify (`0366931`_)

* **common.py**: Modify propogation and setting of internal variables (`8f71b98`_)

* **db_models.asp_db_complex**: Added an override for the asp_complex copy method, to copy the
  energies and coefficients to generate a new object instead of copying asp_im and asp_re
  components. This conforms to the asp_db_complex constructor for copying (`2f25c45`_)

* **db_models.py**: Add and fix scaling to db classes (`f73128f`_)

* **db_models.py**: Changed ``asp_db_extended`` constructor so that copy method is compatible
  withchecking of ``merge_domain`` (`045ac2d`_)

* **db_models.py**: Documentation and implementation changes (`38fd657`_)

* **db_models.py**: Fix copy for asf_db_extended (`6d2eacd`_)

* **db_models.py**: Fixed required kwargs bug for initialising energies & coefs for a polynomial
  class (`bb70685`_)

* **gui**: GUI updates and pyproject changes to numpydoc file checking (`7f10a53`_)

* **kkcalc/models/**: Changed models to properly reflect refractive naming convention for dispersion
  ($\delta$) and absorption ($\beta$) (`d3c819c`_)

* **models**: Fixed the order of a few error messages. Also some invalid constructor calling
  (`023b578`_)

* **models/__init__.py**: Add ``asp_db_abstract`` class to **all** map (`9990171`_)

* **polynomials.py**: Added fix to creating factors from a complex polynomial that had different
  coefficient lengths (`4242790`_)

* **polynomials.py**: Fixed bug in ``asp`` methods where properties_dict wasn't propogating to new
  class (`5949ddc`_)

* **polynomials.py**: Remove target_energies bug (`b8dbad5`_)

* **polynomials.py**: Updated **repr** and **str** methods to better reflect debugging and naming
  object information, respectively (`b81d46c`_)

* **polynomials.py,-factors.py**: Propogate ``atomic_scattering`` properties (`fd81bcb`_)

* **pyinst.yml**: Modify wkflw to gen and upld exe (`06ad149`_)

* **stoich.py**: Added support for formula objects and dictionary objects to construct stiochiometry
  (`b8cb7d6`_)

* **stoich.py**: Dependency on ``periodictable`` updated **iter** method, no longer provides neutron
  mass as first element (`f2fc6af`_)

* **stoich.py**: Fixed inclusion of neutron element to ELEMENT table (`bd4b05a`_)

* **stoichiometry**: Corrected fractional addition and added additional methods between stochiometry
  objects (`a5078be`_)

* **util.py**: Let doc_copy work with @property decorators (`49110da`_)

Build System
------------

* **deps**: Update numpy requirement from ~=2.3.4 to ~=2.4.0 (`5fcb699`_)

* **deps**: Update scipy requirement from ~=1.16.2 to ~=1.17.1 (`3647a79`_)

Documentation
-------------

* **CHANGELOG.rst**: Reset listed versions (`25c94e6`_)

* **conf.py**: Switch to using pydata-sphinx-theme (`2d7ec44`_)

* **conf.py,-pyproject.toml**: Add new configuration for sphinx build (`6495ddb`_)

* **docs**: Remove old doc files (`f741969`_)

* **example_functions.ipynb**: Added example functions for kk transform, will be worked into tests
  later (`16db890`_)

* **examples,-kkcalc**: Add missing docstrings (`229e8d9`_)

* **factors.py,-conversions.py**: Modified the NEXAFS labels to directly reference the photo
  absorption cross section (`211c25a`_)

* **kk_transforms.py**: Added doc changes, and a single extra warning for improve_accuracy
  functionality (`a4f568f`_)

* **README.md**: Added badge for successful/failing workflow usage (`c582fe1`_)

* **README.md**: Added shield.io CI mapping, and badges for numpydoc and black (`44b4d33`_)

* **README.md**: Updated docs to use dependency groups, moved gui doc image to static folder
  (`ba7ccb5`_)

* **README.rst**: Fix formatting (`4284c1a`_)

* **README.rst**: Fix invalid rst for PyPI (`ffa1f6e`_)

* **README.rst**: Update pre-commit badge to v2 branch (`1f27d82`_)

* **stoich.py**: Re-added GL08 comment for stoich constructor, until numpydoc is updated
  (`34b23e8`_)

* **stoich.py**: Signature changes for the PROPERTIES_DICT TypedDict, documentation changes and
  alias changes (`185dd63`_)

Features
--------

* **__main__.py**: Added a main entry point for GUI when using python load module (i.e. python -m
  kkcalc) (`5401d52`_)

* **docs**: Sphinx docs infrastructure (`bdfe806`_)

* **factors.py**: Added NEXAFS property and methods for asf_complex class (`c6ab85a`_)

* **kkcalc\models\***: Added new methods for converting and evaluating refractive compnents /
  indexes (`8cb4958`_)

* **polynomials.py**: Added critical angle method to asp_complex (`2879adb`_)

* **polynomials.py**: Calculation of the attenuation length / penetration depth from the absorption
  coefficients (`e5a0a72`_)

.. _023b578: https://github.com/xraysoftmat/kkcalc/commit/023b5783595ea1ad1f056e74272eebd48bb63f85
.. _0366931: https://github.com/xraysoftmat/kkcalc/commit/03669316ab923b79412a93a9aa026933e194b492
.. _045ac2d: https://github.com/xraysoftmat/kkcalc/commit/045ac2dafa3e7404496a175af99fbaee49ea3c2e
.. _06ad149: https://github.com/xraysoftmat/kkcalc/commit/06ad14936c618841ca77053544e0386a1c3bd769
.. _16db890: https://github.com/xraysoftmat/kkcalc/commit/16db8906ad9d5ef97e82b09bd9571a5cafe028d0
.. _16f5c55: https://github.com/xraysoftmat/kkcalc/commit/16f5c55bf4b4a0e6de390c82c03628dcf3a7ceea
.. _185dd63: https://github.com/xraysoftmat/kkcalc/commit/185dd63ed22703e226824a34a1d40fbe4f9691f1
.. _1f27d82: https://github.com/xraysoftmat/kkcalc/commit/1f27d8226c91253d2f465616b74c8098d08065ca
.. _211c25a: https://github.com/xraysoftmat/kkcalc/commit/211c25ad88fec017b1f5746bc9455a406555c930
.. _229e8d9: https://github.com/xraysoftmat/kkcalc/commit/229e8d9e5ef478dc57a60b1ff5f6c5b697a42887
.. _25c94e6: https://github.com/xraysoftmat/kkcalc/commit/25c94e6e358acbcebbb70a394fbc35a7ae4f72dd
.. _2879adb: https://github.com/xraysoftmat/kkcalc/commit/2879adb0f94a5e6aa20d2e47da9bbcb8ab4d5851
.. _2d7ec44: https://github.com/xraysoftmat/kkcalc/commit/2d7ec4401d1f85722abe7c95b6d6cf1bc4bf2dbf
.. _2f25c45: https://github.com/xraysoftmat/kkcalc/commit/2f25c4583c093f20b1e41fd5cea403346bfac592
.. _34b23e8: https://github.com/xraysoftmat/kkcalc/commit/34b23e8e7a3f861f165dd51a4ce6eee8fdf355c3
.. _3647a79: https://github.com/xraysoftmat/kkcalc/commit/3647a79f337e765d1d20d9f2234684e31f68f44b
.. _38fd657: https://github.com/xraysoftmat/kkcalc/commit/38fd6576af8b3168e90a61d9b0bb8cccd27d01bf
.. _4242790: https://github.com/xraysoftmat/kkcalc/commit/4242790a51449e82f81eec9de69dc17a2342572f
.. _4284c1a: https://github.com/xraysoftmat/kkcalc/commit/4284c1ac3d1378d41ad1c1b00d6692bbc6f213b4
.. _44b4d33: https://github.com/xraysoftmat/kkcalc/commit/44b4d3336bebb22b1e8c9e27083c46ed2815dc01
.. _45dc1f0: https://github.com/xraysoftmat/kkcalc/commit/45dc1f08bd3b62287d88d03b9673f33cfefa909a
.. _49110da: https://github.com/xraysoftmat/kkcalc/commit/49110da623b6ddb15a48ae4f50026fe66f6725c2
.. _4afbd99: https://github.com/xraysoftmat/kkcalc/commit/4afbd99c659241948c5e5a776bbc87e51216df76
.. _5401d52: https://github.com/xraysoftmat/kkcalc/commit/5401d5242842cd943c1efbef098e06d771021bfd
.. _5949ddc: https://github.com/xraysoftmat/kkcalc/commit/5949ddc4255786a6392b3d835ae38114afd42ff2
.. _5fcb699: https://github.com/xraysoftmat/kkcalc/commit/5fcb69905f766af7c0beec6c9313fb4778d4bd2e
.. _6495ddb: https://github.com/xraysoftmat/kkcalc/commit/6495ddbcd23bd2d03a58730c3fe9eb6047d0ecb7
.. _6527361: https://github.com/xraysoftmat/kkcalc/commit/65273616ed74298fc7f62858b8e85cb7d50f9600
.. _6d2eacd: https://github.com/xraysoftmat/kkcalc/commit/6d2eacd278f62888d1e1ae05687f97241f5c8476
.. _7f10a53: https://github.com/xraysoftmat/kkcalc/commit/7f10a538b9d96efe2d574227ab05838403fa1585
.. _81b0925: https://github.com/xraysoftmat/kkcalc/commit/81b092520f57828760b94d68cee37704700761f7
.. _8cb4958: https://github.com/xraysoftmat/kkcalc/commit/8cb495863bbea661ab6206990ff4db799cfac224
.. _8f71b98: https://github.com/xraysoftmat/kkcalc/commit/8f71b9844fa9b41f7ac487f5ab3c93c47a233797
.. _9990171: https://github.com/xraysoftmat/kkcalc/commit/9990171e2ca519386e4ab9c1dbeec006a120b2b8
.. _a4f568f: https://github.com/xraysoftmat/kkcalc/commit/a4f568f2738c80126fe123f93b321d7b49fbf70a
.. _a5078be: https://github.com/xraysoftmat/kkcalc/commit/a5078be58cbe7f907689ee5cf8c82cbc2fc056fb
.. _b81d46c: https://github.com/xraysoftmat/kkcalc/commit/b81d46c9862f0213446252f15961a277965d61ff
.. _b8cb7d6: https://github.com/xraysoftmat/kkcalc/commit/b8cb7d67ddfbf64f1e9f1fe86d8dfa04992db0aa
.. _b8dbad5: https://github.com/xraysoftmat/kkcalc/commit/b8dbad5a836ddfb88a6591e9013877ecdc4726ad
.. _ba7ccb5: https://github.com/xraysoftmat/kkcalc/commit/ba7ccb51ada32c06902b64574d6159a7a37522d6
.. _bb70685: https://github.com/xraysoftmat/kkcalc/commit/bb706859e201f0f6e741955b398739e212abbab3
.. _bd4b05a: https://github.com/xraysoftmat/kkcalc/commit/bd4b05a27fbcbecb7dcd893fdc560ab2b815c892
.. _bdfe806: https://github.com/xraysoftmat/kkcalc/commit/bdfe80672cbef2c3ff4ae34e8c88379e90148c24
.. _c582fe1: https://github.com/xraysoftmat/kkcalc/commit/c582fe1c951d46ff5ec590421db74f677a1ecf01
.. _c6ab85a: https://github.com/xraysoftmat/kkcalc/commit/c6ab85a91bf8a90270eddedb6c383941aa4c7bf1
.. _d3c819c: https://github.com/xraysoftmat/kkcalc/commit/d3c819c7ee1ba0324498993215222d463720ef5a
.. _e5a0a72: https://github.com/xraysoftmat/kkcalc/commit/e5a0a72a793eb608d0d0e11bb543a56893d60ff9
.. _f2fc6af: https://github.com/xraysoftmat/kkcalc/commit/f2fc6af53096804470225fc09306e05f80da76c9
.. _f73128f: https://github.com/xraysoftmat/kkcalc/commit/f73128fbe0ea53a32a5a94498120f44234efb5ba
.. _f741969: https://github.com/xraysoftmat/kkcalc/commit/f7419694c0b271b041d3cf3c9097cc57643278a2
.. _fd81bcb: https://github.com/xraysoftmat/kkcalc/commit/fd81bcb93adca1342946b2d1be58d14bfdf399c7
.. _ffa1f6e: https://github.com/xraysoftmat/kkcalc/commit/ffa1f6e5c2aa37c8aa265b9af8caf9bd4030d83d


.. _changelog-v0.8.4:

v0.8.4 (2024-06-19)
==========================

* Initial Release ported to Xraysoftmat from benajamin.
