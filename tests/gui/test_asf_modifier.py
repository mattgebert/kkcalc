"""
Tests for the `kk_object_modifier` GUI widget's fix-distortions controls.

Covers both `fix_distortions_method` options ("grad_min" and "prepost_fit") end-to-end via the
widget's `extend_obj` method, using the same Polystyrene example data as `kkcalc2.__main__`.
"""

import io
import pkgutil

import numpy as np
import pytest

from kkcalc2.models import asf_im
from kkcalc2.stoich import stoichiometry as kk_stoich

from ..test_stoich import basic_stoichs as bs


@pytest.fixture
def PS_asf_dataset() -> asf_im:
    """Imaginary ASF dataset built from the example Polystyrene NEXAFS data."""
    PS_datafile = pkgutil.get_data("kkcalc2", "data/PS_004_-dc.txt")
    PS_data = np.genfromtxt(io.BytesIO(PS_datafile), skip_header=4)
    PS_energies, PS_NEXAFS = PS_data[:, 0], PS_data[:, 1]
    return asf_im.from_NEXAFS(
        PS_energies,
        PS_NEXAFS,
        name="Polystyrene",
        stoichiometry=kk_stoich(bs.POLYMER_PS),
        density=1.05,
    )


@pytest.fixture
def modifier(qapp, PS_asf_dataset: asf_im):
    """A `kk_object_modifier` widget pre-loaded with the example Polystyrene ASF dataset."""
    from kkcalc2.gui.asf_modifier import kk_object_modifier

    widget = kk_object_modifier()
    widget.object = PS_asf_dataset
    yield widget
    widget.deleteLater()


class TestFixDistortionsUI:
    """Tests the fix-distortions method selector UI and its effect on `extend_obj`."""

    def test_default_method_is_grad_min(self, modifier) -> None:
        """The combobox defaults to 'grad_min', with distortion fixing disabled by default."""
        assert modifier.fix_distortions_method_combo.currentText() == "grad_min"
        assert not modifier.fix_distortions_checkbox.isChecked()
        assert modifier.fix_distortions_kwargs() == {"fix_distortions": False}

    def test_tooltips_differ_between_methods(self, modifier) -> None:
        """The combobox tooltip is updated to describe the currently selected method."""
        modifier.fix_distortions_method_combo.setCurrentText("grad_min")
        modifier.update_fix_distortions_UI()
        grad_min_tooltip = modifier.fix_distortions_method_combo.toolTip()

        modifier.fix_distortions_method_combo.setCurrentText("prepost_fit")
        modifier.update_fix_distortions_UI()
        prepost_fit_tooltip = modifier.fix_distortions_method_combo.toolTip()

        assert grad_min_tooltip != prepost_fit_tooltip
        assert "grad_min" in grad_min_tooltip
        assert "prepost_fit" in prepost_fit_tooltip

    def test_predomain_postdomain_fields_enabled_only_for_prepost_fit(
        self, modifier
    ) -> None:
        """The pre/post domain fields are only enabled for the 'prepost_fit' method."""
        fields = (
            modifier.fix_predomain_lb_edit,
            modifier.fix_predomain_ub_edit,
            modifier.fix_postdomain_lb_edit,
            modifier.fix_postdomain_ub_edit,
        )

        modifier.fix_distortions_checkbox.setChecked(True)
        modifier.fix_distortions_method_combo.setCurrentText("grad_min")
        modifier.update_fix_distortions_UI()
        assert all(not field.isEnabled() for field in fields)

        modifier.fix_distortions_method_combo.setCurrentText("prepost_fit")
        modifier.update_fix_distortions_UI()
        assert all(field.isEnabled() for field in fields)

        modifier.fix_distortions_checkbox.setChecked(False)
        modifier.update_fix_distortions_UI()
        assert all(not field.isEnabled() for field in fields)

    @pytest.mark.parametrize(
        "method, predomain, postdomain",
        [
            ("grad_min", None, None),
            ("prepost_fit", ("277.0", "283.0"), ("350.0", "385.0")),
        ],
    )
    def test_extend_obj_works_for_both_methods(
        self,
        modifier,
        method: str,
        predomain: tuple[str, str] | None,
        postdomain: tuple[str, str] | None,
    ) -> None:
        """`extend_obj` successfully extends the example data using either distortion method."""
        modifier.fix_distortions_checkbox.setChecked(True)
        modifier.fix_distortions_method_combo.setCurrentText(method)
        modifier.update_fix_distortions_UI()
        if predomain is not None:
            modifier.fix_predomain_lb_edit.setText(predomain[0])
            modifier.fix_predomain_ub_edit.setText(predomain[1])
        if postdomain is not None:
            modifier.fix_postdomain_lb_edit.setText(postdomain[0])
            modifier.fix_postdomain_ub_edit.setText(postdomain[1])

        extended = modifier.extend_obj()

        assert extended is not None
        assert extended.is_extended
        assert extended.fix_distortions_method == method
        assert np.all(np.isfinite(extended.asf))
