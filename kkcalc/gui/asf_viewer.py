"""
File for Atomic scattering viewer GUI, build in PyQt6
"""

from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from enum import Enum
import matplotlib.pyplot as plt
from typing import Literal
from kkcalc.models import asf_abstract, asp_abstract, asf, asp, asf_im, asf_re, asf_complex, asp_im, asp_re, asp_complex
from kkcalc.asf_database import asf_db, asp_db, asp_db_extended
from kkcalc.models.factors import KK_Datatype
from kkcalc.stoich import kk_stoichiometry
import numpy.typing as npt
import numpy as np

# No way to attach these docs to the enum, so we will just define them here
GRAPH_TYPE_DOCS = [
    "Show the real and imaginary parts of the scattering factor separately",
    "Show the real and imaginary parts of the scattering factor together",
    "Show the absolute value and phase of the scattering factor separately",
    "Show the absolute value and phase of the scattering factor together"
]

class GraphType(Enum):
    """Graphing types for the atomic scattering factor viewer"""
    
    RE_IM_SEPARATE = 0
    """Show the real and imaginary parts of the scattering factor separately"""
    RE_IM_OVERLAY = 1
    """Show the real and imaginary parts of the scattering factor together"""
    ABS_PHASE_SEPARATE = 2
    """Show the absolute value and phase of the scattering factor separately"""
    ABS_PHASE_OVERLAY = 3
    """Show the absolute value and phase of the scattering factor together"""

class asf_viewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        # Set the margins to 0 if a parent is provided
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._layout.setContentsMargins(0, 0, 0, 0)
        
        # Create settings bars
        self.top_settings_bar = QtWidgets.QHBoxLayout()
        self.bottom_settings_bar = QtWidgets.QHBoxLayout()
        
        # Create an option to switch graphing type
        self.graph_type_label = QtWidgets.QLabel("Repr:")
        self.graph_type = QtWidgets.QComboBox()
        self.graph_type.addItems([x.name.lower().capitalize().replace("_", " ") for x in GraphType])
        # Setup hints
        [self.graph_type.setItemData( #If __docs__ accessible on enums in future use that. 
            i, GRAPH_TYPE_DOCS[i], QtCore.Qt.ItemDataRole.ToolTipRole 
         ) for i, x in enumerate(GraphType)]
        
        # Create an option to switch x scale
        self.x_scale_label = QtWidgets.QLabel("X Scale:")
        self.x_scale_combo = QtWidgets.QComboBox()
        self.x_scale_combo.addItems(["Linear", "Log"])
        
        # Create an option to switch the y-datatype.
        self.y_datatype_label = QtWidgets.QLabel("Y Datatype:")
        self.y_datatype_combo = QtWidgets.QComboBox()
        self.y_datatype_combo.addItems([x.name for x in KK_Datatype if x != KK_Datatype.UNDEFINED])
        
        # Create a matplotlib widget to show real and imaginary parts
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.navbar = NavigationToolbar(self.canvas, self)
        
        # Create a combobox to snap the x-axis to a dataset.
        self.snap_x_label = QtWidgets.QLabel("Snap X:")
        self.snap_x_combo = QtWidgets.QComboBox()
        
        # Create a combobox to normalise the y-axis to a dataset.
        self.norm_y_label = QtWidgets.QLabel("Norm Y:")
        self.norm_y_combo = QtWidgets.QComboBox()
        
        # Assign elements to the settings bar layouts
        self.top_settings_bar.addWidget(self.navbar)
        self.top_settings_bar.addWidget(self.x_scale_label)
        self.top_settings_bar.addWidget(self.x_scale_combo)
        self.top_settings_bar.addWidget(self.y_datatype_label)
        self.top_settings_bar.addWidget(self.y_datatype_combo)
        self.top_settings_bar.addWidget(self.graph_type_label)
        self.top_settings_bar.addWidget(self.graph_type)
        self.bottom_settings_bar.addWidget(self.snap_x_label)
        self.bottom_settings_bar.addWidget(self.snap_x_combo)
        self.bottom_settings_bar.addWidget(self.norm_y_label)
        self.bottom_settings_bar.addWidget(self.norm_y_combo)
        # Add elements to the layout.
        self._layout.addLayout(self.top_settings_bar)
        self._layout.addWidget(self.canvas)
        self._layout.addLayout(self.bottom_settings_bar)
        
        # Setup the plot
        self._graph_style: GraphType = GraphType.RE_IM_SEPARATE
        """The current graphing style"""
        
        # Setup the scattering object list
        self._scattering_objects: list[type[asp_abstract] | type[asf_abstract]] = []
        
        # Connect the graph type change to the switch_graph_style function
        self.graph_type.currentIndexChanged.connect(lambda x: self.reset_graph())
        self.x_scale_combo.currentIndexChanged.connect(lambda x: self.reset_graph())
        self.y_datatype_combo.currentIndexChanged.connect(lambda x: self.reset_graph())
        self.snap_x_combo.currentIndexChanged.connect(self.snap_x)
        self.norm_y_combo.currentIndexChanged.connect(self.norm_y)
        self.reset_graph()
        
    @property
    def scattering_objects(self) -> list[type[asp_abstract] | type[asf_abstract]]:
        """
        The current scattering objects

        Returns
        -------
        list[type[asp_abstract] | type[asf_abstract]]
            The current scattering objects
        """
        return self._scattering_objects    
    
    @scattering_objects.setter
    def scattering_objects(self, objs: list[type[asp_abstract] | type[asf_abstract]]):
        """
        The current scattering objects

        Parameters
        ----------
        list[type[asp_abstract] | type[asf_abstract]]
            The current scattering objects
        """
        # Check inputs
        for obj in objs:
            if not isinstance(obj, asf_abstract) and not isinstance(obj, asp_abstract):
                raise ValueError("Invalid scattering object")
        self._scattering_objects = objs
        # Update the snap x combo box with the new objects
        self.snap_x_combo.blockSignals(True)
        self.snap_x_combo.clear()
        self.snap_x_combo.addItem("") # always add a blank option, used to reset to full x-axis.
        self.snap_x_combo.addItems([f"{type(obj)}: {obj.name}" for obj in self.scattering_objects])
        self.snap_x_combo.blockSignals(False)
        # Update the norm y combo box with the new objects
        self.norm_y_combo.blockSignals(True)
        self.norm_y_combo.clear()
        self.norm_y_combo.addItem("") # always add a blank option, used to reset to full y-axis.
        self.norm_y_combo.addItems([f"{type(obj)}: {obj.name}" for obj in self.scattering_objects])
        self.norm_y_combo.blockSignals(False)
        # Reset the graph
        self.reset_graph()
        
    @property
    def graph_style(self) -> GraphType:
        """
        The current graphing style

        Return
        ------
        GraphType
            The current graphing style
        """
        return GraphType(self.graph_type.currentIndex())
    
    @property
    def x_scale(self) -> Literal["linear", "log"]:
        """
        The current x scale

        Return
        ------        
        Literal["linear", "log"]
            The current x scale
        """
        return self.x_scale_combo.currentText().lower()
    
    @property
    def y_datatype(self) -> KK_Datatype:
        """
        The current y datatype

        Parameters
        ----------
        KK_Datatype
            The current y datatype
        
        """
        return KK_Datatype(self.y_datatype_combo.currentIndex() + 1) #excluded UNDEFINED so need to add 1.
    
    @property
    def y_normalisation_scales(self) -> list[tuple[float, float] | None]:
        """
        Returns the normalisation scales for the y-axes
        if `snap_y_combo` is not set to None.

        Return
        ------
        list[tuple[float, float] | None]
            A list of normalisation scales (min, max) for each y-axis.
        """
        idx = self.norm_y_combo.currentIndex()
        if idx == 0:
            return [None, None]
        else:
            # Get obj
            obj = self.scattering_objects[idx - 1]
            y_scales = []
            for ax in [self.ax1, self.ax2]:
                # Get the displayed x-limits
                xlim = self.ax1.get_xlim()
                
                # Get the y-range within the x-range
                y_min, y_max = None, None
                for line in ax.get_lines():
                    x,y = line.get_data()
                    mask = (x >= xlim[0]) & (x <= xlim[1])
                    if mask.any():
                        y: npt.NDArray = y[mask]
                        # Update the y-range within the x-domain
                        y_min = y.min() if y_min is None else min(y_min, y.min())
                        y_max = y.max() if y_max is None else max(y_max, y.max())
                if y_min is not None and y_max is not None:
                    y_scales.append((y_min, y_max))
                else:
                    y_scales.append(None)
            return y_scales
    
    def snap_x(self, index: int):
        """
        Snap the x-axis to the dataset at the given index

        Parameters
        ----------
        index : int
            The index of the dataset to snap to.
        """
        if index == 0:
            # Restore the x-axis to auto
            self.ax1.set_xlim(auto=True)
            self.ax2.set_xlim(auto=True)
            self.ax1.set_xmargin(0.05) # reset margin to default
            self.ax2.set_xmargin(0.05) # reset margin to default 
        else:
            index -= 1 # Subtract 1 to account for the blank option
            # Get the energies and maximum/minimum of the dataset
            energies: npt.NDArray = self.scattering_objects[index].energies
            mn, mx = energies.min(), energies.max()
            # Use a margin:
            if self.x_scale == "linear":
                mn, mx = mn - 0.05 * (mx - mn), mx + 0.05 * (mx - mn)
            else: #log
                mn, mx = mn / 1.05, mx * 1.05
            for ax in [self.ax1, self.ax2]:
                ax: plt.Axes
                # Set the x limits
                ax.set_xlim(mn, mx)
                # Get the y-range within the x-range
                y_min, y_max = None, None # initialize
                # Loop through the lines
                for line in ax.get_lines():
                    x,y = line.get_data()
                    mask = (x >= mn) & (x <= mx)
                    if mask.any():
                        y: npt.NDArray = y[mask]
                        # Update the y-range within the x-domain
                        y_min = y.min() if y_min is None else min(y_min, y.min())
                        y_max = y.max() if y_max is None else max(y_max, y.max())
                # Add margins to the y-min and y-max
                y_min, y_max = y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)
                ax.set_ylim(y_min, y_max)
                
        # Update the canvas
        self.canvas.draw()
        
    def reset_graph(self):
        """
        Used to update the graph style and x scale attributes.
        """
        # Get graphing attributes
        graph_style = self.graph_style
        x_scale = self.x_scale
        y_datatype = self.y_datatype
        ax1_norm_scale, ax2_norm_scale = self.y_normalisation_scales
        # Clear the existing graph
        self.figure.clear()
        # Choose the y labels and title
        match y_datatype:
            case KK_Datatype.ASF:
                title = "Atomic Scattering Factors (f = f0 + f1 + i*f2)"
                yl1 = "f1"
                yl2 = "f2"
            case KK_Datatype.BETA:
                title = "Refractive Index Components (n = 1 - delta - i*beta)"
                yl1 = "Delta"
                yl2 = "Beta"
            case KK_Datatype.NEXAFS | KK_Datatype.XANES | KK_Datatype.PHOTOABSORPTION:
                title = "Absorption intensities (A.U.)"
                yl1 = "Intensity"
                yl2 = "Intensity"
            case _:
                raise ValueError("Invalid y datatype")    
        self.figure.suptitle(title)
        # Add the axes, and x labels
        match graph_style:
            case GraphType.RE_IM_SEPARATE | GraphType.ABS_PHASE_SEPARATE:
                self.ax1 = self.figure.add_subplot(121)
                self.ax2 = self.figure.add_subplot(122)
                self.ax1.set_xlabel("Energy (eV)")
                self.ax2.set_xlabel("Energy (eV)")
                self.ax1.set_xscale(x_scale)
                self.ax2.set_xscale(x_scale)
            case GraphType.RE_IM_OVERLAY | GraphType.ABS_PHASE_OVERLAY:
                self.ax1 = self.figure.add_subplot(111)
                self.ax2 = self.ax1.twinx()
                self.ax1.set_xlabel("Energy (eV)")
                self.ax1.set_xscale(x_scale)
        # Add the y labels
        match graph_style:
            case GraphType.RE_IM_SEPARATE | GraphType.RE_IM_OVERLAY:
                self.ax1.set_ylabel(yl1)
                self.ax2.set_ylabel(yl2)
            case GraphType.ABS_PHASE_SEPARATE | GraphType.ABS_PHASE_OVERLAY:
                self.ax1.set_ylabel("Magnitude")
                self.ax2.set_ylabel("Phase")
    
        # Plot the data
        for obj in self.scattering_objects:
            # If a polynomial, convert to asf to display.
            if isinstance(obj, asp_abstract):
                obj: asp_abstract
                obj = obj.to_asf()
            # Plot data of an asf object.
            if isinstance(obj, asf_abstract):
                # Get the appropriate y-data
                if y_datatype == KK_Datatype.ASF:
                    y = obj.factors
                elif y_datatype == KK_Datatype.BETA:
                    if obj.can_calc_beta:
                        y = obj.betas
                    else:
                        # Skip this object if it cannot calculate beta
                        continue
                elif y_datatype == KK_Datatype.NEXAFS:
                    y = obj.NEXAFS
                else:
                    raise ValueError(f"Invalid y datatype {y_datatype}")
                
                # Cannot plot re/im on amp/phase graphs unless the object is a complex asf.
                if isinstance(obj, asf_re) and graph_style in [GraphType.RE_IM_SEPARATE, GraphType.RE_IM_OVERLAY]:
                    obj: asf_re
                    self.ax1.plot(obj.energies, y, label=obj.name)
                elif isinstance(obj, asf_im) and graph_style in [GraphType.RE_IM_SEPARATE, GraphType.RE_IM_OVERLAY]:
                    obj: asf_im
                    self.ax2.plot(obj.energies, y, label=obj.name)
                elif isinstance(obj, asf_complex):
                    obj: asf_complex
                    if graph_style in [GraphType.ABS_PHASE_SEPARATE, GraphType.ABS_PHASE_OVERLAY]:
                        self.ax1.plot(obj.energies, obj.abs, label=obj.name)
                        self.ax2.plot(obj.energies, obj.phase, label=obj.name)
                    elif graph_style in [GraphType.RE_IM_SEPARATE, GraphType.RE_IM_OVERLAY]:
                        self.ax1.plot(obj.energies, obj.re, label=obj.name)
                        self.ax2.plot(obj.energies, obj.im, label=obj.name)
                else:
                    raise ValueError(f"Invalid scattering object {obj}")
            else:
                raise ValueError("Invalid scattering object")
        # Add the legend
        self.ax1.legend()
        self.ax2.legend()
        # Draw the plot
        self.figure.tight_layout()
        self.canvas.draw()
        return
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = asf_viewer()
    
    # Import some example data
    import numpy as np
    import os
    PS = "C10H8"
    PS_stoich = kk_stoichiometry(PS)
    
    data_path = os.path.join(os.path.dirname(__file__), "../../examples/data/PS_004_-dc.txt")
    data = np.genfromtxt(data_path, skip_header=4)
    
    polystyrene_asf_im = asf_im.from_NEXAFS(data[:, 0], data[:, 1], stoich=PS_stoich, name="Polystyrene")
    polystyrene_asp_im = asp_db_extended.from_asf_im(data_asf = polystyrene_asf_im, data_stoich=PS_stoich, )
    polystyrene_asp_im2 = asp_db_extended.from_asf_im(data_asf = polystyrene_asf_im, data_stoich=PS_stoich, merge_domain=[280, 320])
    polystyrene_asf_re = polystyrene_asp_im.kk_transform()
    polystyrene_asf_re2 = polystyrene_asp_im.kk_transform()
    
    ui.scattering_objects = [polystyrene_asf_im, polystyrene_asp_im2, polystyrene_asf_re, polystyrene_asf_re]
    
    ui.show()
    sys.exit(app.exec())