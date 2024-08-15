"""
Object loader and lister for objects that implement asf_abstract and asp_abstract classes.
Allows the loading of raw data and duplication objects.
"""
from PyQt6 import QtWidgets, QtCore
from kkcalc.models import asf_abstract, asp_abstract

class kk_object_list(QtWidgets.QWidget):
    """
    A widget for loading and listing objects that implement the `asf_abstract` and `asp_abstract` classes.
    
    Each object is represented by a row in a table, 
    with columns for the object name, stoichiometry, class type, and
    a visibility checkbox.
    """
    
    viewSelectionChanged = QtCore.pyqtSignal()
    """A signal emitted when the 'viewing' selection changes, by toggling visibility checkboxes."""
    
    selectedObjectChanged = QtCore.pyqtSignal()
    """A signal emitted when the selected row (object) changes."""
    
    def __init__(self, parent=None, objs: list[type[asf_abstract | asp_abstract]] | None = None):
        super().__init__(parent=parent)
        self.setWindowTitle("kkcalc Object Loader")
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        
        # Setup margins if parent is provided.
        if parent is not None:
            self.setContentsMargins(0,0,0,0)
            self._layout.setContentsMargins(0,0,0,0)
        
        # Create the load data buttons
        self.import_data_btn = QtWidgets.QPushButton("Import Data")
        
        # Create the table
        self.table = QtWidgets.QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["Name", "Stoich.", "Type", "Vis."])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        
        # Change the table properties
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        
        # Create duplicate and delete buttons
        hlayout = QtWidgets.QHBoxLayout()
        self.duplicate_btn = QtWidgets.QPushButton("Duplicate")
        self.delete_btn = QtWidgets.QPushButton("Delete")
        
        # Assign elements to the layout
        self._layout.addWidget(self.import_data_btn)
        self._layout.addWidget(self.table)
        self._layout.addLayout(hlayout)
        hlayout.addWidget(self.duplicate_btn)
        hlayout.addWidget(self.delete_btn)
                
        # Setup the object sets
        self._visible_rows : set[QtWidgets.QTableWidgetItem] = set() #initialize the set
        """A set of visible rows numbers, corresponding to `_objs` keys."""
        self._objs : dict[int, type[asf_abstract | asp_abstract]] = {}
        """A mapping of table row to object."""
                
        # Add objects to the table
        if objs is not None:
            self.add_kk_objs(objs)
        
        # Setup widget properties
        self.duplicate_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
            
        # Setup connections
        self.table.itemClicked.connect(self.itemViewClicked)
        self.import_data_btn.clicked.connect(self.import_data)
        self.table.itemSelectionChanged.connect(self.on_row_select)
        self.duplicate_btn.clicked.connect(self.duplicate)
        self.delete_btn.clicked.connect(self.delete)
        
    def update_kk_obj(self, obj: type[asf_abstract | asp_abstract]) -> None:
        """
        Updates the row matching the object in the table.
        """
        for row, obj_ in self._objs.items():
            if obj_ == obj:
                self.table.item(row, 0).setText(obj.name)
                self.table.item(row, 1).setText(str(obj.stoichiometry))
                self.table.item(row, 2).setText(obj.__class__.__name__)
                return
        
        
    def add_kk_obj(self, obj: type[asf_abstract | asp_abstract]) -> None:
        """
        Adds a new object to the table.
        
        Generates a 4 column table row for the object, and also stores
        the object in the internal mapping.
        
        Parameters
        ----------
        obj : type[asf_abstract | asp_abstract]
            The object to add to the table.
        """
        # Add the entry to the table
        rows = self.table.rowCount()
        self.table.setRowCount(rows + 1)
        self.table.setItem(rows, 0, QtWidgets.QTableWidgetItem(obj.name))
        self.table.setItem(rows, 1, QtWidgets.QTableWidgetItem(str(obj.stoichiometry)))
        obj_class = QtWidgets.QTableWidgetItem(obj.__class__.__name__)
        obj_class.setToolTip(obj.__class__.__doc__)
        self.table.setItem(rows, 2, obj_class)
        checkbox = QtWidgets.QTableWidgetItem()
        checkbox.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable |
                          QtCore.Qt.ItemFlag.ItemIsEnabled)
        checkbox.setCheckState(QtCore.Qt.CheckState.Checked)
        self.table.setItem(rows, 3, checkbox)
        # Check row doesn't already exist in mappings
        if rows in self._objs or rows in self._visible_rows:
            raise ValueError(f"Row {rows} already exists in the object mapping.")
        # Add the object to the mapping
        self._visible_rows.add(rows)
        self._objs[rows] = obj
        
        # Autoscale the table column widths
        self.table.resizeColumnsToContents()
        
        # Emit a signal
        self.viewSelectionChanged.emit()
        return
        
    def add_kk_objs(self, objs: list[type[asf_abstract | asp_abstract]]) -> None:
        """
        Adds multiple objects to the table.

        Parameters
        ----------
        objs : list[type[asf_abstract  |  asp_abstract]]
            A list of objects to add to the table.
        
        See Also
        --------
        add_kk_obj : Adds a single object to the table.
        """
        for obj in objs:
            self.add_kk_obj(obj)
        # self.table.update()
        return
        
    def itemViewClicked(self, item: QtWidgets.QTableWidgetItem):
        # Check if the item belongs to the checkbox column
        if item.column() == 3:
            # Check if the item is checked
            if item.checkState() == QtCore.Qt.CheckState.Checked and item.row() not in self._visible_rows:
                self._visible_rows.add(item.row())
            elif item.row() in self._visible_rows:
                self._visible_rows.remove(item.row())
            self.viewSelectionChanged.emit()
        return
    
    def on_row_select(self):
        self.selectedObjectChanged.emit()
        if self.selected_object is not None:
            self.duplicate_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
        else:
            self.duplicate_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
    
    @property
    def checked_objects(self) -> list[type[asf_abstract | asp_abstract]]:
        return [self._objs[row] for row in self._visible_rows]
    
    @property
    def selected_object(self) -> type[asf_abstract | asp_abstract] | None:
        selected = self.table.selectedItems()
        if len(selected) == 0:
            return None
        return self._objs[selected[0].row()]
    
    def import_data(self):
        raise NotImplementedError("Importing data is not yet implemented.")
        return
    
    def duplicate(self):
        selected = self.selected_object
        if selected is None:
            return
        # Duplicate the object
        copy = selected.copy()
        # Get all names in the table
        names = [self.table.item(i, 0).text() for i in range(self.table.rowCount())]
        copy_name = copy.name + "_copy"
        copy_name_i = copy_name
        i = 0
        while copy_name_i in names:
            copy_name_i = copy_name + str(i)
            i += 1
        # Set the name of the copy
        copy.name = copy_name_i
        self.add_kk_obj(copy)
        
        # Emit view change signal if selected object is visible
        if self.table.selectedItems()[0].row() in self._visible_rows:
            self.viewSelectionChanged.emit()
    
    def delete(self):
        selected = self.selected_object
        print("Objects:", self._objs.keys())
        if selected is None:
            return
        # Delete the row and the object from the mapping
        row = self.table.selectedItems()[0].row()
        print("Deleting ", row, str(selected))
        self._objs.pop(row)
        was_visible:bool = False
        if row in self._visible_rows:
            self._visible_rows.remove(row)
            was_visible = True
            
        # Update the table
        self.table.removeRow(row)
        
        # Adjust the row numbers
        if len(self._objs) > row:
            # Shift each row backwards
            for i in range(row + 1, len(self._objs)+1):
                obj = self._objs.pop(i)
                if (i-1) not in self._objs:
                    self._objs[i-1] = obj
                else:
                    raise ValueError(f"Row {i-1} already exists in the object mapping.")
                if i in self._visible_rows:
                    self._visible_rows.remove(i)
                    self._visible_rows.add(i-1)
                    
        # Select the row prior
        if row > 0:
            self.table.selectRow(row-1)
        elif len(self._objs) > 0:
            self.table.selectRow(0)
        else:
            self.table.clearSelection()
        
        # Signal a change
        if was_visible:
            self.viewSelectionChanged.emit()
            
        print("Objects post delete:", self._objs.keys())
        print("Visible post delete:", self._visible_rows)