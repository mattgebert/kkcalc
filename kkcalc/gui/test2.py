# pip install openpyxl, pandas, PyQt5... (if you don't use conda)
import sys
import pandas as pd
from PyQt5.QtCore import QSize, Qt, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QTableWidget, QTableWidgetItem,
                             QFileDialog, QVBoxLayout, QWidget, QDialog, QLabel, QLineEdit, QComboBox, QCheckBox, QHBoxLayout)

# Custom TableWidget class
class TableWidget(QTableWidget):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.load_df(df)
    
    def init_table(self, selected_columns = []):
        
        nRows = len(self.df.index)
        nColumns = len(selected_columns) or len(self.df.columns)
        self.setRowCount(nRows)
        self.setColumnCount(nColumns)

        # Display an empty table
        if self.df.empty:
            self.clearContents()
            return

        self.setHorizontalHeaderLabels(selected_columns or self.df.columns)
        self.setVerticalHeaderLabels(self.df.index.astype(str))

        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                item = QTableWidgetItem(str(self.df.iat[row, col]))
                self.setItem(row, col, item)
        # Enable sorting on the table
        self.setSortingEnabled(True)
        # Enable column moving by drag and drop
        self.horizontalHeader().setSectionsMovable(True)
    
    def load_df(self, df, selected_columns = []):
        self.df = df
        self.init_table(selected_columns)

# Filter Dialog class
class FilterDialog(QDialog):
    filter_applied = pyqtSignal(str, str)

    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Data")
        self.setGeometry(700, 100, 350, 100)

        self.column_label = QLabel("Column:")
        self.column_combobox = QComboBox()
        self.column_combobox.addItems(columns)

        self.value_label = QLabel("Value:")
        self.value_lineedit = QLineEdit()

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_filter)

        layout = QVBoxLayout()
        layout.addWidget(self.column_label)
        layout.addWidget(self.column_combobox)
        layout.addWidget(self.value_label)
        layout.addWidget(self.value_lineedit)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.apply_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def apply_filter(self):
        column = self.column_combobox.currentText()
        value = self.value_lineedit.text()
        self.filter_applied.emit(column, value)
        self.accept()

# Main Window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_columns = []
        self.init_columns = []
        
    def initUI(self):
        self.setWindowTitle("Upload a CSV or Excel file")
        self.setGeometry(200, 200, 600, 400)
        
        self.df = pd.DataFrame()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.open_file)
        self.layout.addWidget(self.browse_button)

        self.header_checkbox = QCheckBox("CSV has header")
        self.header_checkbox.setChecked(True)
        self.layout.addWidget(self.header_checkbox)

        self.table_widget = TableWidget(self.df)
        self.layout.addWidget(self.table_widget)
        
        self.row_count_label = QLabel(f"Number of rows: {self.get_row_count()}")
        self.layout.addWidget(self.row_count_label)

        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(self.show_filter_dialog)
        self.filter_button.setEnabled(False)
        self.layout.addWidget(self.filter_button)
        
        self.remove_filter_button = QPushButton("Remove Filter", self)
        self.remove_filter_button.clicked.connect(self.remove_filter)
        self.remove_filter_button.setEnabled(False)
        self.layout.addWidget(self.remove_filter_button)
        
        # Add the button to show the column selection dialog
        self.column_select_btn = QPushButton("Select Columns", self)
        self.column_select_btn.setEnabled(False)
        self.column_select_btn.clicked.connect(self.show_column_selection_dialog)
        self.layout.addWidget(self.column_select_btn)
#        self.column_select_btn.move(100, 0)

#        self.menu_bar = self.menuBar()
#        self.file_menu = self.menu_bar.addMenu("File")
#        self.open_action = self.file_menu.addAction("Open")
#        self.open_action.triggered.connect(self.import_csv)

    def updateUI(self):
        self.update_row_count_label()
        self.column_select_btn.setEnabled(self.get_row_count() > 0)
        self.filter_button.setEnabled(self.get_row_count() > 0)
        

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All files (*.*)", options=options)
        # check if the file exist
        if file_name:
            if file_name.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_name, sheet_name=0)
            else:
                header_option = 'infer' if self.header_checkbox.isChecked() else None
                self.df = pd.read_csv(file_name)
            self.table_widget.load_df(self.df)
            self.updateUI()
            

#    def import_csv(self):
#        file_path, _ = QFileDialog.getOpenFileName(self, 'CSV File', '', 'CSV Files (*.csv)')
#        if file_path:
#            header_option = 'infer' if self.header_checkbox.isChecked() else None
#            self.df = pd.read_csv(file_path, header=header_option)
#            self.table_widget.df = self.df
#            self.table_widget.init_table()
#            self.update_row_count_label()
    
    def show_column_selection_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Columns")
        layout = QVBoxLayout()

        checkboxes = []
        if len(self.init_columns) == 0:
            self.init_columns = self.table_widget.df.columns
        
        for col in self.init_columns:
            checkbox = QCheckBox(col)
            if len(self.selected_columns) > 0:
                checkbox.setChecked(col in self.selected_columns)
            else:
                checkbox.setChecked(True)
            layout.addWidget(checkbox)
            checkboxes.append(checkbox)

        button_box = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")

        apply_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)

        button_box.addWidget(apply_btn)
        button_box.addWidget(cancel_btn)
        layout.addLayout(button_box)

        dialog.setLayout(layout)

        result = dialog.exec()

        if result == QDialog.Accepted:
            self.selected_columns = [checkbox.text() for checkbox in checkboxes if checkbox.isChecked()]
            self.table_widget.load_df(self.table_widget.df, self.selected_columns)
    
    def show_filter_dialog(self):
        if not self.df.empty:
            filter_dialog = FilterDialog(self.selected_columns or self.df.columns, self)
            filter_dialog.filter_applied.connect(self.apply_filter)
            filter_dialog.exec()

    @pyqtSlot(str, str)
    def apply_filter(self, column, value):
        filtered_df = self.df[self.df[column] == value]
        self.table_widget.df = filtered_df
        self.table_widget.init_table(self.selected_columns)
        self.update_row_count_label()
        self.remove_filter_button.setEnabled(True)
        
    def remove_filter(self):
        self.table_widget.df = self.df
        self.table_widget.init_table(self.selected_columns)
        self.update_row_count_label()
        self.remove_filter_button.setEnabled(False)

    def get_row_count(self):
        return len(self.table_widget.df.index)
    
    def update_row_count_label(self):
        row_count = self.get_row_count()
        self.row_count_label.setText(f"Number of rows: {row_count}")

#    def show_filter_dialog(self):
#        if not self.df.empty:
#            filter_dialog = FilterDialog(self.df.columns, self)
#            if filter_dialog.exec() == QDialog.Accepted:
#                column = filter_dialog.column_combobox.currentText()
#                value = filter_dialog.value_lineedit.text()
#                filtered_df = self.df[self.df[column] == value]
#                self.table_widget.df = filtered_df
#                self.table_widget.init_table()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()

