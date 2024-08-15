from PyQt6 import QtGui, QtCore, QtWidgets

class Window(QtWidgets.QWidget):
    def __init__(self, rows, columns):
        QtWidgets.QWidget.__init__(self)
        self.table = QtWidgets.QTableWidget(rows, columns, self)
        for column in range(columns):
            for row in range(rows):
                item = QtWidgets.QTableWidgetItem('Text%d' % row)
                if row % 2:
                    item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable |
                                  QtCore.Qt.ItemFlag.ItemIsEnabled)
                    item.setCheckState(QtCore.Qt.CheckState.Unchecked)
                self.table.setItem(row, column, item)
        self.table.itemClicked.connect(self.handleItemClicked)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table)
        self._list = set()
    
    def handleItemClicked(self, item: QtWidgets.QTableWidgetItem):
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            if item not in self._list:
                self._list.add(item)
            print('"%s" Checked' % item.text())
            # self._list.append(item.row())
            print([item.row() for item in self._list])
        else:
            print('"%s" Clicked' % item.text())

if __name__ == '__main__':

    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window(6, 3)
    window.resize(350, 300)
    window.show()
    sys.exit(app.exec())