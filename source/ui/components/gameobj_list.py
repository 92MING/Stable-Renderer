from PySide6.QtWidgets import QWidget, QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem\

class GameObjListWindow(QTreeWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setColumnCount(1)
        self.setHeaderLabels(['Name'])
        self.setSortingEnabled(True)
        self.setAlternatingRowColors(True)
        self.setAnimated(True)
        self.setIndentation(10)
        self.setRootIsDecorated(False)
        self.setExpandsOnDoubleClick(False)
        self.setAllColumnsShowFocus(True)
        test_item = self.addItem('test')
        test_item.addChild(QTreeWidgetItem(['test_child']))
        test_item.addChild(QTreeWidgetItem(['test_child2']))
        
    def addItem(self, name):
        item = QTreeWidgetItem(self)
        item.setText(0, name)
        self.addTopLevelItem(item)
        return item


__all__ = ['GameObjListWindow']