from PySide6.QtWidgets import QApplication, QFileSystemModel, QTreeView, QListView, QSplitter, QVBoxLayout, QWidget
from PySide6.QtCore import QDir
from PySide6 import QtWidgets, QtCore

class FileExplorer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('Simple File Explorer')

        # 文件系统模型
        model = QFileSystemModel()
        model.setRootPath(QDir.currentPath())

        # 树状视图，展示文件夹结构
        tree = QTreeView()
        tree.setModel(model)
        tree.setRootIndex(model.index(QDir.currentPath()))
        tree.setColumnWidth(0, 250)
        tree.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding))
        
        # 列表视图，展示文件夹内容
        list_view = QListView()
        list_view.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding))
        list_view.setModel(model)
        list_view.setRootIndex(model.index(QDir.currentPath()))
        list_view.setViewMode(QtWidgets.QListWidget.IconMode)
        list_view.setGridSize(QtCore.QSize(50,50))
        list_view.setIconSize(QtCore.QSize(40,40))
        list_view.setResizeMode(QtWidgets.QListWidget.Adjust)
        list_view.setMovement(QtWidgets.QListWidget.Static)
        list_view.setWrapping(True)
        list_view.setContentsMargins(15, 15, 15, 15)
        
        
        def on_path_click():
            if (tree.currentIndex().isValid() and model.isDir(tree.currentIndex())):
                list_view.setRootIndex(tree.currentIndex())
        
        tree.selectionModel().currentChanged.connect(on_path_click)
        splitter = QSplitter()
        splitter.addWidget(tree)
        splitter.addWidget(list_view)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 6)

        layout = QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)




__all__ = ['FileExplorer']