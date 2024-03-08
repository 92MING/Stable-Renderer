import sys, os
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    __package__ = 'ui'


from PySide6.QtWidgets import (QWidget, QApplication, QMainWindow, QVBoxLayout, QDockWidget, QTextEdit, QTabWidget,
                               QLabel, QMenuBar, QHBoxLayout, QPushButton)
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QPixmap
from qt_material import apply_stylesheet, list_themes
from qtawesome import icon
from .components import *

class TitleBar(QWidget):
    
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedHeight(35)
        self.setContentsMargins(5, 1, 5, 1)
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(4)
        
        self.icon = QLabel()
        _icon_path = os.path.join(os.path.dirname(__file__), 'resources', 'icon_white.png')
        _icon = QPixmap(_icon_path)
        _icon = _icon.scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.icon.setPixmap(_icon)
        self.layout().addWidget(self.icon, 0, Qt.AlignmentFlag.AlignLeft)
        
        self.menu = QMenuBar()
        self.layout().addWidget(self.menu, 0, Qt.AlignmentFlag.AlignLeft)
        self.menu.addMenu('File')
        self.menu.addMenu('Edit')
        self.menu.addMenu('View')
        self.menu.addMenu('Help')
        
        self.title = QLabel("Test")
        self.layout().addWidget(self.title, 1, Qt.AlignmentFlag.AlignCenter)

        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonLayout.setSpacing(0)
        self.layout().addLayout(self.buttonLayout, Qt.AlignmentFlag.AlignRight)
        
        self.closeButton = QPushButton(icon=icon('fa5s.times', color='white'))
        self.closeButton.setFixedSize(25, 25)
        self.closeButton.clicked.connect(self.parent().close)
        
        self.maximizeButton = QPushButton(icon=icon('fa5s.window-maximize', color='white'))
        def maximize(btn):
            if self.parent().isMaximized():
                self.parent().showNormal()
                btn.setIcon(icon('fa5s.window-maximize', color='white'))
            else:
                self.parent().showMaximized()
                btn.setIcon(icon('fa5s.window-restore', color='white'))
        self.maximizeButton.clicked.connect(lambda: maximize(self.maximizeButton))
        self.maximizeButton.setFixedSize(25, 25)
        
        self.minimizeButton = QPushButton(icon=icon('fa5s.window-minimize', color='white'))
        self.minimizeButton.clicked.connect(self.parent().showMinimized)
        self.minimizeButton.setFixedSize(25, 25)
        
        self.buttonLayout.addWidget(self.minimizeButton)
        self.buttonLayout.addWidget(self.maximizeButton)
        self.buttonLayout.addWidget(self.closeButton)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.parent().dragPos = event.globalPos()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.parent().move(self.parent().pos() + event.globalPos() - self.parent().dragPos)
            self.parent().dragPos = event.globalPos()
            event.accept()
            
    def mouseDoubleClickEvent(self, event):
        if self.parent().isMaximized():
            self.parent().showNormal()
        else:
            self.parent().showMaximized()
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setMinimumSize(800, 600)
        
        self.setMenuWidget(TitleBar(self))
        
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)
        self.setTabShape(QTabWidget.TabShape.Rounded)

        self.left_dock = QDockWidget("Hierarchy", self)
        self.left_dock.setWidget(GameObjListWindow(self))
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.left_dock)
        self.left_dock.setFeatures(~QDockWidget.DockWidgetFeature.DockWidgetClosable & ~QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        
        self.right_dock = QDockWidget("Inspector", self)
        self.right_dock.setWidget(Inspector(self))
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_dock)
        self.right_dock.setFeatures(~QDockWidget.DockWidgetFeature.DockWidgetClosable & ~QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        
        self.gamePreview = GamePreview(self)
        self.pipeline_editor = PipelineEditor(self)
        
        self.center_tab = QTabWidget()
        self.center_tab.addTab(self.gamePreview, "Preview")
        self.center_tab.addTab(self.pipeline_editor, "Pipeline")
        self.setCentralWidget(self.center_tab)
        
        self.bottom_dock = QDockWidget("", self)
        self.bottom_widget = FileExplorer(self)
        self.bottom_dock.setWidget(self.bottom_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.bottom_dock)
        self.bottom_dock.setFeatures(~QDockWidget.DockWidgetFeature.DockWidgetClosable & ~QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        
        self.showMaximized()
        
        width, height = self.width(), self.height()
        self.resizeDocks([self.left_dock, self.right_dock], [width * 0.3, width * 0.3], Qt.Orientation.Horizontal)
        self.resizeDocks([self.bottom_dock], [height * 0.4], Qt.Orientation.Vertical)
        

if __name__ == '__main__':
    os.environ['QSG_RHI_BACKEND']='opengl'
    app = QApplication([])
    
    extra = {
        'density_scale': '-2',
    }
    apply_stylesheet(app, theme='dark_teal.xml', extra=extra)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
