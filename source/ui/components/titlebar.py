import os
from PySide6.QtWidgets import (QWidget, QLabel, QMenuBar, QHBoxLayout, QPushButton)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from qtawesome import icon

from common_utils.constants import NAME, VERSION

class TitleBar(QWidget):
    
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedHeight(35)
        self.setContentsMargins(3, 1, 5, 1)
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(4)
        
        self.icon = QLabel()
        _icon_path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'icon_white.png')
        _icon = QPixmap(_icon_path)
        _icon = _icon.scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.icon.setPixmap(_icon)
        self.layout().addWidget(self.icon, 0, Qt.AlignmentFlag.AlignLeft)
        
        self.menu = QMenuBar()
        self.layout().addWidget(self.menu, 0, Qt.AlignmentFlag.AlignLeft)
        self.menu.addMenu('File')
        self.menu.addMenu('Edit')
        self.menu.addMenu('Configs')
        self.menu.addMenu('Help')
        
        self.title = QLabel(f"{NAME} v{VERSION}")
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

__all__ = ['TitleBar']