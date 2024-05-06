import sys, os
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    __package__ = 'ui'

import time
from typing import TYPE_CHECKING
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (QDialog, QWidget, QApplication, QVBoxLayout, QMainWindow, QHBoxLayout, QDockWidget, QTabWidget, 
                               QLabel)
from PySide6.QtCore import Qt, QSize
from qt_material import apply_stylesheet
from common_utils.decorators import singleton
from common_utils.constants import NAME, VERSION
from common_utils.debug_utils import LogEvent
from .components import *

if TYPE_CHECKING:
    from comfyUI.server import PromptServer
    from comfyUI.execution import PromptExecutor


class LoadingWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setLayout(QHBoxLayout())
        self.setFixedSize(600, 300)
        self.app = QApplication.instance()
        screen_size: QSize = self.app.screens()[0].size()
        self.setGeometry((screen_size.width() - 600) // 2, 
                         (screen_size.height() - 400) // 2, 
                         600, 220)
        
        self.icon = QLabel()
        _icon_path = os.path.join(os.path.dirname(__file__), 'resources', 'icon_white.png')
        _icon = QPixmap(_icon_path)
        _icon = _icon.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.icon.setPixmap(_icon)
        self.layout().addWidget(self.icon, 0, Qt.AlignmentFlag.AlignLeft)
        
        self.title = QLabel(f'{NAME} - v{VERSION}')
        self.title.setStyleSheet('font-size: 20px; font-weight: bold; color: white;')
        self.logging_title = QLabel("Loading...")
        self.logging_title.setStyleSheet('font-size: 15px; color: white;')
        self.logging_box = QLabel()
        self.logging_box.setWordWrap(True)
        self.logging_text = ""
        self.MAX_LOGGING_COUNT = 512
        LogEvent.addListener(self.add_log_listener)
        self.log_window = QWidget()
        self.log_window.setLayout(QVBoxLayout())
        self.log_window.layout().addWidget(self.title)
        self.log_window.layout().addWidget(self.logging_title)
        self.log_window.layout().addWidget(self.logging_box, 1)
        self.layout().addWidget(self.log_window, 1, Qt.AlignmentFlag.AlignCenter)
        
        self.show()
        self.raise_()
        
    def add_log_listener(self, record):
        self.add_log(record.msg)
        
    def add_log(self, msg: str):
        self.logging_text += msg + '\n'
        if len(self.logging_text) > self.MAX_LOGGING_COUNT:
            self.logging_text = self.logging_text[-self.MAX_LOGGING_COUNT:]
        self.logging_box.setText(self.logging_text)
        self.app.processEvents()
        
    def set_log_title(self, title: str):
        self.logging_title.setText(title)
        self.app.processEvents()
    
    def closeEvent(self, event: QCloseEvent) -> None:
        LogEvent.removeListener(self.add_log_listener)
        return super().closeEvent(event)

def _start_comfy():
    from comfyUI.main import run
    return run(editor_mode=True)

@singleton(cross_module_singleton=True)
class MainWindow(QMainWindow):
    
    prompt_executor: 'PromptExecutor'
    prompt_server: "PromptServer"
    
    def __init__(self, loading_window):
        super().__init__()
        
        self.prompt_executor, self._comfy_stopper = _start_comfy()   # type: ignore
        from comfyUI.server import PromptServer
        self.prompt_server: PromptServer = PromptServer.instance # type: ignore
        
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setMinimumSize(800, 600)
        
        self.title_bar = TitleBar(self)
        self.setMenuWidget(self.title_bar)
        
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
        
        timeout = 0.0
        while timeout< 12.0 and not hasattr(self.prompt_server, 'port'):
            time.sleep(0.5)
            timeout += 0.5
        if timeout >= 12.0:
            raise Exception("ComfyUI Prompt Server Initiation Fail. Please check for the reason.")
        self.pipeline_editor = PipelineEditor(self, comfy_url=f'http://localhost:{self.prompt_server.port}/')
        
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
        
        if loading_window is not None:
            loading_window.close()
        self.show()
        self.raise_()
    
    def closeEvent(self, event: QCloseEvent) -> None:
        self._comfy_stopper() if self._comfy_stopper is not None else None
        return super().closeEvent(event)


def main():
    os.environ['EDITOR_MODE'] = '1'
    os.environ['QSG_RHI_BACKEND']='opengl'
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    extra = {'density_scale': '-2'}
    apply_stylesheet(app, theme='dark_teal.xml', extra=extra)
    
    log_window = LoadingWindow()
    app.processEvents()
    MainWindow(log_window)
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
