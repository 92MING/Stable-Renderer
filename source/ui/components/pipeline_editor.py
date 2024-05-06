import os, sys
from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from common_utils.debug_utils import EditorLogger

if __name__ == '__main__':
    source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.append(source_path)
    __package__ = 'ui.components.pipeline_editor'


class PipelineEditor(QWebEngineView):
    
    def __init__(self, *args, comfy_url='http://localhost:8188/', **kwargs):
        super().__init__(*args, **kwargs)
        self.comfy_url = QUrl(comfy_url)
        self.setUrl(self.comfy_url)
        self.show()

    def closeEvent(self, event):
        EditorLogger.print("terminating comfyUI process...")
        #self.comfyUI_process.terminate()
        event.accept()


__all__ = ['PipelineEditor']