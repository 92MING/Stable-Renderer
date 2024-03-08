import os, sys
import time
from multiprocessing import Process
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QWidget, QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView
import requests

if __name__ == '__main__':
    source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.append(source_path)
    __package__ = 'ui.components.pipeline_editor'

def test_comfyUI_started(port=8188, timeout: float=1)->bool:
    url = f'http://localhost:{port}/'
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return True
    except:
        pass
    return False
    
def start_comfyUI(timeout=12):
    print('starting ComfyUI...')
    
    # from comfyUI.comfy.main import main
    #process = Process(target=main)
    # return process


class PipelineEditor(QWebEngineView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.comfyUI_process = start_comfyUI()
        
        self.setUrl(QUrl("http://localhost:8188/"))

        self.show()

    def closeEvent(self, event):
        print("terminating comfyUI process...")
        #self.comfyUI_process.terminate()
        event.accept()


__all__ = ['PipelineEditor']


if __name__ == '__main__':
    start_comfyUI()
    exit()
    app = QApplication([])
    window = QMainWindow()
    window.setCentralWidget(PipelineEditor())
    window.show()
    app.exec_()