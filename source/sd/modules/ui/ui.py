import gradio as gr
from pathlib import Path
from .ui_utils import *


def view_generation_progress_block():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    value=load_images(config.test_dir / 'boat' / 'corr_map_vis'),
                    rows=6,
                    columns=4,
                    object_fit='scale-down',
                )
            with gr.Column(scale=1):
                ...

    return demo


def api(share=False):
    demo = view_generation_progress_block()
    demo.queue(4)
    demo.launch(share=share)
