import gradio as gr
from pathlib import Path
from .ui_utils import *


def view_generation_progress_block(source):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    value=load_images(source),
                    rows=6,
                    columns=4,
                    object_fit='scale-down',
                )
            # with gr.Column(scale=1):
            #     ...

    return demo


def api(
    source=config.test_dir / 'boat' / 'latents',
    share=False
):
    demo = view_generation_progress_block(source)
    demo.queue(4)
    demo.launch(share=share)
