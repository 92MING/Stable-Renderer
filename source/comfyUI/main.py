import os, sys
_COMFYUI_PROJ_PATH = os.path.dirname(os.path.abspath(__file__))
_STABLE_RENDERER_PROJ_PATH = os.path.dirname(_COMFYUI_PROJ_PATH)
sys.path.insert(0, _COMFYUI_PROJ_PATH)
sys.path.insert(0, _STABLE_RENDERER_PROJ_PATH)

from common_utils.debug_utils import ComfyUILogger
from common_utils.global_utils import GetOrCreateGlobalValue, is_game_mode
from common_utils.system_utils import get_available_port, check_port_is_using

import asyncio
import itertools
import shutil
import threading
import gc
import yaml
import importlib.util
import time
import asyncio

from typing import Union

import folder_paths
import server
import execution
import nodes
import cuda_malloc
import comfy.utils
import comfy.model_management

if __name__ == '__main__':
    import comfy.options
    comfy.options.enable_args_parsing()

from comfy.cli_args import args

def run()->Union[execution.PromptExecutor, None]:
    '''
    Run comfyUI.
    
    When game_mode, it will return the prompt_executor.
    '''
    
    should_run_web_server = (__name__ == '__main__')
    if not should_run_web_server:
        should_run_web_server = not is_game_mode()  # web server only run in editor mode
    
    if should_run_web_server:
        event_loop = GetOrCreateGlobalValue("__COMFYUI_EVENT_LOOP__", lambda: asyncio.new_event_loop())
        asyncio.set_event_loop(event_loop)
        prompt_server = server.PromptServer(event_loop)
        prompt_queue = execution.PromptQueue(prompt_server)
    else:
        prompt_server = None
        prompt_queue = None
    prompt_executor = execution.PromptExecutor(prompt_server)   # sever can be none, means running in game mode
    
    def execute_prestartup_script():
        def execute_script(script_path):
            module_name = os.path.splitext(script_path)[0]
            try:
                spec = importlib.util.spec_from_file_location(module_name, script_path)
                module = importlib.util.module_from_spec(spec)  # type: ignore
                spec.loader.exec_module(module) # type: ignore
                return True
            except Exception as e:
                ComfyUILogger.warn(f"Failed to execute startup-script: {script_path} / {e}")
            return False

        node_paths = folder_paths.get_folder_paths("custom_nodes")
        for custom_node_path in node_paths:
            possible_modules = os.listdir(custom_node_path)
            node_prestartup_times = []

            for possible_module in possible_modules:
                module_path = os.path.join(custom_node_path, possible_module)
                if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                    continue

                script_path = os.path.join(module_path, "prestartup_script.py")
                if os.path.exists(script_path):
                    time_before = time.perf_counter()
                    success = execute_script(script_path)
                    node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
                    
        if len(node_prestartup_times) > 0:
            ComfyUILogger.debug("\nPrestartup times for custom nodes:")
            for n in sorted(node_prestartup_times):
                if n[2]:
                    import_message = ""
                else:
                    import_message = " (PRESTARTUP FAILED)"
                ComfyUILogger.debug("{:6.1f} seconds{}:".format(n[0], import_message), n[1])

    if should_run_web_server:
        execute_prestartup_script()

    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        ComfyUILogger.debug("Set cuda device to:", args.cuda_device)

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    def cuda_malloc_warning():
        device = comfy.model_management.get_torch_device()
        device_name = comfy.model_management.get_torch_device_name(device)
        cuda_malloc_warning = False
        if "cudaMallocAsync" in device_name:
            for b in cuda_malloc.blacklist:
                if b in device_name:
                    cuda_malloc_warning = True
            if cuda_malloc_warning:
                ComfyUILogger.warn("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")
    
    def prompt_worker(e: execution.PromptExecutor, q: execution.PromptQueue, server: server.PromptServer):
        
        last_gc_collect = 0
        need_gc = False
        gc_collect_interval = 10.0

        while True:
            timeout = 1000.0
            if need_gc:
                timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

            queue_item = q.get(timeout=timeout)
            if queue_item is not None:
                item, item_id = queue_item
                execution_start_time = time.perf_counter()
                prompt_id = item[1]
                server.last_prompt_id = prompt_id

                e.execute(item[2], prompt_id, item[3], item[4])
                need_gc = True
                q.task_done(item_id,
                            e.outputs_ui,
                            status=execution.PromptQueue.ExecutionStatus(
                                status_str='success' if e.success else 'error',
                                completed=e.success,
                                messages=e.status_messages
                                )
                            )
                if server.client_id is not None:
                    server.send_sync("executing", { "node": None, "prompt_id": prompt_id }, server.client_id)

                current_time = time.perf_counter()
                execution_time = current_time - execution_start_time
                ComfyUILogger.debug("Prompt executed in {:.2f} seconds".format(execution_time))

            flags = q.get_flags()
            free_memory = flags.get("free_memory", False)

            if flags.get("unload_models", free_memory):
                comfy.model_management.unload_all_models()
                need_gc = True
                last_gc_collect = 0

            if free_memory:
                e.reset()
                need_gc = True
                last_gc_collect = 0

            if need_gc:
                current_time = time.perf_counter()
                if (current_time - last_gc_collect) > gc_collect_interval:
                    gc.collect()
                    comfy.model_management.soft_empty_cache()
                    last_gc_collect = current_time
                    need_gc = False
    
    async def run_web_server(server: server.PromptServer, address='', port=8188, verbose=True, call_on_start=None):
        if check_port_is_using(port):
            ComfyUILogger.info(f'Port {port} is not available, try to find another port...')
            port = get_available_port()
        await asyncio.gather(server.start(address, port, verbose, call_on_start), server.publish_loop())

    def hijack_progress(prompt_server: server.PromptServer):
        def hook(value, total, preview_image):
            comfy.model_management.throw_exception_if_processing_interrupted()
            progress = {"value": value, "max": total, "prompt_id": prompt_server.last_prompt_id, "node": prompt_server.last_node_id}

            prompt_server.send_sync("progress", progress, prompt_server.client_id)
            if preview_image is not None:
                prompt_server.send_sync(server.BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, prompt_server.client_id)
        comfy.utils.set_progress_bar_global_hook(hook)

    def cleanup_temp():
        temp_dir = folder_paths.get_temp_directory()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    def load_extra_path_config(yaml_path):
        with open(yaml_path, 'r') as stream:
            config = yaml.safe_load(stream)
        for c in config:
            conf = config[c]
            if conf is None:
                continue
            base_path = None
            if "base_path" in conf:
                base_path = conf.pop("base_path")
            for x in conf:
                for y in conf[x].split("\n"):
                    if len(y) == 0:
                        continue
                    full_path = y
                    if base_path is not None:
                        full_path = os.path.join(base_path, full_path)
                    ComfyUILogger.debug("Adding extra search path", x, full_path)
                    folder_paths.add_model_folder_path(x, full_path)

    if args.temp_directory: # TODO: changed to put tmp dir to the project tmp dir instead of comfyUI's tmp dir
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        ComfyUILogger.debug(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    # I disabled the auto update, since the project is greatly modified from original comfyUI
    
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)

    nodes.init_custom_nodes()

    cuda_malloc_warning()

    if should_run_web_server:
        prompt_server.add_routes()  # type: ignore
        hijack_progress(prompt_server)   # type: ignore
        threading.Thread(target=prompt_worker, daemon=True, args=(prompt_executor, prompt_queue, prompt_server,)).start()

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        ComfyUILogger.debug(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    #These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        ComfyUILogger.debug(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if should_run_web_server:
        call_on_start = None
        if args.auto_launch:
            def startup_server(address, port):
                import webbrowser
                if os.name == 'nt' and address == '0.0.0.0':
                    address = '127.0.0.1'
                webbrowser.open(f"http://{address}:{port}")
            call_on_start = startup_server

        try:
            event_loop.run_until_complete(
                run_web_server(
                    server=prompt_server,  # type: ignore
                    address=args.listen, 
                    port=args.port, 
                    verbose=not args.dont_print_server, 
                    call_on_start=call_on_start
                    )
                )
        except KeyboardInterrupt:
            ComfyUILogger.success("Server Stopped")
            
        cleanup_temp()
    
    else:
        return prompt_executor

if __name__ == '__main__':
    run()