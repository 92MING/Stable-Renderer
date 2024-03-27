import sys
import copy
import logging
import threading
import heapq
import traceback
import inspect
import torch
import uuid

from typing import (List, Literal, NamedTuple, Optional, TYPE_CHECKING, Tuple,
                    Set, Dict, Union, Any, Type)
from dataclasses import dataclass

import nodes
import comfy.model_management

from common_utils.decorators import singleton, class_property
from comfyUI.types import *
from comfyUI.adapters import find_adapter, Adapter

if TYPE_CHECKING:
    from comfyUI.server import PromptServer


def _get_input_data(inputs: NodeInputs, 
                    class_def: Type[ComfyUINode], 
                    unique_id: str, 
                    outputs={}, 
                    prompt={}, 
                    extra_data={}):
    '''
    Pack inputs form value/nodes into a dict for the node to use.
    
    Args:
        - inputs: {param name, value}. Values can be:
            * [from_node_id, from_node_output_slot_index] (means the value is from another node's output)
            * a value
        - class_def: the class of the node
        - unique_id: the id of this node
    '''
    valid_inputs = class_def.INPUT_TYPES()
    lazy_inputs = class_def.LAZY_INPUTS if hasattr(class_def, "LAZY_INPUTS") else []
    
    input_data_all = {}
    for input_param_name in inputs:
        input_data = inputs[input_param_name]
        
        if isinstance(input_data, NodeBindingParam):    # [from_node_id, from_node_output_slot_index]
            from_node_id = input_data[0]
            output_slot_index = input_data[1]
            
            if input_param_name in lazy_inputs:
                val = Lazy(from_node_id, output_slot_index)
            elif from_node_id not in outputs:
                val = (None,)
            else:
                val = outputs[from_node_id][output_slot_index]
            
            input_data_all[input_param_name] = val
            
        else:   # a value
            if input_param_name in lazy_inputs:
                input_data_all[input_param_name] = Lazy(input_data)
                
            elif ("required" in valid_inputs and input_param_name in valid_inputs["required"]) or ("optional" in valid_inputs and input_param_name in valid_inputs["optional"]):
                input_data_all[input_param_name] = [input_data]

    def get_proper_hidden_keys(key: str):
        keys = set()
        keys.add(key)
        no_bottom_bar_key = key
        while no_bottom_bar_key.startswith("_"):
            no_bottom_bar_key = key[1:]
        keys.add(no_bottom_bar_key)
        for k in keys.copy():
            keys.add(k.lower())
            keys.add(k.upper())
        return keys
    def try_put_hidden_value(origin_param_name: str):
        if not extra_data:
            return
        possible_keys = get_proper_hidden_keys(origin_param_name)
        for k in possible_keys:
            if k in extra_data:
                input_data_all[origin_param_name] = [extra_data[k]]
                break
    
    if "hidden" in valid_inputs:
        h = valid_inputs["hidden"]
        for input_param_name in h:
            if h[input_param_name] == "PROMPT":
                input_data_all[input_param_name] = [prompt]
            if h[input_param_name] == "EXTRA_PNGINFO":
                if "extra_pnginfo" in extra_data:
                    input_data_all[input_param_name] = [extra_data['extra_pnginfo']]
            if h[input_param_name] == "UNIQUE_ID":
                input_data_all[input_param_name] = [unique_id]
            else:
                try_put_hidden_value(input_param_name)
    return input_data_all

def _get_node_func_ret(node: Union[ComfyUINode, Type[ComfyUINode]], 
                       func_params: dict,    # not `NodeInputs`, cuz values should already be resolved
                       func: str,   # the target function name of the node
                       allow_interrupt=False):
    '''return the result of the given function name of the node.'''
    node_type = node.__class__ if not isinstance(node, type) else node
    if not hasattr(node_type, func):
        raise AttributeError(f"Node {node} has no function {func}.")
    
    # check if node wants the lists
    input_is_list = False
    if hasattr(node, "INPUT_IS_LIST"):
        input_is_list = node.INPUT_IS_LIST

    if len(func_params) == 0:
        max_len_input = 0
    else:
        max_len_input = max([len(x) for x in func_params.values()])
    
    # get a slice of inputs, repeat last input when list isn't long enough
    def slice_dict(d, i):
        d_new = dict()
        for k,v in d.items():
            d_new[k] = v[i if len(v) > i else -1]
        return d_new
    
    real_func = getattr(node, func)
    if not callable(real_func):
        raise AttributeError(f"Node {node} has no function {func}.")
    
    results = []
    if input_is_list:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(real_func(**func_params))
    elif max_len_input == 0:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(real_func())
    else:
        for i in range(max_len_input):
            if allow_interrupt:
                nodes.before_node_execution()
            results.append(real_func(**slice_dict(func_params, i)))
    return results

def _format_value(x):
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)

class RecursiveExecuteResult(NamedTuple):
    success: bool
    error_details: Optional[dict]
    exception: Optional[Exception]

@singleton(cross_module_singleton=True)
class PromptExecutor:
    
    server: Optional['PromptServer']
    '''The server that this executor is associated with. If None, the executor'''
    node_pool: NodePool
    '''
    Pool of all created nodes
    {node_id: node_instance, ...}
    '''
    old_prompt: PROMPT
    '''old prompt'''
    outputs_ui: NodeOutputs_UI
    '''outputs for UI'''
    outputs: NodeOutputs
    '''outputs for the prompt'''
    status_messages: StatusMsg
    '''status messages'''
    success: bool
    '''whether the execution is successful or not'''
    
    @class_property
    def Instance(cls)->'PromptExecutor':
        return cls()    # type: ignore
    
    def __init__(self, server: Optional['PromptServer']=None):
        self.server = server
        self.reset()
    
    def _get_output_data(self, node: ComfyUINode, input_data_all)->Tuple[List[Any], Dict[str, Any]]:
        '''
        Get the output data of the given node.
        
        Return:
            - outputs (list of list of values, e.g. [[val1, val2], [val3, val4], ...])
            - ui values (values to be shown on UI, e.g. {key: [val1, val2, ...], ...})
        '''
        results = []
        uis = {}
        return_values = _get_node_func_ret(node, 
                                            input_data_all, 
                                            node.FUNCTION, 
                                            allow_interrupt=True)

        for r in return_values:
            if isinstance(r, dict):
                if 'ui' in r:
                    for key, val in r['ui'].items():
                        if key not in uis:
                            uis[key] = []
                        uis[key].extend(val)    # val is a list or tuple
                if 'result' in r:
                    results.append(r['result'])
            else:
                results.append(r)
        
        output = []
        if len(results) > 0:
            # check which outputs need concatenating
            if hasattr(node, "OUTPUT_IS_LIST"):
                output_is_list = node.OUTPUT_IS_LIST
            else:
                output_is_list = [False] * len(results[0])

            # merge node execution results
            for i, is_list in zip(range(len(results[0])), output_is_list):
                if is_list:
                    output.append([x for o in results for x in o[i]])
                else:
                    output.append([o[i] for o in results])

        return output, uis

    def _recursive_will_execute(self, prompt: PROMPT, current_node_id, memo={}):
        outputs = self.outputs
        unique_id = current_node_id

        if unique_id in memo:
            return memo[unique_id]

        inputs = prompt[unique_id]['inputs']
        will_execute = []
        if unique_id in outputs:
            return []

        for x in inputs:
            input_data = inputs[x]
            if isinstance(input_data, list):
                input_unique_id = input_data[0]
                output_index = input_data[1]
                if input_unique_id not in outputs:
                    will_execute += self._recursive_will_execute(prompt, input_unique_id, memo)

        memo[unique_id] = will_execute + [unique_id]
        return memo[unique_id]
    
    def _get_node_output(self, node_id: str, slot_index: int):
        '''special method for "Lazy" type values, to get the real value from a binding'''
        pass
    
    def _recursive_execute(self,
                           prompt: PROMPT, 
                           current_node_id: str, 
                           extra_data, 
                           executed: Set[str],   # set of node ids that have been executed 
                           prompt_id: str,   # just a random string by uuid4
                           )->RecursiveExecuteResult:
        inputs = prompt[current_node_id]['inputs']
        class_type = prompt[current_node_id]['class_type']
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        
        if current_node_id in self.outputs:
            return RecursiveExecuteResult(True, None, None)
        
        for param_name in inputs:
            input_data = inputs[param_name]

            if isinstance(input_data, NodeBindingParam):
                node_id = input_data[0]
                output_slot_index = input_data[1]
                if node_id not in self.outputs:
                    result = self._recursive_execute(prompt, 
                                                    node_id, 
                                                    extra_data, 
                                                    executed,
                                                    prompt_id)
                    if result[0] is not True:
                        # Another node failed further upstream
                        return result

        input_data_all = None
        try:
            input_data_all = _get_input_data(inputs,
                                            class_def, 
                                            current_node_id,
                                            self.outputs, 
                                            prompt, 
                                            extra_data)
            if self.server:
                if self.server.client_id is not None:
                    self.server.last_node_id = current_node_id
                    self.server.send_sync("executing",
                                        { "node": current_node_id, "prompt_id": prompt_id }, 
                                        self.server.client_id)

            node = self.get_node(current_node_id, class_type)

            output_data, output_ui = self._get_output_data(node, input_data_all)
            self.outputs[current_node_id] = output_data
            if len(output_ui) > 0:
                self.outputs_ui[current_node_id] = output_ui
                if self.server:
                    if self.server.client_id is not None:
                        self.server.send_sync("executed", 
                                              { "node": current_node_id, "output": output_ui, "prompt_id": prompt_id }, 
                                              self.server.client_id)
        
        except comfy.model_management.InterruptProcessingException as iex:
            logging.info("Processing interrupted")

            # skip formatting inputs/outputs
            error_details = {
                "node_id": current_node_id,
            }

            return RecursiveExecuteResult(False, error_details, iex)
        
        except Exception as ex:
            typ, _, tb = sys.exc_info()
            exception_type = _get_full_type_name(typ)
            input_data_formatted = {}
            if input_data_all is not None:
                input_data_formatted = {}
                for name, inputs in input_data_all.items():
                    input_data_formatted[name] = [_format_value(x) for x in inputs]

            output_data_formatted = {}
            for node_id, node_outputs in self.outputs.items():
                output_data_formatted[node_id] = [[_format_value(x) for x in l] for l in node_outputs]

            logging.error("!!! Exception during processing !!!")
            logging.error(traceback.format_exc())

            error_details = {
                "node_id": current_node_id,
                "exception_message": str(ex),
                "exception_type": exception_type,
                "traceback": traceback.format_tb(tb),
                "current_inputs": input_data_formatted,
                "current_outputs": output_data_formatted
            }
            return RecursiveExecuteResult(False, error_details, ex)

        executed.add(current_node_id)

        return RecursiveExecuteResult(True, None, None)

    def _recursive_output_delete_if_changed(self, 
                                            prompt: PROMPT, 
                                            current_node_id: str):
        old_prompt = self.old_prompt
        outputs = self.outputs
        unique_id = current_node_id
        inputs = prompt[unique_id]['inputs']
        class_type = prompt[unique_id]['class_type']
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

        is_changed_old = ''
        is_changed = ''
        to_delete = False
        if hasattr(class_def, 'IS_CHANGED'):
            if unique_id in old_prompt and 'is_changed' in old_prompt[unique_id]:
                is_changed_old = old_prompt[unique_id]['is_changed']
            if 'is_changed' not in prompt[unique_id]:
                input_data_all = _get_input_data(inputs, class_def, unique_id, outputs)
                if input_data_all is not None:
                    try:
                        #is_changed = class_def.IS_CHANGED(**input_data_all)
                        is_changed = _get_node_func_ret(class_def, input_data_all, "IS_CHANGED")
                        prompt[unique_id]['is_changed'] = is_changed
                    except:
                        to_delete = True
            else:
                is_changed = prompt[unique_id]['is_changed']

        if unique_id not in outputs:
            return True

        if not to_delete:
            if is_changed != is_changed_old:
                to_delete = True
            elif unique_id not in old_prompt:
                to_delete = True
            elif inputs == old_prompt[unique_id]['inputs']:
                for x in inputs:
                    input_data = inputs[x]

                    if isinstance(input_data, NodeBindingParam):
                        input_unique_id = input_data[0]
                        output_index = input_data[1]
                        if input_unique_id in outputs:
                            to_delete = self._recursive_output_delete_if_changed(prompt, input_unique_id)
                        else:
                            to_delete = True
                        if to_delete:
                            break
            else:
                to_delete = True

        if to_delete:
            d = outputs.pop(unique_id)
            del d
        return to_delete

    def reset(self):
        self.node_pool = NodePool()
        
        self.outputs_ui = {}
        self.outputs = {}
        
        self.status_messages = []
        
        self.success = True
        
        self.old_prompt = PROMPT()
        
    def get_node(self, node_id:str, class_type_name: str)->ComfyUINode:
        '''Return the target node from the pool, or create a new one if not found.'''
        return self.node_pool.get_node(node_id, class_type_name)
    
    def add_message(self, event, data: dict, broadcast: bool):
        self.status_messages.append((event, data))
        if self.server:
            if self.server.client_id is not None or broadcast:
                self.server.send_sync(event, data, self.server.client_id)

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]

        # First, send back the status to the frontend depending
        # on the exception type
        if isinstance(ex, comfy.model_management.InterruptProcessingException):
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
            }
            self.add_message("execution_interrupted", mes, broadcast=True)
        else:
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),

                "exception_message": error["exception_message"],
                "exception_type": error["exception_type"],
                "traceback": error["traceback"],
                "current_inputs": error["current_inputs"],
                "current_outputs": error["current_outputs"],
            }
            self.add_message("execution_error", mes, broadcast=False)
        
        # Next, remove the subsequent outputs since they will not be executed
        to_delete = []
        for o in self.outputs:
            if (o not in current_outputs) and (o not in executed):
                to_delete += [o]
                if o in self.old_prompt:
                    d = self.old_prompt.pop(o)
                    del d
        for o in to_delete:
            d = self.outputs.pop(o)
            del d

    def execute(self, 
                prompt: Union[PROMPT, dict], 
                prompt_id: Optional[str]=None, # random string by uuid4 
                extra_data={}, 
                execute_outputs=[]):
        if not isinstance(prompt, PROMPT):
            prompt = PROMPT(prompt)
        
        if prompt_id is None:
            prompt_id = str(uuid.uuid4())
        
        nodes.interrupt_processing(False)

        if self.server:
            if "client_id" in extra_data:
                self.server.client_id = extra_data["client_id"]
            else:
                self.server.client_id = None

        self.status_messages = []
        self.add_message("execution_start", { "prompt_id": prompt_id}, broadcast=False)

        with torch.inference_mode():
            #delete cached outputs if nodes don't exist for them
            to_delete = []
            for node_pool_key in self.outputs:
                if node_pool_key not in prompt:
                    to_delete += [node_pool_key]
            for node_pool_key in to_delete:
                d = self.outputs.pop(node_pool_key)
                del d
                
            nodes_to_be_deleted: List[NodePool.NodePoolKey] = []
            for node_pool_key in self.node_pool:
                if node_pool_key.node_id not in prompt:
                    nodes_to_be_deleted += [node_pool_key]
                else:
                    p = prompt[node_pool_key.node_id]
                    if node_pool_key.node_type_name != p['class_type']:
                        nodes_to_be_deleted += [node_pool_key]
            for node_pool_key in nodes_to_be_deleted:
                d = self.node_pool.pop(node_pool_key.node_id)
                del d

            for node_id in prompt:
                self._recursive_output_delete_if_changed(prompt, node_id)

            current_outputs = set(self.outputs.keys())
            for x in list(self.outputs_ui.keys()):
                if x not in current_outputs:
                    d = self.outputs_ui.pop(x)
                    del d

            comfy.model_management.cleanup_models()
            self.add_message("execution_cached", { "nodes": list(current_outputs) , "prompt_id": prompt_id}, broadcast=False)
            executed = set()
            output_node_id = None
            to_execute = []

            for node_id in list(execute_outputs):
                to_execute += [(0, node_id)]

            while len(to_execute) > 0:
                #always execute the output that depends on the least amount of unexecuted nodes first
                memo = {}
                to_execute = sorted(list(map(lambda a: (len(self._recursive_will_execute(prompt, a[-1], memo)), a[-1]), to_execute)))
                output_node_id = to_execute.pop(0)[-1]

                # This call shouldn't raise anything if there's an error deep in
                # the actual SD code, instead it will report the node where the
                # error was raised
                self.success, error, ex = self._recursive_execute(prompt,  
                                                                output_node_id, 
                                                                extra_data, 
                                                                executed, 
                                                                prompt_id,)
                if self.success is not True:
                    self.handle_execution_error(prompt_id, prompt, current_outputs, executed, error, ex)
                    break

            for x in executed:
                self.old_prompt[x] = copy.deepcopy(prompt[x])
                
            if self.server:
                self.server.last_node_id = None
            
            if comfy.model_management.DISABLE_SMART_MEMORY:
                comfy.model_management.unload_all_models()

@dataclass
class ValidateInputsResult:
    '''dataclass for the result of `validate_inputs` function.'''
    
    result: bool
    '''whether the validation is successful or not'''
    errors: List[dict]
    '''list of error messages'''
    node_id: str
    '''the node id that is being validated'''
    adapter: Optional[Adapter] = None
    '''the adapter that is used to convert the input'''
    
    def __getitem__(self, item: int):
        if item not in (0, 1, 2, 3):
            raise IndexError(f"Index out of range: {item}")
        if item == 0:
            return self.result
        if item == 1:
            return self.errors
        if item == 2:
            return self.node_id
        if item == 3:
            return self.adapter

def validate_inputs(prompt: PROMPT, 
                    node_id: str, 
                    validated: Dict[str, ValidateInputsResult])->ValidateInputsResult:
    if node_id in validated:
        return validated[node_id]

    inputs = prompt[node_id]['inputs']
    class_type = prompt[node_id]['class_type']
    obj_class = nodes.NODE_CLASS_MAPPINGS[class_type]

    class_inputs = obj_class.INPUT_TYPES()
    required_inputs = class_inputs['required']

    errors = []
    valid = True
    adapter = None

    validate_function_inputs = []
    if hasattr(obj_class, "VALIDATE_INPUTS"):
        validate_function_inputs = inspect.getfullargspec(obj_class.VALIDATE_INPUTS).args
        if isinstance(obj_class.__dict__['VALIDATE_INPUTS'], classmethod):
            validate_function_inputs = validate_function_inputs[1:] # classmethod, remove 'cls'
        elif not isinstance(obj_class.__dict__['VALIDATE_INPUTS'], staticmethod):
            validate_function_inputs = validate_function_inputs[1:] # normal method, remove 'self'
        
    for x in required_inputs:
        if x not in inputs:
            error = {
                "type": "required_input_missing",
                "message": "Required input is missing",
                "details": f"{x}",
                "extra_info": {
                    "input_name": x
                }
            }
            errors.append(error)
            continue

        val = inputs[x]
        info = required_inputs[x]
        type_input = info[0]
        if isinstance(val, list) and isinstance(val[0], str) and isinstance(val[1], int):   # [str, int] means binding 
            if len(val) != 2:
                error = {
                    "type": "bad_linked_input",
                    "message": "Bad linked input, must be a length-2 list of [node_id, slot_index]",
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val
                    }
                }
                errors.append(error)
                continue

            o_id = val[0]
            o_class_type = prompt[o_id]['class_type']
            r = nodes.NODE_CLASS_MAPPINGS[o_class_type].RETURN_TYPES
            
            from_type_name = r[val[1]]
            if from_type_name == 'ANY':
                from_type_name = '*'
            
            if '*' not in (from_type_name, type_input) and from_type_name != type_input:
                adapter = find_adapter(from_type_name, type_input)
                if adapter is None:
                    details = f"{x}, {from_type_name} != {type_input}"
                    error = {
                        "type": "return_type_mismatch",
                        "message": "Return type mismatch between linked nodes",
                        "details": details,
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_type": from_type_name,
                            "linked_node": val
                        }
                    }
                    errors.append(error)
                    continue
                
            try:
                r = validate_inputs(prompt, o_id, validated)
                if r[0] is False:
                    # `r` will be set in `validated[o_id]` already
                    valid = False
                    continue
            except Exception as ex:
                typ, _, tb = sys.exc_info()
                valid = False
                exception_type = _get_full_type_name(typ)
                reasons = [{
                    "type": "exception_during_inner_validation",
                    "message": "Exception when validating inner node",
                    "details": str(ex),
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "exception_message": str(ex),
                        "exception_type": exception_type,
                        "traceback": traceback.format_tb(tb),
                        "linked_node": val
                    }
                }]
                validated[o_id] = ValidateInputsResult(False, reasons, o_id)
                continue
        else:
            try:
                if type_input == "INT":
                    val = int(val)
                    inputs[x] = val
                if type_input == "FLOAT":
                    val = float(val)
                    inputs[x] = val
                if type_input == "STRING":
                    val = str(val)
                    inputs[x] = val
            except Exception as ex:
                error = {
                    "type": "invalid_input_type",
                    "message": f"Failed to convert an input value to a {type_input} value",
                    "details": f"{x}, {val}, {ex}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val,
                        "exception_message": str(ex)
                    }
                }
                errors.append(error)
                continue

            if len(info) > 1:
                if "min" in info[1] and val < info[1]["min"]:
                    error = {
                        "type": "value_smaller_than_min",
                        "message": "Value {} smaller than min of {}".format(val, info[1]["min"]),
                        "details": f"{x}",
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue
                if "max" in info[1] and val > info[1]["max"]:
                    error = {
                        "type": "value_bigger_than_max",
                        "message": "Value {} bigger than max of {}".format(val, info[1]["max"]),
                        "details": f"{x}",
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue

            if x not in validate_function_inputs:
                if isinstance(type_input, list):
                    if val not in type_input:
                        input_config = info
                        list_info = ""

                        # Don't send back gigantic lists like if they're lots of
                        # scanned model filepaths
                        if len(type_input) > 20:
                            list_info = f"(list of length {len(type_input)})"
                            input_config = None
                        else:
                            list_info = str(type_input)

                        error = {
                            "type": "value_not_in_list",
                            "message": "Value not in list",
                            "details": f"{x}: '{val}' not in {list_info}",
                            "extra_info": {
                                "input_name": x,
                                "input_config": input_config,
                                "received_value": val,
                            }
                        }
                        errors.append(error)
                        continue

    if len(validate_function_inputs) > 0:
        input_data_all = _get_input_data(inputs, obj_class, node_id)
        input_filtered = {}
        for x in input_data_all:
            if x in validate_function_inputs:
                input_filtered[x] = input_data_all[x]

        ret = _get_node_func_ret(obj_class, input_filtered, "VALIDATE_INPUTS")
        for x in input_filtered:
            for i, r in enumerate(ret):
                if r is not True:
                    details = f"{x}"
                    if r is not False:
                        details += f" - {str(r)}"

                    error = {
                        "type": "custom_validation_failed",
                        "message": "Custom validation failed for node",
                        "details": details,
                        "extra_info": {
                            "input_name": x,
                            "input_config": info,
                            "received_value": val,
                        }
                    }
                    errors.append(error)
                    continue

    if len(errors) > 0 or valid is not True:
        ret = ValidateInputsResult(result=False, errors=errors, node_id=node_id, adapter=adapter)
    else:
        ret = ValidateInputsResult(result=True, errors=[], node_id=node_id, adapter=adapter)
    
    validated[node_id] = ret
    return ret

def _get_full_type_name(klass):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__

@dataclass
class ValidatePromptResult:
    '''dataclass for the result of `validate_prompt` function.'''
    
    result: bool
    '''whether the validation is successful or not'''
    errors: Optional[dict]
    '''the error messages if the validation failed'''
    good_outputs: List[str]
    '''list of output node ids that passed the validation.'''
    node_errors: Dict[str, dict]
    '''dict of node_id: error messages'''
    
    _prompt: dict
    '''The real input prompt, a dictionary (converted from json)'''
    _formatted_prompt: PROMPT = None     # type: ignore
    '''The properly formatted prompt, in `PROMPT` type'''
    
    @property
    def formatted_prompt(self):
        if not self._formatted_prompt:
            self._formatted_prompt = PROMPT(self._prompt)
        return self._formatted_prompt
    
    def __getitem__(self, item: int):
        if item not in (0, 1, 2, 3, 4):
            raise IndexError(f"Index out of range: {item}. It should be in [0, 1, 2, 3, 4]")
        if item == 0:
            return self.result
        if item == 1:
            return self.errors
        if item == 2:
            return self.good_outputs
        if item == 3:
            return self.node_errors
        if item == 4:
            return self.formatted_prompt
            
def validate_prompt(prompt: PROMPT) -> ValidatePromptResult:
    outputs = set()
    for x in prompt:
        class_ = nodes.NODE_CLASS_MAPPINGS[prompt[x]['class_type']]
        if hasattr(class_, 'OUTPUT_NODE') and class_.OUTPUT_NODE == True:
            outputs.add(x)

    if len(outputs) == 0:
        error = {
            "type": "prompt_no_outputs",
            "message": "Prompt has no outputs",
            "details": "",
            "extra_info": {}
        }
        return ValidatePromptResult(result=False, errors=error, good_outputs=[], node_errors={}, _prompt=prompt)

    good_outputs = set()
    errors = []
    node_errors = {}
    validated = {}
    for o in outputs:
        valid = False
        reasons = []
        try:
            m = validate_inputs(prompt, o, validated)
            valid = m[0]
            reasons = m[1]
        except Exception as ex:
            typ, _, tb = sys.exc_info()
            valid = False
            exception_type = _get_full_type_name(typ)
            reasons = [{
                "type": "exception_during_validation",
                "message": "Exception when validating node",
                "details": str(ex),
                "extra_info": {
                    "exception_type": exception_type,
                    "traceback": traceback.format_tb(tb)
                }
            }]
            validated[o] = (False, reasons, o)

        if valid is True:
            good_outputs.add(o)
        else:
            logging.error(f"Failed to validate prompt for output {o}:")
            if len(reasons) > 0:
                logging.error("* (prompt):")
                for reason in reasons:
                    logging.error(f"  - {reason['message']}: {reason['details']}")
            errors += [(o, reasons)]
            for node_id, result in validated.items():
                valid = result[0]
                reasons = result[1]
                # If a node upstream has errors, the nodes downstream will also
                # be reported as invalid, but there will be no errors attached.
                # So don't return those nodes as having errors in the response.
                if valid is not True and len(reasons) > 0:
                    if node_id not in node_errors:
                        class_type = prompt[node_id]['class_type']
                        node_errors[node_id] = {
                            "errors": reasons,
                            "dependent_outputs": [],
                            "class_type": class_type
                        }
                        logging.error(f"* {class_type} {node_id}:")
                        for reason in reasons:
                            logging.error(f"  - {reason['message']}: {reason['details']}")
                    node_errors[node_id]["dependent_outputs"].append(o)
            logging.error("Output will be ignored")

    if len(good_outputs) == 0:
        errors_list = []
        for o, errors in errors:
            for error in errors:
                errors_list.append(f"{error['message']}: {error['details']}")
        errors_list = "\n".join(errors_list)

        error = {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": errors_list,
            "extra_info": {}
        }

        return ValidatePromptResult(result=False, errors=error, good_outputs=list(good_outputs), node_errors=node_errors, _prompt=prompt)

    return ValidatePromptResult(result=True, errors=None, good_outputs=list(good_outputs), node_errors=node_errors, _prompt=prompt)

MAXIMUM_HISTORY_SIZE = 10000

class PromptQueue:
    def __init__(self, server: 'PromptServer'):
        self.server = server
        self.mutex = threading.RLock()
        self.not_empty = threading.Condition(self.mutex)
        self.task_counter = 0
        self.queue = []
        self.currently_running = {}
        self.history = {}
        self.flags = {}
        server.prompt_queue = self

    def put(self, item: QueueTask):
        with self.mutex:
            heapq.heappush(self.queue, item)
            self.server.queue_updated()
            self.not_empty.notify()

    def get(self, timeout=None):
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait(timeout=timeout)
                if timeout is not None and len(self.queue) == 0:
                    return None
            item = heapq.heappop(self.queue)
            i = self.task_counter
            self.currently_running[i] = copy.deepcopy(item)
            self.task_counter += 1
            self.server.queue_updated()
            return (item, i)

    class ExecutionStatus(NamedTuple):
        status_str: Literal['success', 'error']
        completed: bool
        messages: StatusMsg

    def task_done(self, 
                  item_id, 
                  outputs,
                  status: Union['PromptQueue.ExecutionStatus', None]):
        with self.mutex:
            prompt = self.currently_running.pop(item_id)
            if len(self.history) > MAXIMUM_HISTORY_SIZE:
                self.history.pop(next(iter(self.history)))

            status_dict: Optional[dict] = None
            if status is not None:
                status_dict = copy.deepcopy(status._asdict())
                    
            self.history[prompt[1]] = {
                "prompt": prompt,
                "outputs": copy.deepcopy(outputs),
                'status': status_dict,
            }
            self.server.queue_updated()

    def get_current_queue(self):
        with self.mutex:
            out = []
            for x in self.currently_running.values():
                out += [x]
            return (out, copy.deepcopy(self.queue))

    def get_tasks_remaining(self):
        with self.mutex:
            return len(self.queue) + len(self.currently_running)

    def wipe_queue(self):
        with self.mutex:
            self.queue = []
            self.server.queue_updated()

    def delete_queue_item(self, function):
        with self.mutex:
            for x in range(len(self.queue)):
                if function(self.queue[x]):
                    if len(self.queue) == 1:
                        self.wipe_queue()
                    else:
                        self.queue.pop(x)
                        heapq.heapify(self.queue)
                    self.server.queue_updated()
                    return True
        return False

    def get_history(self, prompt_id=None, max_items=None, offset=-1):
        with self.mutex:
            if prompt_id is None:
                out = {}
                i = 0
                if offset < 0 and max_items is not None:
                    offset = len(self.history) - max_items
                for k in self.history:
                    if i >= offset:
                        out[k] = self.history[k]
                        if max_items is not None and len(out) >= max_items:
                            break
                    i += 1
                return out
            elif prompt_id in self.history:
                return {prompt_id: copy.deepcopy(self.history[prompt_id])}
            else:
                return {}

    def wipe_history(self):
        with self.mutex:
            self.history = {}

    def delete_history_item(self, id_to_delete):
        with self.mutex:
            self.history.pop(id_to_delete, None)

    def set_flag(self, name, data):
        with self.mutex:
            self.flags[name] = data
            self.not_empty.notify()

    def get_flags(self, reset=True):
        with self.mutex:
            if reset:
                ret = self.flags
                self.flags = {}
                return ret
            else:
                return self.flags.copy()
