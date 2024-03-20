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

import nodes
import comfy.model_management
from common_utils.decorators import singleton
from comfyUI.types import *

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
        - inputs: it can be:
            * [from_node_id, from_node_output_slot_index]
            * a value
        - class_def: the class of the node
        - unique_id: the id of this node
        
    '''
    valid_inputs = class_def.INPUT_TYPES()
    lazy_inputs = class_def.LAZY_INPUTS if hasattr(class_def, "LAZY_INPUTS") else []
    reduce_inputs = class_def.REDUCE_INPUTS if hasattr(class_def, "REDUCE_INPUTS") else []
    
    input_data_all = {}
    for node_id in inputs:
        input_data = inputs[node_id]
        
        if isinstance(input_data, list):    # [from_node_id, from_node_output_slot_index]
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                input_data_all[node_id] = (None,)
                continue
            obj = outputs[input_unique_id][output_index]
            input_data_all[node_id] = obj
            
        else:   # a value
            if ("required" in valid_inputs and node_id in valid_inputs["required"]) or ("optional" in valid_inputs and node_id in valid_inputs["optional"]):
                input_data_all[node_id] = [input_data]

    if "hidden" in valid_inputs:
        h = valid_inputs["hidden"]
        for node_id in h:
            if h[node_id] == "PROMPT":
                input_data_all[node_id] = [prompt]
            if h[node_id] == "EXTRA_PNGINFO":
                if "extra_pnginfo" in extra_data:
                    input_data_all[node_id] = [extra_data['extra_pnginfo']]
            if h[node_id] == "UNIQUE_ID":
                input_data_all[node_id] = [unique_id]
    return input_data_all

def _map_node_over_list(node: Union[ComfyUINode, Type[ComfyUINode]], 
                       input_data_all, 
                       func: str,   # the target function name of the node
                       allow_interrupt=False):
    # check if node wants the lists
    input_is_list = False
    if hasattr(node, "INPUT_IS_LIST"):
        input_is_list = node.INPUT_IS_LIST

    if len(input_data_all) == 0:
        max_len_input = 0
    else:
        max_len_input = max([len(x) for x in input_data_all.values()])
    
    # get a slice of inputs, repeat last input when list isn't long enough
    def slice_dict(d, i):
        d_new = dict()
        for k,v in d.items():
            d_new[k] = v[i if len(v) > i else -1]
        return d_new
    
    results = []
    if input_is_list:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(getattr(node, func)(**input_data_all))
    elif max_len_input == 0:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(getattr(node, func)())
    else:
        for i in range(max_len_input):
            if allow_interrupt:
                nodes.before_node_execution()
            results.append(getattr(node, func)(**slice_dict(input_data_all, i)))
    return results

def _format_value(x):
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)


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
        return_values = _map_node_over_list(node, 
                                            input_data_all, 
                                            node.FUNCTION, 
                                            allow_interrupt=True)

        for r in return_values:
            if isinstance(r, dict):
                if 'ui' in r:
                    uis.update(r['ui'])
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
        
    def _recursive_execute(self,
                        prompt: PROMPT, 
                        current_node_id: str, 
                        extra_data, 
                        executed: Set[str],   # set of node ids that have been executed 
                        prompt_id: str,   # just a random string by uuid4
                        ):
        unique_id = current_node_id
        inputs = prompt[unique_id]['inputs']
        class_type = prompt[unique_id]['class_type']
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        
        if unique_id in self.outputs:
            return (True, None, None)
        
        for x in inputs:
            input_data = inputs[x]

            if isinstance(input_data, list):
                input_unique_id = input_data[0]
                output_index = input_data[1]
                if input_unique_id not in self.outputs:
                    result = self._recursive_execute(prompt, 
                                                    input_unique_id, 
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
                                            unique_id,
                                            self.outputs, 
                                            prompt, 
                                            extra_data)
            if self.server:
                if self.server.client_id is not None:
                    self.server.last_node_id = unique_id
                    self.server.send_sync("executing",
                                        { "node": unique_id, "prompt_id": prompt_id }, 
                                        self.server.client_id)

            node = self.get_node(unique_id, class_type)

            output_data, output_ui = self._get_output_data(node, input_data_all)
            self.outputs[unique_id] = output_data
            if len(output_ui) > 0:
                self.outputs_ui[unique_id] = output_ui
                if self.server:
                    if self.server.client_id is not None:
                        self.server.send_sync("executed", 
                                            { "node": unique_id, "output": output_ui, "prompt_id": prompt_id }, 
                                            self.server.client_id)
        
        except comfy.model_management.InterruptProcessingException as iex:
            logging.info("Processing interrupted")

            # skip formatting inputs/outputs
            error_details = {
                "node_id": unique_id,
            }

            return (False, error_details, iex)
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
                "node_id": unique_id,
                "exception_message": str(ex),
                "exception_type": exception_type,
                "traceback": traceback.format_tb(tb),
                "current_inputs": input_data_formatted,
                "current_outputs": output_data_formatted
            }
            return (False, error_details, ex)

        executed.add(unique_id)

        return (True, None, None)

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
                        is_changed = _map_node_over_list(class_def, input_data_all, "IS_CHANGED")
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

                    if isinstance(input_data, list):
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
            for o in self.outputs:
                if o not in prompt:
                    to_delete += [o]
            for o in to_delete:
                d = self.outputs.pop(o)
                del d
            to_delete = []
            for o in self.node_pool:
                if o[0] not in prompt:
                    to_delete += [o]
                else:
                    p = prompt[o[0]]
                    if o[1] != p['class_type']:
                        to_delete += [o]
            for o in to_delete:
                d = self.node_pool.pop(o)
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



def validate_inputs(prompt, item, validated):
    unique_id = item
    if unique_id in validated:
        return validated[unique_id]

    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    obj_class = nodes.NODE_CLASS_MAPPINGS[class_type]

    class_inputs = obj_class.INPUT_TYPES()
    required_inputs = class_inputs['required']

    errors = []
    valid = True

    validate_function_inputs = []
    if hasattr(obj_class, "VALIDATE_INPUTS"):
        validate_function_inputs = inspect.getfullargspec(obj_class.VALIDATE_INPUTS).args

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
        if isinstance(val, list):
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
            if r[val[1]] != type_input:
                received_type = r[val[1]]
                details = f"{x}, {received_type} != {type_input}"
                error = {
                    "type": "return_type_mismatch",
                    "message": "Return type mismatch between linked nodes",
                    "details": details,
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_type": received_type,
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
                validated[o_id] = (False, reasons, o_id)
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
        input_data_all = _get_input_data(inputs, obj_class, unique_id)
        input_filtered = {}
        for x in input_data_all:
            if x in validate_function_inputs:
                input_filtered[x] = input_data_all[x]

        #ret = obj_class.VALIDATE_INPUTS(**input_filtered)
        ret = _map_node_over_list(obj_class, input_filtered, "VALIDATE_INPUTS")
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
        ret = (False, errors, unique_id)
    else:
        ret = (True, [], unique_id)

    validated[unique_id] = ret
    return ret

def _get_full_type_name(klass):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__

def validate_prompt(prompt: PROMPT):
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
        return (False, error, [], [])

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

        return (False, error, list(good_outputs), node_errors)

    return (True, None, list(good_outputs), node_errors)

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