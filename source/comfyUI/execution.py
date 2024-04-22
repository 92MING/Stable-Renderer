import sys
import copy
import logging
import threading
import heapq
import traceback
import inspect
import torch
import uuid

from deprecated import deprecated
from typing import (List, Literal, NamedTuple, Optional, TYPE_CHECKING, Tuple, Set, Dict, Union, Any, 
                    Type)

import nodes
import comfy.model_management

from common_utils.debug_utils import ComfyUILogger, get_log_level_by_name
from common_utils.decorators import singleton, class_property, Overload
from comfyUI.types import *
from comfyUI.adapters import find_adapter

if TYPE_CHECKING:
    from comfyUI.server import PromptServer

def _get_comfy_node_input_type_name(node_type: Type[ComfyUINode], input_name: str)->str:
    if not isinstance(node_type, type):
        node_type = node_type.__class__
    input_param_dict = node_type.INPUT_TYPES()
    all_input_params = {}
    all_input_params.update(input_param_dict.get("required", {}))
    all_input_params.update(input_param_dict.get("optional", {}))
    all_input_params.update(input_param_dict.get("hidden", {}))
    if input_name not in all_input_params:
        raise ValueError(f"Input name {input_name} not found in node {node_type}.")
    return all_input_params[input_name][0]
    
def get_input_data(inputs: dict, 
                   class_def: Type[ComfyUINode], 
                   unique_id: str, 
                   outputs: Optional[dict]=None, 
                   prompt: Optional[dict]=None, 
                   extra_data: Optional[dict]=None):
    '''
    Pack inputs form value/nodes into a dict for the node to use.
    
    Args:
        - inputs: {param name, value}. Values can be:
            * [from_node_id, from_node_output_slot_index] (means the value is from another node's output)
            * a value
        - class_def: the class of the node
        - unique_id: the id of this node
        - outputs: {node_id: [output1, output2, ...], ...}
        - prompt: the prompt dict
    
    Returns:
        a dict of {param name: value}. !Values from node bindings are not resolved yet!
    '''
    outputs = outputs or {}
    prompt = prompt or {}
    extra_data = extra_data or {}
    valid_inputs = class_def.INPUT_TYPES()
    lazy_inputs = class_def.LAZY_INPUTS if hasattr(class_def, "LAZY_INPUTS") else []
    
    input_data_all = {}
    for input_param_name in inputs:
        input_data = inputs[input_param_name]
        
        if isinstance(input_data, NodeBindingParam):    # [from_node_id, from_node_output_slot_index]
            from_node_id = input_data[0]
            output_slot_index = input_data[1]
            
            if input_param_name in lazy_inputs:
                val = Lazy(from_node_id, output_slot_index, _get_comfy_node_input_type_name(class_def, input_param_name))
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
            elif hidden_cls := HIDDEN.FindHiddenClsByName(h[input_param_name]): # type: ignore
                node = NodePool().get(unique_id)
                context = PromptExecutor().current_context
                context.current_node = node # type: ignore
                input_data_all[input_param_name] = [hidden_cls.GetValue(context)] # type: ignore
            else:
                try_put_hidden_value(input_param_name)
    return input_data_all

@deprecated # changed name to `get_node_func_ret`
def map_node_over_list(obj, input_data_all, func, allow_interrupt=False):
    '''
    Name of `map_node_over_list` is misleading, i.e. it's actually a general function to get the return of a node's specific function.
    The name has now changed to `_get_node_func_ret`, so please use the new name instead.
    '''
    return get_node_func_ret(obj, input_data_all, func, allow_interrupt)

_node_ins_methods = {"IS_CHANGED", "ON_DESTROY"}

def get_node_func_ret(node: Union[str, ComfyUINode, Type[ComfyUINode]], 
                      func_params: dict,    # not `NodeInputs`, cuz values should already be resolved
                      func: Optional[str] = None,   # the target function name of the node
                      allow_interrupt: bool = False,
                      create_node_if_need: bool = True):
    '''return the result of the given function name of the node.'''
    if isinstance(node, str):   # node id
        node = NodePool().get(node, create_new=False)
        if not node:
            raise ValueError(f"Node {node} not found.")
    
    temp_node_ins = None
    if not isinstance(node, type):
        node_type = node.__class__
    else:
        node_type = node
    
    if not func:
        func = node_type.FUNCTION   # type: ignore
    if not hasattr(node_type, func):
        raise AttributeError(f"Node {node} has no function {func}.")
    
    # check if node wants the lists
    input_is_list = False
    if hasattr(node, "INPUT_IS_LIST"):
        input_is_list = node_type.INPUT_IS_LIST   # type: ignore
    # TODO: better treatment when node class comes from advance node class

    if len(func_params) == 0:
        max_len_input = 0
    else:
        max_len_input = 0
        for x in func_params.values():
            try:
                if len(x) > max_len_input:
                    max_len_input = len(x)
            except TypeError:
                pass
    
    # get a slice of inputs, repeat last input when list isn't long enough
    def slice_dict(d, i):
        d_new = dict()
        for k, v in d.items():
            if isinstance(v, ComfyUINode):
                d_new[k] = v
            else:
                try:
                    d_new[k] = v[i if len(v) > i else -1]
                except TypeError:
                    d_new[k] = v
        return d_new
    
    real_func = getattr(node_type, func)
    if not callable(real_func):
        raise AttributeError(f"Node type `{node_type}` has no function {func}.")
    
    if func in (*_node_ins_methods, node_type.FUNCTION):    # type: ignore
        params = inspect.signature(real_func).parameters
        first_param = list(params.keys())[0]
        if first_param not in func_params:
            if isinstance(node, type):
                if not create_node_if_need:
                    raise ValueError(f"Node {node_type} is not created.")
                temp_node_ins = node()
                temp_node_ins.ID = str(uuid.uuid4()).replace("-", "")
                func_params[first_param] = temp_node_ins
            else:
                func_params[first_param] = node
    
    results = []
    if input_is_list:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(real_func(**func_params))
    
    elif max_len_input == 0:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(real_func())
        
    else:   # target node accepts single val, but list is provided, so cut into slices
        for i in range(max_len_input):
            if allow_interrupt:
                nodes.before_node_execution()
            results.append(real_func(**slice_dict(func_params, i)))
            
    if temp_node_ins is not None:
        NodePool().pop(temp_node_ins.ID)
        temp_node_ins.ON_DESTROY() if hasattr(temp_node_ins, "ON_DESTROY") else None
    return results

@deprecated # should be a private function
def format_value(x):        
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)

@deprecated('`recursive_execute` method has been moved into `PromptExecutor`.')
def recursive_execute(server: "PromptServer", 
                      prompt: dict, 
                      outputs: dict, 
                      current_item: str,    # current node id 
                      extra_data: dict, 
                      executed: set,    # executed nodes' ids 
                      prompt_id: str,
                      outputs_ui: dict,     
                      object_storage: dict  # node pool, will not use now.
                      ):
    e = PromptExecutor()
    e.server = server
    context = e.current_context
    if not context: # no current context
        context = InferenceContext(prompt=prompt,   # type: ignore
                                   extra_data=extra_data,
                                   current_node_id=current_item,
                                   old_prompt=None,
                                   outputs=outputs or {},
                                   outputs_ui=outputs_ui or {},
                                   frame_data=None,
                                   baking_data=None,
                                   status_messages=[],
                                   executed_node_ids=executed or set(),
                                   success=False)
        return e.execute(context)

    elif context.prompt.id == prompt_id:   # is current context
        cur_output = copy.deepcopy(e.current_context.outputs) if e.current_context and e.current_context.outputs else {}
        cur_output.update(outputs)
        context.outputs = cur_output
        
        cur_output_ui = copy.deepcopy(e.current_context.outputs_ui) if e.current_context and e.current_context.outputs_ui else {}
        cur_output_ui.update(outputs_ui)
        context.outputs_ui = cur_output_ui
        
        context.extra_data.update(extra_data)
        context.current_node_id = current_item
        context.executed_node_ids.update(executed)
        e.current_context = context
        
        return e._recursive_execute(context)
    
    else:   # already some other context is running
        raise ValueError("Prompt id mismatch, some other prompt is running.")

        
@singleton(cross_module_singleton=True)
class PromptExecutor:
    '''The executor for running inference.'''
    
    server: Optional['PromptServer']
    '''The server that this executor is associated with. If None, the executor'''
    node_pool: NodePool = NodePool()
    '''
    Pool of all created nodes
    {node_id: node_instance, ...}
    '''
    
    # runtime data
    is_baking: bool = False
    '''whether the executor is in baking mode or not'''
    
    current_context: Optional[InferenceContext] = None
    '''current context of inference. It will be set when executing a node.'''
    last_context: Optional[InferenceContext] = None
    '''last context of inference. It will be set when executing a node.'''
    
    # region deprecated
    @property
    @deprecated(reason="access PromptExecutor's context directly instead.")
    def prompt(self)->Optional[PROMPT]:
        '''
        current prompt to execute.
        You should use `current_context.prompt` instead.
        '''
        return self.current_context.prompt if self.current_context else None
    
    @property
    @deprecated(reason="access PromptExecutor's context directly instead.")
    def extra_data(self)->Optional[dict]:
        '''
        extra data for current execution.
        You should use `current_context.extra_data` instead.
        '''
        return self.current_context.extra_data if self.current_context else None
    
    @property
    @deprecated(reason="access PromptExecutor's context directly instead.")
    def executed_node_ids(self)->Optional[Set[str]]:
        '''
        executed node ids in current execution.
        You should use `current_context.executed_node_ids` instead.
        '''
        return self.current_context.executed_node_ids if self.current_context else None
    
    @property
    @deprecated(reason="use `last_context.prompt` directly instead.")
    def old_prompt(self)->Optional[PROMPT]:
        '''
        last executed prompt(running).
        You should use `last_context.prompt` instead.
        '''
        return self.last_context.prompt if self.last_context else None
    
    @property
    @deprecated(reason="access PromptExecutor's context directly instead.")
    def object_storage(self)->NodePool:
        '''
        !!Deprecated!! 
        This is actually a dictionary containing all node instance. Since NodePool is now a universal singleton,
        You should get the pool by getting the singleton instance directly instead.
        '''
        return self.node_pool
    
    @property
    @deprecated(reason="access PromptExecutor's context directly instead.")
    def outputs_ui(self)-> NodeOutputs_UI:
        '''
        !!Deprecated!!
        Latest UI outputs from nodes(running/executed).
        You should use `current_context.outputs_ui` instead.
        '''
        return self.current_context.outputs_ui if self.current_context else {}
    
    @property
    @deprecated(reason="access PromptExecutor's context directly instead.")
    def status_messages(self)->StatusMsgs:
        '''
        !!Deprecated!!
        latest status messages in current execution.
        You should use `current_context.status_messages` instead.
        '''
        return self.current_context.status_messages if self.current_context else []
    
    @property
    @deprecated(reason="access PromptExecutor's context directly instead.")
    def outputs(self)-> NodeOutputs:
        '''
        !!Deprecated!!
        Latest outputs from nodes(running/executed).
        You should use `current_context.outputs` instead.
        '''
        return self.current_context.outputs if self.current_context else {}
    
    @property
    @deprecated(reason="use `current_context.success` directly instead.")
    def success(self)-> bool:
        '''
        !!Deprecated!!
        Whether the latest execution is successful or not.
        You should use `current_context.success` instead.
        '''
        return self.current_context.success if self.current_context else False
    # endregion
    
    @class_property # type: ignore
    def Instance(cls)->'PromptExecutor':
        return cls()    # type: ignore
    
    def __init__(self, server: Optional['PromptServer']=None):
        self.server = server
        self.reset()
    
    def _format_value(self, x):        
        if x is None:
            return None
        elif isinstance(x, (int, float, bool, str)):
            return x
        else:
            return str(x)
    
    def _get_output_data(self, context: InferenceContext, input_data_all: dict, save_to_current_output=True)->Tuple[List[Any], Dict[str, Any]]:
        '''
        Get the output data of the given node.
        
        Return:
            - outputs (list of list of values, e.g. [[val1, val2], [val3, val4], ...])
            - ui values (values to be shown on UI, e.g. {key: [val1, val2, ...], ...})
        '''
        node = context.current_node
        if not node:
            raise ValueError("No node to execute for `_get_output_data`.")
        results = []
        uis = {}
        return_values = get_node_func_ret(node, input_data_all, node.FUNCTION, allow_interrupt=True)

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
        
        if save_to_current_output:
            context.outputs[node.ID] = output
            context.outputs_ui[node.ID] = uis
        return output, uis
        
    def _recursive_will_execute(self, context:"InferenceContext", current_node_id: Optional[str]=None, memo:Optional[dict]=None):
        memo = memo or {}
        outputs = context.outputs
        unique_id = current_node_id or context.current_node_id
        if not unique_id:
            raise ValueError("No node id to execute.")
        context.current_node_id = unique_id # set back when `current_node_id` is passed by parameter
        prompt = context.prompt
        
        if unique_id in memo:
            return memo[unique_id]

        inputs = prompt[unique_id]['inputs']
        will_execute = []
        if unique_id in outputs:
            return []

        for input_param_name in inputs:
            input_data = inputs[input_param_name]
            if isinstance(input_data, NodeBindingParam):
                input_node_unique_id = input_data[0]
                if input_node_unique_id not in outputs:
                    context.current_node_id = input_node_unique_id
                    will_execute += self._recursive_will_execute(context, memo=memo)

        memo[unique_id] = will_execute + [unique_id]
        return memo[unique_id]
    
    def _get_node_output(self, node_id: str, slot_index: int, to_type_name: str):
        '''special method for "Lazy" type values, to get the real value from a binding'''
        context = self.current_context
        if not context:
            raise ValueError("No context to get node output.")
        
        node = self.node_pool.get(node_id, create_new=False)
        if not node:
            raise ValueError(f"Node with id `{node_id}` not found.")
        from_val_type = node.RETURN_TYPES[slot_index]
        
        if hasattr(self, 'outputs') and self.outputs and node_id in self.outputs:
            val = self.outputs[node_id][slot_index]
        else:
            context.current_node_id = node_id
            self._recursive_execute(context)    # continue execution
            val = self.outputs[node_id][slot_index]
            
        if adapter := find_adapter(from_val_type, to_type_name):
            val = adapter(val)
        return val
    
    def _recursive_execute(self, context: InferenceContext)->RecursiveExecuteResult:
        prompt = context.prompt
        if not prompt:
            return RecursiveExecuteResult(False, None, ValueError("No prompt to execute."))
        
        extra_data = context.extra_data
        executed = context.executed_node_ids
        if not context.current_node_id:
            return RecursiveExecuteResult(False, None, ValueError("No node specify to execute in the context."))
        
        current_node_id = context.current_node_id
        inputs = prompt[current_node_id]['inputs']
        class_type = prompt[current_node_id]['class_type']
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        
        if current_node_id in context.outputs:
            return RecursiveExecuteResult(True, None, None)
        
        for param_name in inputs:
            input_data = inputs[param_name]

            if isinstance(input_data, NodeBindingParam):
                node_id = input_data[0]
                if node_id not in context.outputs:
                    context.current_node_id = node_id
                    result = self._recursive_execute(context)
                    if result.success is not True:
                        return result # Another node failed further upstream

        input_data_all = None
        try:
            input_data_all = get_input_data(inputs,
                                             class_def, 
                                             current_node_id,
                                             context.outputs, 
                                             prompt, 
                                             extra_data)
            if self.server:
                if self.server.client_id is not None:
                    self.server.last_node_id = current_node_id
                    self.server.send_sync("executing", 
                                          {"node": current_node_id, "prompt_id": prompt.id if prompt else None}, 
                                          self.server.client_id)
            
            context.current_node_id = current_node_id
            self._get_output_data(context, input_data_all, save_to_current_output=True)
            if len(context.outputs_ui) > 0:
                if self.server:
                    if self.server.client_id is not None:
                        self.server.send_sync("executed", 
                                              { "node": current_node_id, "output": context.outputs_ui, "prompt_id": prompt.id}, 
                                              self.server.client_id)
        
        except comfy.model_management.InterruptProcessingException as iex:
            ComfyUILogger.info("Processing interrupted")

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
                    try:
                        input_data_formatted[name] = [self._format_value(x) for x in inputs]
                    except TypeError:
                        input_data_formatted[name] = self._format_value(inputs)
                        
            output_data_formatted = {}
            for node_id, node_outputs in context.outputs.items():
                output_data_formatted[node_id] = [[self._format_value(x) for x in l] for l in node_outputs]

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

    def _recursive_output_delete_if_changed(self, context: InferenceContext)->bool:
        if not context.current_node_id:
            raise ValueError("No node to execute in the context.")
        unique_id = context.current_node_id
        prompt = context.prompt
        old_prompt = context.old_prompt
        if not old_prompt or old_prompt.id == prompt.id:
            return True
        
        outputs = context.outputs
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
                input_data_all = get_input_data(inputs, class_def, unique_id, outputs)
                if input_data_all is not None:
                    try:
                        is_changed = get_node_func_ret(class_def, input_data_all, "IS_CHANGED")
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
                        if input_unique_id in outputs:
                            context.current_node_id = input_unique_id
                            to_delete = self._recursive_output_delete_if_changed(context)
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
        self.node_pool.clear()
        if self.last_context:
            self.last_context.destroy()
            self.last_context = None
        if self.current_context:
            self.current_context.destroy()
            self.current_context = None
    
    def get_node(self, node_id:str, class_type_name: str)->ComfyUINode:
        '''Return the target node from the pool, or create a new one if not found.'''
        return self.node_pool.get((node_id, class_type_name))
    
    def add_message(self, 
                    event: StatusMsgEvent, 
                    data: dict, 
                    broadcast: bool, 
                    level: Literal['debug', 'info', 'warning', 'error', 'critical', 'success']='info'):
        if not self.current_context:
            raise ValueError("No context to add message.")
        self.current_context.status_messages.append((event, data))
        if self.server:
            if self.server.client_id is not None or broadcast:
                self.server.send_sync(event, data, self.server.client_id)
        else:
            ComfyUILogger.log(get_log_level_by_name(level), f"Comfy Execution Msg: {event}, {data}")

    @Overload
    @deprecated
    def handle_execution_error(self, prompt_id: str, prompt: dict, current_outputs, executed, error, ex):   # type: ignore
        '''!Deprecated! Use `handle_execution_error(context, error, ex)` instead.'''
        context = self.current_context
        if not context:
            raise ValueError("No context to handle error.")
        if context.prompt.id != prompt_id:
            raise ValueError("Prompt id mismatch.")
        context.outputs.update(current_outputs)
        return self.handle_execution_error(context, error, ex)
        
    @Overload
    def handle_execution_error(self, context: InferenceContext, error, ex):
        prompt = context.prompt
        prompt_id = prompt.id
        
        executed = context.executed_node_ids
        current_outputs = context.outputs
        old_outputs = self.last_context.outputs if self.last_context else {}
        
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
            self.add_message("execution_interrupted", mes, broadcast=True, level="warning")
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
            self.add_message("execution_error", mes, broadcast=False, level="error")
        
        # Next, remove the subsequent outputs since they will not be executed
        to_delete = []
        for o in old_outputs:
            if (o not in current_outputs) and (o not in executed):
                to_delete += [o]
                if prompt and o in prompt:
                    d = prompt.pop(o)
                    del d
        for o in to_delete:
            d = self.outputs.pop(o)
            del d

    @Overload
    def execute(self, context: InferenceContext)->InferenceContext:     # type: ignore
        all_node_ids = list(context.prompt.keys())
        return self.execute(prompt=context.prompt, 
                            prompt_id=context.prompt_id, 
                            extra_data=context.extra_data, 
                            node_ids_to_be_ran=all_node_ids,    # type: ignore
                            frame_data=context.frame_data,
                            baking_data=context.baking_data)
        
    @Overload
    def execute(self, 
                prompt: Union[PROMPT, dict], 
                prompt_id: Optional[str]=None, # random string by uuid4 
                extra_data: Optional[dict]=None, 
                node_ids_to_be_ran: Union[List[str], List[int], None]=None,
                frame_data: Optional['FrameData']= None,
                baking_data: Optional[BakingData]=None)->InferenceContext:
        '''The entry for inference.'''
        extra_data = extra_data or {}
        node_ids_to_be_ran = [str(x) for x in node_ids_to_be_ran] if node_ids_to_be_ran else []
        executed_outputs = copy.deepcopy(self.last_context.outputs) if self.last_context else {}
        
        if prompt_id is None:
            prompt_id = str(uuid.uuid4())
        if not isinstance(prompt, PROMPT):
            prompt = PROMPT(prompt, id=prompt_id)
        
        current_context = InferenceContext(prompt=prompt, 
                                           extra_data=extra_data, 
                                           current_node_id=None,
                                           old_prompt=self.last_context.prompt if self.last_context else None,
                                           outputs=executed_outputs,
                                           outputs_ui=self.last_context.outputs_ui if self.last_context else {},
                                           frame_data=frame_data,
                                           baking_data=baking_data,
                                           status_messages=[],
                                           executed_node_ids=set(),
                                           success=False)
        self.current_context = current_context
        
        if self.server:
            if "client_id" in extra_data:
                self.server.client_id = extra_data["client_id"]
            else:
                self.server.client_id = None
        
        nodes.interrupt_processing(False)
        
        self.add_message("execution_start", { "prompt_id": prompt_id}, broadcast=False, level="info")

        with torch.inference_mode():
            #delete cached outputs if nodes don't exist for them
            to_delete = []
            for node_pool_key in current_context.outputs:
                if node_pool_key not in prompt:
                    to_delete += [node_pool_key]
            for node_pool_key in to_delete:
                d = current_context.outputs.pop(node_pool_key)
                del d
            
            nodes_to_be_deleted: List[Tuple[str, str]] = []
            for node_id, node_type in self.node_pool:
                if node_id not in prompt:
                    nodes_to_be_deleted += [(node_id, node_type)]
                else:
                    p = prompt[node_id]
                    if node_type != p['class_type']:
                        nodes_to_be_deleted += [(node_id, node_type)]
            
            for (node_id, node_type) in nodes_to_be_deleted:
                d = self.node_pool.pop((node_id, node_type))
                del d

            for node_id in prompt:
                current_context.current_node_id = node_id
                self._recursive_output_delete_if_changed(current_context)

            current_outputs = set(current_context.outputs.keys())
            for x in list(current_context.outputs_ui.keys()):
                if x not in current_outputs:
                    d = current_context.outputs_ui.pop(x)
                    del d

            comfy.model_management.cleanup_models()
            self.add_message(event="execution_cached", 
                             data={"nodes": list(current_outputs), "prompt_id": prompt_id}, 
                             broadcast=False,
                             level="info")
            output_node_id = None
            to_execute = []

            for node_id in list(node_ids_to_be_ran):
                to_execute += [(0, node_id)]

            while len(to_execute) > 0:
                #always execute the output that depends on the least amount of unexecuted nodes first
                memo = {}
                to_execute = sorted(list(map(lambda a: (len(self._recursive_will_execute(current_context, a[-1], memo)), a[-1]), to_execute)))
                output_node_id = to_execute.pop(0)[-1]
                current_context.current_node_id = output_node_id
                
                # This call shouldn't raise anything if there's an error deep in
                # the actual SD code, instead it will report the node where the
                # error was raised
                success, error, ex = self._recursive_execute(current_context)
                current_context.success = success
                if success is not True:
                    self.handle_execution_error(current_context, error, ex)
                    break

            for x in current_context.executed_node_ids:
                prompt[x] = copy.deepcopy(prompt[x])
                
            if self.server:
                self.server.last_node_id = None
            
            if comfy.model_management.DISABLE_SMART_MEMORY:
                comfy.model_management.unload_all_models()
        
        if self.last_context:
            self.last_context.destroy()
        self.last_context = current_context
        self.last_context._current_node_id = None
        self.current_context = None
        return self.last_context

def validate_inputs(prompt: Union[dict, PROMPT], node_id: str, validated: Dict[str, ValidateInputsResult])->ValidateInputsResult:
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
                    val = int(val)  # type: ignore
                    inputs[x] = val
                elif type_input == "FLOAT":
                    val = float(val)    # type: ignore
                    inputs[x] = val
                elif type_input == "STRING":
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
                    if val not in type_input:   # type: ignore
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
        input_data_all = get_input_data(inputs, obj_class, node_id)
        input_filtered = {}
        for x in input_data_all:
            if x in validate_function_inputs:
                input_filtered[x] = input_data_all[x]

        ret = get_node_func_ret(obj_class, input_filtered, "VALIDATE_INPUTS")
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
 
def validate_prompt(prompt: Union[dict, PROMPT], prompt_id: Optional[str]=None) -> ValidatePromptResult:
    if prompt_id is not None:
        if isinstance(prompt, PROMPT) and prompt_id != prompt.id:
            raise ValueError(f"Prompt id mismatch, got {prompt_id}, expected {prompt.id}")
    else:
        if not isinstance(prompt, PROMPT):
            raise ValueError("Prompt id is required when prompt is a normal dictionary instead of `PROMPT` type")
        prompt_id = prompt.id
    
    outputs = set()
    for node_id in prompt:
        node_class = nodes.NODE_CLASS_MAPPINGS[prompt[node_id]['class_type']]
        if hasattr(node_class, 'OUTPUT_NODE') and node_class.OUTPUT_NODE == True:
            outputs.add(node_id)

    if len(outputs) == 0:
        error = {
            "type": "prompt_no_outputs",
            "message": "Prompt has no outputs",
            "details": "",
            "extra_info": {}
        }
        return ValidatePromptResult(result=False, errors=error, good_outputs=[], node_errors={}, _prompt=prompt, _prompt_id=prompt_id)

    good_outputs = set()
    errors = []
    node_errors = {}
    validated = {}
    for node_id in outputs:
        valid = False
        reasons = []
        try:
            m = validate_inputs(prompt, node_id, validated)
            valid = m[0]
            reasons: list = m[1]    # type: ignore
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
            validated[node_id] = (False, reasons, node_id)

        if valid is True:
            good_outputs.add(node_id)
        else:
            logging.error(f"Failed to validate prompt for output {node_id}:")
            if len(reasons) > 0:
                logging.error("* (prompt):")
                for reason in reasons:
                    logging.error(f"  - {reason['message']}: {reason['details']}")
            errors += [(node_id, reasons)]
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
                    node_errors[node_id]["dependent_outputs"].append(node_id)
            logging.error("Output will be ignored")

    if len(good_outputs) == 0:
        errors_list = []
        for node_id, errors in errors:
            for error in errors:
                errors_list.append(f"{error['message']}: {error['details']}")
        errors_list = "\n".join(errors_list)

        error = {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": errors_list,
            "extra_info": {}
        }

        return ValidatePromptResult(result=False, 
                                    errors=error, 
                                    good_outputs=list(good_outputs), 
                                    node_errors=node_errors, 
                                    _prompt=prompt,
                                    _prompt_id=prompt_id)

    return ValidatePromptResult(result=True, 
                                errors=None, 
                                good_outputs=list(good_outputs), 
                                node_errors=node_errors, 
                                _prompt=prompt,
                                _prompt_id=prompt_id)

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

    def get(self, timeout=None)->Optional[Tuple[QueueTask, int]]:
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
        messages: StatusMsgs

    def task_done(self, 
                  item_id, 
                  outputs_ui,
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
                "outputs": copy.deepcopy(outputs_ui),
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
