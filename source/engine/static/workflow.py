'''ComfyUI's Workflow class, for loading to engine.'''
import os
if __name__ == '__main__':  # for debugging
    import sys
    _proj_path = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    sys.path.append(_proj_path)
    __package__ = 'engine.static'
    
    from common_utils.path_utils import SOURCE_DIR

    if str(SOURCE_DIR.absolute()) not in sys.path:
        sys.path.insert(0, str(SOURCE_DIR.absolute()))

    _COMFYUI_PATH = str((SOURCE_DIR / 'comfyUI').absolute())
    if _COMFYUI_PATH not in sys.path:
        sys.path.insert(0, _COMFYUI_PATH)

import json
from typing import Tuple, Type, Union, Dict, List, Any, TYPE_CHECKING, Optional, Tuple, ClassVar, overload
from pathlib import Path
from dataclasses import dataclass, field
from uuid import uuid4

from common_utils.debug_utils import EngineLogger
from common_utils.global_utils import is_dev_mode, is_verbose_mode
from common_utils.path_utils import BUILTIN_WORKFLOW_DIR
from comfyUI.types._utils import get_comfy_name

if TYPE_CHECKING:
    from comfyUI.types import ComfyUINode, NodeBindingParam, PROMPT


def _get_comfy_node_type(type_name: str):
    from comfyUI.types import get_node_cls_by_name
    if not (t := get_node_cls_by_name(type_name, init_nodes_if_not_yet=True)):
        raise ValueError(f'Cannot find the type {type_name}.')
    return t

class WorkflowNodeLink(Tuple[int, int, int, int, int, str]):
    '''The link between two nodes.'''
    
    @property
    def id(self)->int:
        '''id of the link.'''
        return self[0]
    
    @property
    def from_node_id(self)->str:
        '''The id of the node that the link comes from.'''
        return str(self[1])
    
    @property
    def to_node_id(self)->str:
        '''The id of the node that the link goes to.'''
        return str(self[3])
    
    @property
    def from_output_slot(self)->int:
        '''The output slot of the from node.'''
        return self[2]
    
    @property
    def to_input_slot(self)->int:
        '''The input slot of the to node.'''
        return self[4]
    
    @property
    def val_type(self)->str:
        '''The type name of the value transfer.'''
        return self[5]
    
    def to_node_binding_param(self)->'NodeBindingParam':
        from comfyUI.types import NodeBindingParam
        return NodeBindingParam([self.from_node_id, self.from_output_slot])

@dataclass
class WorkflowNodeInputParam:
    '''The input parameter of a node.'''
    
    name: str
    '''the name of the input parameter.'''
    value: Union[Any, 'NodeBindingParam']
    '''the value of the input parameter. It can be a constant value or a link to an output parameter from another node.'''
    type_name: str
    '''the type name of the input parameter.'''

@dataclass
class WorkflowNodeOutputParam:
    '''The output parameter of a node.'''
    name: str
    '''the name of the output parameter.'''
    type_name: str
    '''the output type's name.'''
    slot: int
    '''the slot of the output parameter.'''
    to_nodes: List[str] = field(default_factory=list)
    '''ids of the nodes that the output parameter links to.'''

class InvalidNodeError(Exception): ...

class WorkflowNodeInfo(Dict[str, Any]):
    
    _cls_type: Type['ComfyUINode']
    '''the type of the node.'''
    _inputs: Dict[str, WorkflowNodeInputParam]
    '''the input parameters of the node.'''
    _outputs: List[WorkflowNodeOutputParam]
    '''the output parameters of the node.'''
    origin_data: dict
    '''the original data of the node.'''
    workflow: "Workflow"
    '''the parent workflow that the node belongs to.'''
    
    def __init__(self, origin_data: dict, workflow: "Workflow"):
        '''
        Args:
            - origin_data: the original data of the node.
            - workflow: the parent workflow that the node belongs to.
        '''
        super().__init__(origin_data)
        links = workflow.node_links
        self.workflow = workflow
        self.origin_data = origin_data
        self['id'] = str(self['id'])
        self._cls_type = _get_comfy_node_type(self['type'])
        
        # init inputs
        widget_values:Union[list, Dict[str, Any]] = self.get('widgets_values', [])
        '''The values of the widgets. It can be a list of values or a dict of values.'''
        widget_kw_values: Dict[str, Any] = self.get('widget_kw_values', {})
        '''The values of the widget keyword arguments.'''
        
        workflow_inputs = self.get('inputs', [])    # original `inputs` is a list of dict for linked inputs.
        workflow_inputs = {i['name']: i for i in workflow_inputs}
        formatted_inputs:Dict[str, WorkflowNodeInputParam] = {}   # name, param
        
        node_cls = self.cls_type
        node_cls_input_types = node_cls.INPUT_TYPES()
        required_input_names = tuple((node_cls_input_types.get('required', {})).keys())
        optional_input_names = tuple((node_cls_input_types.get('optional', {})).keys())
        # no need to add hidden input types.
        
        def get_input_default_value(input_name:str):
            if input_name in required_input_names:
                input_info = node_cls_input_types['required'][input_name]
                if len(input_info)>=2:
                    return input_info[1].get('default', None)
                return None
            elif input_name in optional_input_names:
                input_info = node_cls_input_types['optional'][input_name]
                if len(input_info)>=2:
                    return input_info[1].get('default', None)
                return None
            return None
            
        def get_input_value_type(input_name:str):
            if input_name in required_input_names:
                return node_cls_input_types['required'][input_name][0]
            elif input_name in optional_input_names:
                return node_cls_input_types['optional'][input_name][0]
            return None
        
        for input_name in (required_input_names + optional_input_names):
            if input_name in workflow_inputs:   # input is a binding(means from another node's output)
                input_info = workflow_inputs.pop(input_name)
                if not (link_id := input_info.get('link')):
                    if (default_value := get_input_default_value(input_name)) is not None:
                        formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                              default_value, 
                                                                              node_cls_input_types['required'][input_name][0])
                    elif input_name in optional_input_names:
                        formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                              None, 
                                                                              node_cls_input_types['optional'][input_name][0])
                    else:
                        raise InvalidNodeError(f'Cannot find the link id for input {input_name}.')
                else:
                    link = links[link_id]
                    param_type = 'required' if input_name in required_input_names else 'optional'
                    formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                          link.to_node_binding_param(),
                                                                          node_cls_input_types[param_type][input_name][0])
            
            else:   # input is a constant value
                widget_value_len = len(widget_kw_values) if self.workflow.is_stable_renderer_workflow else len(widget_values)
                if widget_value_len == 0:
                    if (default_value := get_input_default_value(input_name)) is not None:
                        formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                              default_value, 
                                                                              node_cls_input_types['required'][input_name][0])
                    elif input_name in optional_input_names:
                        formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                              None, 
                                                                              node_cls_input_types['optional'][input_name][0])
                    else:
                        raise ValueError(f'Cannot find the widget value for input {input_name}.')
                    
                else:
                    if self.workflow.is_stable_renderer_workflow:
                        if not (widget_value := widget_kw_values.get(input_name)):
                            if input_name in required_input_names:
                                if (default_value := get_input_default_value(input_name)) is not None:
                                    formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                                          default_value, 
                                                                                          node_cls_input_types['required'][input_name][0])
                                else:
                                    if is_dev_mode() and is_verbose_mode():
                                        EngineLogger.debug(f'Cannot find the widget value for input {input_name}. Skipped.')    
                                        # probably function has different params in `__call__` and `__server_call__`
                                                                                        
                            elif input_name in optional_input_names:
                                formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                                      None, 
                                                                                      node_cls_input_types['optional'][input_name][0])
                            else:
                                if is_dev_mode() and is_verbose_mode():
                                    EngineLogger.debug(f'Cannot find the widget value for input {input_name}. Skipped.')  # will not raise error in this case, cuz it's a stable renderer workflow.
                        else:
                            param_type = 'required' if input_name in required_input_names else 'optional'
                            formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                                  widget_value, 
                                                                                  node_cls_input_types[param_type][input_name][0])
                        del widget_kw_values[input_name]
                    
                    else:
                        if isinstance(widget_values, dict): # dict widget values, {input_name: value}
                            if not (widget_value := widget_values.get(input_name)):
                                if input_name in optional_input_names:
                                    formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                                          None, 
                                                                                          node_cls_input_types['optional'][input_name][0])
                                else:
                                    raise ValueError(f'Cannot find the widget value for input {input_name}.')
                                
                        else:   # list widget values (comfyUI's origin json data, which will probably results in error.)
                            value_type_name = None
                            input_type_name = get_input_value_type(input_name)
                            while value_type_name != input_type_name:
                                if len(widget_values) == 0:
                                    if (default_value := get_input_default_value(input_name)) is not None:
                                        formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                                              default_value, 
                                                                                              node_cls_input_types['required'][input_name][0])
                                        break
                                    elif input_name in optional_input_names:
                                        formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                                              None, 
                                                                                              node_cls_input_types['optional'][input_name][0])
                                        break
                                    else:
                                        raise ValueError(f'Cannot find the widget value for input {input_name}.')
                                
                                widget_value = widget_values.pop(0)
                                value_type_name = get_comfy_name(type(widget_value))
                                if value_type_name == 'FLOAT' and input_type_name == 'INT':
                                    value_type_name = 'INT'
                                    widget_value = int(widget_value)
                                    break
                                elif value_type_name == 'INT' and input_type_name == 'FLOAT':
                                    value_type_name = 'FLOAT'
                                    widget_value = float(widget_value)
                                    break
                                
                            param_type = 'required' if input_name in required_input_names else 'optional'
                            formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                                  widget_value, 
                                                                                  node_cls_input_types[param_type][input_name][0])
        self._inputs = formatted_inputs
        
        # init outputs
        formatted_outputs = []
        if 'outputs' in self:
            for i, output in enumerate(self['outputs']):
                output_name = output['name']
                output_type = output['type']
                output_links = output.get('links', [])
                if output_links:    # can be None if no nodes are linked to this output.
                    output_to_nodes = [str(i) for i in output_links]
                    formatted_outputs.append(WorkflowNodeOutputParam(name=output_name, 
                                                                     type_name=output_type, 
                                                                     slot=i, 
                                                                     to_nodes=output_to_nodes))        
        self._outputs = formatted_outputs
        
    @property
    def cls_type(self)->Type['ComfyUINode']:
        '''the type of the node. Same as `type` property.'''
        return self._cls_type
    
    @property
    def cls_type_name(self)->str:
        '''the name of the type of the node. Equivalent to self['type'].'''
        return self['type']
    
    @property
    def order(self)->Optional[int]:
        '''Not sure what is `order` in ComfyUI. Maybe the creation order of node.'''
        return self.get('order')
    
    @property
    def mode(self)->Optional[int]:
        '''Not sure what is `mode` in ComfyUI.'''
        return self.get('mode')
    
    @property
    def flags(self)->Optional[dict]:
        '''Not sure what is `flags` in ComfyUI.'''
        return self.get('flags')
    
    @property
    def properties(self)->Optional[dict]:
        '''Not sure what is `properties` in ComfyUI. Probably some attributes for showing on web UI.'''
        return self.get('properties')
    
    @property
    def inputs(self)->Dict[str, WorkflowNodeInputParam]:
        '''
        The input parameters of the node.
        {param_name, value(binding or constant).
        '''
        return self._inputs
    
    @property
    def outputs(self)->List[WorkflowNodeOutputParam]:
        '''outputs information of the node.'''
        return self._outputs
    
    @property
    def id(self)->str:
        '''the unique id of the node.'''
        return self['id']
    
    @property
    def title(self)->Optional[str]:
        '''the title of the node.'''
        return self.get('title')
    
    @staticmethod
    def _ParseNodeInfos(infos: List[Dict], workflow: "Workflow")->'Dict[str, WorkflowNodeInfo]':
        datas: Dict[str, WorkflowNodeInfo] = {}
        invalid_node_ids = set()
        for info in infos:
            info['id'] = str(info['id'])
            try:
                datas[info['id']] = WorkflowNodeInfo(info, workflow)
            except InvalidNodeError as e:
                # the node's input is not complete, means this node is not a valid node
                # e.g a useless node but forgot to delete it. So we just skip it.
                if 'type' in info:
                    node_type = info['type']
                    EngineLogger.warning(f'Node {info["id"]}({node_type}) is not a valid node, reason: {str(e)}. Skipped.')
                else:
                    EngineLogger.warning(f'Node {info["id"]} is not a valid node, reason: {str(e)}. Skipped.')
                try:
                    del datas[info['id']]
                except KeyError:
                    pass
                invalid_node_ids.add(info['id'])
        
        from comfyUI.types import NodeBindingParam
        has_add_new_invalid = True
        while has_add_new_invalid:
            need_break = False
            for node_id in datas.keys():
                data = datas[node_id]
                for input_param in data.inputs.values():
                    if isinstance(input_param.value, NodeBindingParam):
                        if input_param.value[0] in invalid_node_ids:
                            EngineLogger.warning(f'Node {node_id}({data.cls_type_name}) has an invalid input link to node {input_param.value[0]}, skipped.')
                            del datas[node_id]
                            invalid_node_ids.add(node_id)
                            has_add_new_invalid = True
                            need_break = True
                            break
                if need_break:
                    break
                else:
                    for output_param in data.outputs:
                        if any([to_node_id in invalid_node_ids for to_node_id in output_param.to_nodes]):
                            output_param.to_nodes = [to_node_id for to_node_id in output_param.to_nodes if to_node_id not in invalid_node_ids]
                    if need_break:
                        break
            else:
                has_add_new_invalid = False
        return datas

class Workflow(Dict[str, Any]):
    '''
    The workflow for ComfyUI.
    Each workflow represents a rendering process/pipeline.
    '''
    
    _DefaultGameWorkflow: ClassVar[Optional['Workflow']] = None
    @staticmethod
    def DefaultGameWorkflow():
        '''workflow for default rendering process.'''
        if not Workflow._DefaultGameWorkflow:
            default_game_workflow_path = BUILTIN_WORKFLOW_DIR / 'default_game_workflow.json'
            Workflow._DefaultGameWorkflow = Workflow.Load(default_game_workflow_path)
            Workflow._DefaultGameWorkflow['name'] = 'Default Game Workflow'
        return Workflow._DefaultGameWorkflow
    
    _DefaultBakeWorkflow: ClassVar[Optional['Workflow']] = None
    @staticmethod
    def DefaultBakeWorkflow():
        '''workflow for baking process.'''
        if not Workflow._DefaultBakeWorkflow:
            default_bake_workflow_path = BUILTIN_WORKFLOW_DIR / 'default_bake_workflow.json'
            Workflow._DefaultBakeWorkflow = Workflow.Load(default_bake_workflow_path)
            Workflow._DefaultBakeWorkflow['name'] = 'Default Bake Workflow'
        return Workflow._DefaultBakeWorkflow
    
    original_data: dict
    '''the original dict data of the workflow.'''

    @property
    def name(self)->Optional[str]:
        '''Get the name of the workflow. This is only for printing on log to debug'''
        return self.get('name', None)

    @property
    def last_node_id(self)->int:
        '''Get the id of the last node in the workflow.'''
        return self['last_node_id']
    
    @property
    def last_link_id(self)->int:
        '''Get the id of the last link in the workflow.'''
        return self['last_link_id']

    @property
    def version(self)->Optional[str]:
        '''Get the version of the workflow.'''
        ver = self.get('version')
        if ver is None:
            return ver
        return str(ver)
    
    @property
    def stable_renderer_version(self)->Optional[str]:
        '''Get the stable renderer version of the workflow.'''
        return self.get('stable_renderer_version')
    
    @property
    def is_stable_renderer_workflow(self)->bool:
        '''Check if the workflow is a stable renderer workflow.'''
        return self.stable_renderer_version is not None
    
    @property
    def extra(self)->Optional[dict]:
        '''extra information. Not sure what is this.'''
        return self.get('extra')
    
    @property
    def groups(self)->Optional[dict]:
        '''Not sure what is `groups` in ComfyUI.'''
        return self.get('groups')
    
    @property
    def config(self)->Optional[dict]:
        '''Not sure what is `config` in ComfyUI.'''
        return self.get('config')
    
    @property
    def nodes(self)->Dict[str, WorkflowNodeInfo]:
        '''Get the nodes in the workflow.'''
        return self['nodes']
    
    @property
    def node_links(self)->Dict[int, WorkflowNodeLink]:
        '''Get the links between nodes.'''
        return self['links']
    
    def build_prompt(self)->Tuple['PROMPT', List[str], dict]:
        '''
        Build the required input parameters for PromptExecutor.execute().
        
        Returns:
            - prompt: the prompt for the execution.
            - extra_data: extra data for the execution.
        '''
        prompt_dict = {}
        node_ids = [int(key) for key in self.nodes.keys()]
        node_ids.sort()
        node_ids = [str(node_id) for node_id in node_ids]
        node_ids_to_be_ran = []
        
        for node_id in node_ids:
            node = self.nodes[node_id]
            node_data = {}
            node_data['inputs'] = {param_name: param.value for param_name, param in node.inputs.items()}
            node_data['class_type'] = node.cls_type_name
            if hasattr(node.cls_type, 'OUTPUT_NODE') and node.cls_type.OUTPUT_NODE:
                node_ids_to_be_ran.append(node_id)
            prompt_dict[node_id] = node_data
        
        extra_data = {}
        extra_data['extra_pnginfo'] = {'workflow': self.original_data}   

        from comfyUI.types import PROMPT
        prompt = PROMPT(prompt_dict, id=str(uuid4()).replace('-', ''))
        
        return prompt, node_ids_to_be_ran, extra_data
    
    @overload
    def __init__(self, json_str: str): ...
    @overload
    def __init__(self, *args, **kwargs): ...

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and type(args[0]) == str:
            super().__init__(json.loads(args[0]))
        else:
            super().__init__(*args, **kwargs)
        
        self.original_data = self.copy()
        
        if version:=self.get('version'):
            versions = [int(val) for val in str(version).split('.')]
            if versions[0] >= 1:
                EngineLogger.warning(f'Workflow version {version} may not be supported. Latest supported version is 0.4')
            elif versions[1] >= 5:
                EngineLogger.warning(f'Workflow version {version} may not be supported. Latest supported version is 0.4')
        
        if not self.is_stable_renderer_workflow:
            EngineLogger.warning('This workflow is not a stable renderer workflow. Some data may be loaded in wrong way, probably causing errors during execution.')
        
        if 'links' in self:
            links = {}
            for link in self['links']:
                node_link = WorkflowNodeLink(link)
                links[node_link.id] = node_link
            self['links'] = links
        else:
            self['links'] = {}
        
        if 'nodes' not in self:
            raise ValueError('Invalid workflow file, cannot find `nodes`.')
        else:
            nodes = WorkflowNodeInfo._ParseNodeInfos(self['nodes'], self)
            self['nodes'] = nodes # replace the original nodes with the formatted nodes.
        
    def __repr__(self):
        return f'<Workflow name={self.name}>'
    
    @classmethod
    def Load(cls, path: Union[str, Path])->'Workflow':
        '''Load a workflow from a file.'''
        if not os.path.exists(path) and not str(path).endswith('.json'):
            path = BUILTIN_WORKFLOW_DIR / f'{path}.json'
        if not os.path.exists(path):
            raise FileNotFoundError(f'Cannot find the workflow file {path}.')
        elif not os.path.isfile(path):
            raise ValueError(f'The path {path} is not a file.')
        workflow_name = os.path.basename(path).split('.')[0]
        with open(path, 'r') as f:
            workflow = cls(json.load(f))
            workflow['name'] = workflow_name
            return workflow
    
    
__all__ = ['Workflow']


if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    from common_utils.path_utils import EXAMPLE_WORKFLOWS_DIR, TEMP_DIR
    from comfyUI.main import run
    
    test_workflow_path = EXAMPLE_WORKFLOWS_DIR / 'boat-img2img-example.json'
    workflow = Workflow.Load(test_workflow_path)
    prompt, node_ids_to_be_ran, extra_data = workflow.build_prompt()
    
    prompt_executor = run()
    context = prompt_executor.execute(prompt, node_ids_to_be_ran=node_ids_to_be_ran, extra_data=extra_data)
    color_img = context.final_output.frame_color
    img = 255. * color_img[0].cpu().numpy()
    img_bytes = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    saving_path = TEMP_DIR / 'boat_img2img_example.png'
    img_bytes.save(saving_path)