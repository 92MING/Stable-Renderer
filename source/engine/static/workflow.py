'''ComfyUI's Workflow class, for loading to engine.'''

if __name__ == '__main__':  # for debugging
    import sys, os
    _proj_path = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    sys.path.append(_proj_path)
    __package__ = 'engine.static'

import json
from typing import Tuple, Type, Union, Dict, List, Any, TYPE_CHECKING, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from uuid import uuid4
from common_utils.debug_utils import EngineLogger
if TYPE_CHECKING:
    from comfyUI.types import ComfyUINode, NodeBindingParam, PROMPT


def _get_comfy_node_type(type_name: str):
    from comfyUI.types import get_node_type_by_name
    if not (t := get_node_type_by_name(type_name)):
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
    to_nodes: List[str] = field(default_factory=list)
    '''ids of the nodes that the output parameter links to.'''
    
class WorkflowNodeInfo(Dict[str, Any]):
    
    _cls_type: Type['ComfyUINode']
    '''the type of the node.'''
    _inputs: Dict[str, WorkflowNodeInputParam]
    '''the input parameters of the node.'''
    _outputs: List[WorkflowNodeOutputParam]
    '''the output parameters of the node.'''
    origin_data: dict
    '''the original data of the node.'''
    
    def __init__(self, origin_data: dict, links: Dict[int, WorkflowNodeLink]):
        super().__init__(origin_data)
        self.origin_data = origin_data
        self['id'] = str(self['id'])
        self._cls_type = _get_comfy_node_type(self['type'])
        
        # init inputs
        widget_values:Union[list, Dict[str, Any]] = self.get('widgets_values', [])
        '''The values of the widgets. It can be a list of values or a dict of values.'''
        workflow_inputs = self.get('inputs', [])    # original `inputs` is a list of dict for linked inputs.
        workflow_inputs = {i['name']: i for i in workflow_inputs}
        formatted_inputs:Dict[str, WorkflowNodeInputParam] = {}   # name, param
        
        node_cls = self.cls_type
        node_cls_input_types = node_cls.INPUT_TYPES()
        required_input_names = [i[0] for i in node_cls_input_types.get('required', [])]
        optional_input_names = [i[0] for i in node_cls_input_types.get('optional', [])]
        # no need to add hidden input types.
        
        def get_input_default_value(input_name:str):
            if input_name in node_cls_input_types.get('required', {}):
                input_info = node_cls_input_types['required'][input_name]
                if len(input_info)>=2:
                    return input_info[1].get('default', None)
                return None
            elif input_name in node_cls_input_types.get('optional', {}):
                input_info = node_cls_input_types['optional'][input_name]
                if len(input_info)>=2:
                    return input_info[1].get('default', None)
                return None
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
                        raise ValueError(f'Cannot find the link id for input {input_name}.')
                else:
                    link = links[link_id]
                    param_type = 'required' if input_name in required_input_names else 'optional'
                    formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                          link.to_node_binding_param(),
                                                                          node_cls_input_types[param_type][input_name][0])
            
            else:   # input is a constant value
                if len(widget_values) == 0:
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
                    if isinstance(widget_values, dict): # dict widget values, {input_name: value}
                        if not (widget_value := widget_values.get(input_name)):
                            if input_name in optional_input_names:
                                formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                                     None, 
                                                                                     node_cls_input_types['optional'][input_name][0])
                            else:
                                raise ValueError(f'Cannot find the widget value for input {input_name}.')
                    else:   # list widget values
                        widget_value = widget_values.pop(0)
                        param_type = 'required' if input_name in required_input_names else 'optional'
                        formatted_inputs[input_name] = WorkflowNodeInputParam(input_name, 
                                                                            widget_value, 
                                                                            node_cls_input_types[param_type][input_name][0])
        self._inputs = formatted_inputs
        
        # init outputs
        formatted_outputs = []
        if 'outputs' in self:
            for output in self['outputs']:
                output_name = output['name']
                output_type = output['type']
                output_to_nodes = [str(i) for i in output.get('links', [])]
                formatted_outputs.append(WorkflowNodeOutputParam(output_name, output_type, output_to_nodes))        
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
    def _ParseNodeInfos(infos: List[Dict], links: Dict[int, WorkflowNodeLink])->'Dict[str, WorkflowNodeInfo]':
        datas = {}
        for info in infos:
            info['id'] = str(info['id'])
            datas[info['id']] = WorkflowNodeInfo(info, links)
        return datas

class Workflow(Dict[str, Any]):
    '''
    The workflow for ComfyUI.
    Each workflow represents a rendering process/pipeline.
    '''
    original_data: dict
    '''the original dict data of the workflow.'''

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
        return self.get('version')
    
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
    
    def build_prompt(self)->Tuple['PROMPT', dict]:
        '''
        Build the required input parameters for PromptExecutor.execute().
        
        Returns:
            - prompt: the prompt for the execution.
            - extra_data: extra data for the execution.
        '''
        from comfyUI.types import PROMPT
        prompt = PROMPT(id=str(uuid4()).replace('-', ''))
        for node_id, node in self.nodes.items():
            node_data = {}
            node_data['inputs'] = {param_name: param.value for param_name, param in node.inputs.items()}
            node_data['class_type'] = node.cls_type_name
            prompt[node_id] = node_data 
        
        extra_data = {}
        extra_data['extra_pnginfo'] = {'workflow': self.original_data}   

        return prompt, extra_data

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and type(args[0]) == str:
            super().__init__(json.loads(args[0]))
        else:
            super().__init__(*args, **kwargs)
        
        self.original_data = self.copy()
        
        if version:=self.get('version'):
            versions = [int(val) for val in version.split('.')]
            if versions[0] >= 1:
                EngineLogger.warning(f'Workflow version {version} may not be supported. Latest supported version is 0.4')
            elif versions[1] >= 5:
                EngineLogger.warning(f'Workflow version {version} may not be supported. Latest supported version is 0.4')
        
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
            nodes = WorkflowNodeInfo._ParseNodeInfos(self['nodes'], self['links'])
            self['nodes'] = nodes # replace the original nodes with the formatted nodes.
        

    @classmethod
    def Load(cls, path: Union[str, Path])->'Workflow':
        '''Load a workflow from a file.'''
        workflow = cls(json.load(open(path, 'r')))
        return workflow
    
    
__all__ = ['Workflow']


if __name__ == '__main__':
    from common_utils.path_utils import TEMP_DIR
    test_workflow_path = TEMP_DIR / 'workflow.json'
    workflow = Workflow.Load(test_workflow_path)
    print(workflow)