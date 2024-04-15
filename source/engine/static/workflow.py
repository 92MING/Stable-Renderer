'''ComfyUI's Workflow class.'''
import json
from typing import Tuple, Union, Dict, List, Any, TYPE_CHECKING
from pathlib import Path
if TYPE_CHECKING:
    from engine.runtime.runtime_prompt import RuntimePrompt


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

class WorkflowNodeInputsInfo(Dict[str, Any]):
    pass

class WorkflowNodeOutputsInfo(Dict[str, Any]):
    pass

class WorkflowNodeInfo(Dict[str, Any]):
    
    _POP_ATTRS = ['color', 'bgcolor', 'size', 'pos', 'title', 'properties', 'mode']
    '''useless attributes that should be removed.'''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['id'] = str(self['id'])
        for attr in self._POP_ATTRS:
            if attr in self:
                del self[attr]
    
    @property
    def inputs(self)->'WorkflowNodeInputsInfo':
        '''inputs information of the node.'''
        return self['inputs']
    
    @property
    def outputs(self)->'WorkflowNodeOutputsInfo':
        '''outputs information of the node.'''
        return self['outputs']
    
    @property
    def id(self)->str:
        '''the unique id of the node.'''
        return self['id']
    
    @staticmethod
    def _ParseNodeInfos(infos: List[Dict])->'Dict[str, WorkflowNodeInfo]':
        datas = {}
        for info in infos:
            info['id'] = str(info['id'])
            datas[info['id']] = WorkflowNodeInfo(info)
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
    def version(self)->str:
        '''Get the version of the workflow.'''
        return self['version']
    @property
    def extra(self)->dict:
        '''extra information. Not sure what is this.'''
        return self['extra']
    @property
    def nodes(self)->Dict[str, WorkflowNodeInfo]:
        '''Get the nodes in the workflow.'''
        return self['nodes']
    @property
    def node_links(self)->List[WorkflowNodeLink]:
        '''Get the links between nodes.'''
        return self['links']
    
    def pack_as_prompt(self, runtime_prompt: 'RuntimePrompt')->dict:
        '''put 'runtime_prompt' info into the workflow and built the prompt data for submitting to ComfyUI'''
        pass

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and type(args[0]) == str:
            super().__init__(json.loads(args[0]))
        else:
            super().__init__(*args, **kwargs)
        self.original_data = self.copy()
        
        if 'nodes' not in self:
            raise ValueError('Invalid workflow file, cannot find `nodes`.')
        else:
            nodes = WorkflowNodeInfo._ParseNodeInfos(self['nodes'])
            self['nodes'] = nodes # replace the original nodes with the formatted nodes.
        
        if 'links' in self:
            links = []
            for link in self['links']:
                links.append(WorkflowNodeLink(link))
            self['links'] = links
        else:
            self['links'] = []

    @classmethod
    def Load(cls, path: Union[str, Path])->'Workflow':
        '''Load a workflow from a file.'''
        workflow = cls(json.load(open(path, 'r')))
        return workflow
    
    
__all__ = ['Workflow']