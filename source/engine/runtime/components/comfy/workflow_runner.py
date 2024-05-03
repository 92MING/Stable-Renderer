from typing import TYPE_CHECKING, Union, Literal
from pathlib import Path
from ...component import Component
from common_utils.data_struct import Event
from engine.static.workflow import Workflow
from comfyUI.types import InferenceContext

if TYPE_CHECKING:
    from engine.runtime.gameObj import GameObject


class WorkflowRunner(Component):
    
    workflow: Union[Workflow, None] = None
    run_on_state: Literal['update', 'late_update', 'fixed_update']
    
    on_workflow_ran = Event(InferenceContext)
    
    _workflow_to_be_loaded: Union[str, Path, None] = None
    
    def __init__(self, 
                 gameObj: 'GameObject', 
                 enable=True, 
                 workflow: Union[str, Workflow, None] = None,
                 run_on_state: Literal['update', 'late_update', 'fixed_update'] = 'late_update'):
        super().__init__(gameObj, enable)
        if run_on_state not in ['update', 'late_update', 'fixed_update']:
            raise ValueError(f'Invalid run_on_state: {run_on_state}')
        self.run_on_state = run_on_state
        if not isinstance(workflow, Workflow):
            self._workflow_to_be_loaded = workflow
        else:
            self.workflow = workflow
    
    def awake(self):
        if self._workflow_to_be_loaded is not None:
            self.workflow = Workflow.Load(self._workflow_to_be_loaded)
            self._workflow_to_be_loaded = None
    
    def run_workflow(self):
        if self.workflow is None:
            return
        context = self.engine.DiffusionManager.SubmitPrompt(workflow=self.workflow)
        self.on_workflow_ran.invoke(context)
    
    def fixedUpdate(self):
        super().fixedUpdate()
        if self.run_on_state == 'fixed_update':
            self.run_workflow()
            
    def update(self):
        super().update()
        if self.run_on_state == 'update':
            self.run_workflow()

    def lateUpdate(self):
        super().lateUpdate()
        if self.run_on_state == 'late_update':
            self.run_workflow()



__all__ = ['WorkflowRunner']