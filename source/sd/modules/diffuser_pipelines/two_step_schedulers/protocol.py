import torch
from typing import Protocol, Union, Optional


class TwoStepScheduler(Protocol):
    """
    The protocol for a two-step scheduler.
    """
    def predict_original_sample(self,
                             model_output: torch.FloatTensor,
                             timestep: Union[float, torch.FloatTensor],
                             sample: torch.FloatTensor,
                             generator: Optional[torch.Generator] = None,
                             **kwargs) -> torch.FloatTensor:
        """
        Predict the original sample `(x_{0})` based on the model output from the current timestep
        
        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.

        Returns:
            `torch.FloatTensor`: The predicted original sample `(x_{0})` given the current timestep.
        """
        ...
    
    def predict_previous_sample(self,
                             pred_original_sample: torch.FloatTensor,
                             model_output: torch.FloatTensor,
                             timestep: Union[float, torch.FloatTensor],
                             sample: torch.FloatTensor,
                             generator: Optional[torch.Generator] = None,
                             **kwargs) -> torch.FloatTensor:
        """
        Predict the previous sample `(x_{t-1})` based on the model output from the current timestep
        
        Args:
            pred_original_sample (`torch.FloatTensor`):
                The predicted original sample `(x_{0})` given the current timestep.
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.

        Returns:
            `torch.FloatTensor`: The predicted previous sample `(x_{t-1})` given the current timestep.
        """
        ...