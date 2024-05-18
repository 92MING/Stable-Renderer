if __name__ == '__main__':  # for debugging
    import sys, os
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(_proj_path)
    __package__ = 'common_utils'
    
import torch
import taichi as ti
from typing import Literal
from einops import rearrange
from .global_utils import GetOrAddGlobalValue, SetGlobalValue


# region taichi
def init_taichi():
    '''this method will initialize taichi if it has not been initialized yet'''
    if not GetOrAddGlobalValue('__TAICHI_INITED__', False):
        ti.init(arch=ti.gpu)
        SetGlobalValue('__TAICHI_INITED__', True)

__all__ = ['init_taichi']
# endregion



# from https://github.com/naoto0804/pytorch-AdaIN/blob/master/function.py
def calc_map_mean_std(feat:torch.Tensor, eps=1e-5):
    '''
    Calculate the mean and standard deviation of the feature map.
    
    Args:
        - feat: the feature tensor with shape (N, C, H, W)
        - eps: a small value added to the variance to avoid divide-by-zero.
    '''
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    
    change_back = False
    if feat_var.dtype == torch.float16:
        feat_var = feat_var.float()
        change_back = True
    
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    
    if change_back:
        feat_var = feat_var.half()
        feat_std = feat_std.half()
    return feat_mean, feat_std

# from https://github.com/naoto0804/pytorch-AdaIN/blob/master/function.py
def adaptive_instance_normalization(content_feat: torch.Tensor, 
                                    style_feat: torch.Tensor, 
                                    eps: float=1e-5,
                                    mode: Literal['NCHW', 'NHWC']='NCHW'):
    '''
    Calculate the adaptive instance normalization of the content feature tensor with the style feature tensor.
    
    Args:
        - content_feat: the content feature tensor
        - style_feat: the style feature tensor
        - eps: a small value added to the variance to avoid divide-by-zero.
        - mode: the mode of the two tensors, 'NCHW' or 'NHWC'
    '''
    if mode == 'NCHW':
        assert (content_feat.size()[:2] == style_feat.size()[:2])   # N, C
    elif mode == 'NHWC':
        assert ((content_feat.size()[0], content_feat.size()[3]) == (style_feat.size()[0], style_feat.size()[3]))
        content_feat = rearrange(content_feat, 'N H W C -> N C H W')
        style_feat = rearrange(style_feat, 'N H W C -> N C H W')
    size = content_feat.size()
    style_mean, style_std = calc_map_mean_std(style_feat, eps)
    content_mean, content_std = calc_map_mean_std(content_feat, eps)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def tensor_group_by_then_average(t: torch.Tensor,
                     index_column: int,
                     value_columns: list[int],
                     return_unique: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Groups the values in a tensor by the unique values in a specified index column,
    and calculates the average of the corresponding values in the specified value columns.

    Args:
        t (torch.Tensor): The input tensor, where the last dimension contains the index and value columns.
        index_column (int): The index column to group by.
        value_columns (list[int]): The value columns to calculate the average of.
        return_unique (bool, optional): Whether to return the unique values. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: If `return_unique` is True, returns a tuple containing
            the unique values and the corresponding average values. If `return_unique` is False,
            returns only the average values.
    
    Example:
    ```python
    >>>> t = torch.tensor([[2, 1, 4],
                           [2, 9, 12],
                           [6, 4, 4],
                           [7, 3, 99],
                           [8, 1, 3]])
    >>> tensor_group_by_average(t, index_column=0, value_columns=[1, 2])
    tensor([[ 5.,  8.],
        [ 5.,  8.],
        [ 4.,  4.],
        [ 3., 99.],
        [ 1.,  3.]])
    >>> tensor_group_by_average(t, index_column=1, value_columns=[0], return_unique=True)
    (tensor([[5.],
            [2.],
            [6.],
            [7.],
            [5.]]), tensor([1, 3, 4, 9]))
    ```
    """

    if index_column >= t.shape[-1]:
        raise ValueError(f"Index column {index_column} is out of range.")
    if any([col >= t.shape[-1] for col in value_columns]):
        raise ValueError(f"Value columns {value_columns} contain out of range values.")
    
    # Get unique values and their corresponding inverse indices
    unique_values, inverse_indices = t[:, index_column].unique(return_inverse=True)

    expanded_unique_values = unique_values.unsqueeze(1).expand(-1, len(value_columns))
    expanded_inverse_indices = inverse_indices.unsqueeze(1).expand(-1, len(value_columns))

    # Use scatter_add to sum the second column values based on the unique values
    sum_values = torch.zeros_like(
        expanded_unique_values, dtype=torch.float
    ).scatter_add_(0, expanded_inverse_indices, t[:, value_columns].to(torch.float))

    # Count occurrences of each unique value
    counts = torch.zeros_like(
        expanded_unique_values, dtype=torch.float
    ).scatter_add_(0, expanded_inverse_indices, torch.ones_like(t[:, value_columns], dtype=torch.float))

    # Calculate the average by dividing the summed values by their counts
    average_values = sum_values / counts

    # Expand average values to the original tensor
    expanded_average_values = average_values[inverse_indices]

    if return_unique:
        return expanded_average_values, unique_values
    else:
        return (expanded_average_values, )


__all__ .extend(['calc_map_mean_std', 'adaptive_instance_normalization', 'tensor_group_by_then_average'])


if __name__ == '__main__':
    def test_adaptive_instance_normalization():
        x = torch.randn(1,4,64,64)
        y = torch.randn(1,4,512,512)
        print(adaptive_instance_normalization(x, y).shape)
    def test_tensor_group_by_average():
        t = torch.tensor([[2, 1, 4],
                          [2, 9, 12],
                          [6, 4, 4],
                          [7, 3, 99],
                          [8, 1, 3]])
        print(tensor_group_by_then_average(t, index_column=0, value_columns=[1, 2]))
        print(tensor_group_by_then_average(t, index_column=1, value_columns=[0], return_unique=True))
    
    test_tensor_group_by_average()