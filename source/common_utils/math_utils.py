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


__all__ .extend(['calc_map_mean_std', 'adaptive_instance_normalization'])


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
    
    Raises:
        ValueError: If the index column is out of range.
        ValueError: If any value column is out of range.

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
    (tensor([[ 5.,  8.],
        [ 5.,  8.],
        [ 4.,  4.],
        [ 3., 99.],
        [ 1.,  3.]]), )
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


def tensor_group_by_then_randn_init(
        t: torch.Tensor,
        index_column: int,
        value_columns: list[int],
        return_unique: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Groups the values in a tensor by the unique values in a specified index column,
    and randomly initializes the corresponding values in the specified value columns. The values
    in the same group (index) will be initialized with the same random values.

    Args:
        t (torch.Tensor): The input tensor, where the last dimension contains the index and value columns.
        index_column (int): The index column to group by.
        value_columns (list[int]): The value columns to calculate the average of.
        return_unique (bool, optional): Whether to return the unique values. Defaults to False.

    Raises:
        ValueError: If the index column is out of range.
        ValueError: If the value columns contain out of range values.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: If `return_unique` is True, returns a tuple containing
            the unique values and the corresponding random values. If `return_unique` is False,
            returns only the random values.
    
    Example:
    ```python
    >>>> t = torch.tensor([[2, 1, 4],
                           [2, 9, 12],
                           [6, 4, 4],
                           [7, 3, 99],
                           [8, 1, 3]])
    >>> tensor_group_by_randn_init(t, index_column=0, value_columns=[1, 2])
    (tensor([[-0.9705,  1.2572],
        [-0.9705,  1.2572],
        [-0.5329, -0.4554],
        [-1.9584, -0.7853],
        [ 1.6839, -0.0536]]),)
    >>> tensor_group_by_randn_init(t, index_column=1, value_columns=[0], return_unique=True)
    (tensor([[ 0.3773],
        [ 1.2324],
        [-0.1884],
        [ 0.1770],
        [ 0.3773]]), tensor([1, 3, 4, 9]))
    ```
    """
    if index_column >= t.shape[-1]:
        raise ValueError(f"Index column {index_column} is out of range.")
    if any([col >= t.shape[-1] for col in value_columns]):
        raise ValueError(f"Value columns {value_columns} contain out of range values.")
    
    # Get unique values and their corresponding inverse indices
    unique_values, inverse_indices = t[:, index_column].unique(return_inverse=True)

    expanded_unique_values = unique_values.unsqueeze(1).expand(-1, len(value_columns))

    # Randomly initialize the second column values based on the unique values
    random_values = torch.randn_like(
        expanded_unique_values, dtype=torch.float
    )
    # Expand random values to the original tensor
    expanded_random_values = random_values[inverse_indices]

    if return_unique:
        return expanded_random_values, unique_values
    else:
        return (expanded_random_values, )


def tensor_group_by_then_set_first_occurance(
        t: torch.Tensor,
        index_column: int,
        occurance_column: int,
        value_columns: list[int],
        return_unique: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    if index_column >= t.shape[-1]:
        raise ValueError(f"Index column {index_column} is out of range.")
    if occurance_column >= t.shape[-1]:
        raise ValueError(f"Occurance column {occurance_column} is out of range.")
    if any([col >= t.shape[-1] for col in value_columns]):
        raise ValueError(f"Value columns {value_columns} contain out of range values.")
    
    # Get unique values and their corresponding inverse indices
    unique_values, inverse_indices = t[:, index_column].unique(return_inverse=True)
    
    # Determine the first occurrence of each unique value based on occurance_column
    sorted_indices = torch.argsort(t[:, occurance_column])
    sorted_inverse_indices = inverse_indices[sorted_indices]

    first_occurrences = torch.zeros_like(unique_values, dtype=torch.long)
    seen = torch.zeros(unique_values.size(0), dtype=torch.bool)

    for i, idx in enumerate(sorted_inverse_indices):
        if not seen[idx]:
            first_occurrences[idx] = i
            seen[idx] = True

    first_occurrences = sorted_indices[first_occurrences]

    # Gather the first occurrence values for value_columns
    first_values = t[first_occurrences][:, value_columns]

    # Expand and assign the first occurrence values to the result tensor
    result = t.clone()
    result[:, value_columns] = first_values[inverse_indices]
    result = result[:, value_columns]

    if return_unique:
        return result, unique_values
    else:
        return result, torch.tensor([])


__all__.extend(['tensor_group_by_then_average', 'tensor_group_by_then_randn_init'])


if __name__ == '__main__':
    def test_adaptive_instance_normalization():
        x = torch.randn(1,4,64,64)
        y = torch.randn(1,4,512,512)
        print(adaptive_instance_normalization(x, y).shape)
        
    def test_tensor_group_by_then_average():
        t = torch.tensor([[2, 1, 4],
                          [2, 9, 12],
                          [6, 4, 4],
                          [7, 3, 99],
                          [8, 1, 3]])
        print(tensor_group_by_then_average(t, index_column=0, value_columns=[1, 2]))
        print(tensor_group_by_then_average(t, index_column=1, value_columns=[0], return_unique=True))

    def test_tensor_group_by_then_randn_init():
        t = torch.tensor([[2, 1, 4],
                          [2, 9, 12],
                          [6, 4, 4],
                          [7, 3, 99],
                          [8, 1, 3]])
        print(tensor_group_by_then_randn_init(t, index_column=0, value_columns=[1, 2]))
        print(tensor_group_by_then_randn_init(t, index_column=1, value_columns=[0], return_unique=True))
    
    def test_tensor_group_by_then_set_first_occurance():
        t = torch.tensor([[2, 1, 0, 4],
                          [2, 9, 1, 12],
                          [6, 4, 0, 5],
                          [7, 3, 0, 99],
                          [8, 1, 1, 3]])
        print(tensor_group_by_then_set_first_occurance(t, index_column=0, occurance_column=2, value_columns=[1, 3]))
        print(tensor_group_by_then_set_first_occurance(t, index_column=1, occurance_column=2, value_columns=[0], return_unique=True))
    

    test_tensor_group_by_then_set_first_occurance()
    