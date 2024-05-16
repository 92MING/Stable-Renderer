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
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
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


if __name__ == '__main__':
    x = torch.randn(1,4,64,64)
    y = torch.randn(1,4,512,512)
    print(adaptive_instance_normalization(x, y).shape)