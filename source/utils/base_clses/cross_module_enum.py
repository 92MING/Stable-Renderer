# -*- coding: utf-8 -*-
'''跨模組Enum，無論在哪裡import，都返回同一個Enum類。名稱不能重複'''

from ..global_utils import GetOrAddGlobalValue
_CrossModuleEnumDict = GetOrAddGlobalValue("_CrossModuleEnumDict", dict())
from enum import EnumMeta, Enum

class CrossModuleEnumMeta(EnumMeta):
    def __new__(cls, *args, **kwargs):
        clsName = args[0]
        if clsName in _CrossModuleEnumDict:
            return _CrossModuleEnumDict[clsName]
        else:
            thisEnumCls = super().__new__(cls, *args, **kwargs)
            _CrossModuleEnumDict[clsName] = thisEnumCls
            return thisEnumCls

class CrossModuleEnum(Enum, metaclass=CrossModuleEnumMeta):
    '''跨模組Enum，無論在哪裡import，都返回同一個Enum類。名稱不能重複'''
    pass

__all__ = ["CrossModuleEnum", "CrossModuleEnumMeta"]