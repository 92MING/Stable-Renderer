# -*- coding: utf-8 -*-
'''跨模組類，無論在哪裡import，都會得到相同的類'''

from ..global_utils import GetOrCreateGlobalValue
_CrossModuleClassDict: dict = GetOrCreateGlobalValue("__CROSS_MODULE_CLASS_DICT__", dict)

class CrossModuleClassMeta(type):
    '''跨模組類的Meta class'''
    def __new__(cls, *args, **kwargs):
        clsName = args[0]
        if clsName in _CrossModuleClassDict:
            return _CrossModuleClassDict[clsName]
        else:
            thisCls = super().__new__(cls, *args, **kwargs)
            _CrossModuleClassDict[clsName] = thisCls
            return thisCls

class CrossModuleClass(metaclass=CrossModuleClassMeta):
    '''跨模組類，無論在哪裡import，都會得到相同的類'''
    pass

__all__ = ["CrossModuleClass", "CrossModuleClassMeta"]