# -*- coding: utf-8 -*-
'''跨模組類，無論在哪裡import，都會得到相同的類'''

from abc import ABCMeta, abstractmethod
from ..global_utils import GetOrCreateGlobalValue
_CrossModuleClassDict: dict = GetOrCreateGlobalValue("__CROSS_MODULE_CLASS_DICT__", dict)

class CrossModuleClassMeta(type):
    '''Cross Module class. You will get the same class no matter the way you import it.'''
    def __new__(cls, *args, **kwargs):
        clsName = args[0]
        if clsName in _CrossModuleClassDict:
            return _CrossModuleClassDict[clsName]
        else:
            thisCls = super().__new__(cls, *args, **kwargs)
            _CrossModuleClassDict[clsName] = thisCls
            return thisCls

class CrossModuleClass(metaclass=CrossModuleClassMeta):
    '''Cross Module class. You will get the same class no matter the way you import it.'''
    pass


class CrossModuleABCMeta(ABCMeta):
    '''Cross Module Abstract Base Class. You will get the same class no matter the way you import it.'''
    def __new__(cls, *args, **kwargs):
        clsName = args[0]
        if clsName in _CrossModuleClassDict:
            return _CrossModuleClassDict[clsName]
        else:
            thisCls = super().__new__(cls, *args, **kwargs)
            _CrossModuleClassDict[clsName] = thisCls
            return thisCls

class CrossModuleABC(metaclass=CrossModuleABCMeta):
    '''Cross Module Abstract Base Class. You will get the same class no matter the way you import it.'''
    pass


__all__ = ["CrossModuleClass", "CrossModuleClassMeta", "CrossModuleABC", "CrossModuleABCMeta"]