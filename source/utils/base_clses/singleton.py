# -*- coding: utf-8 -*-
'''單例模式基類。默認在定義時，自動生成實例（init)'''

from .cross_module_class import CrossModuleClassMeta
from functools import partial
from types import FunctionType
class SingletonMeta(CrossModuleClassMeta):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cls.__qualname__!="Singleton" and cls.ifInitOnDefine():
            cls._instance = cls()
    def __getattribute__(self, item):
        obj = super().__getattribute__(item)
        if (item.startswith('__') and item.endswith('__')) or item == '_instance':
            # 特殊屬性直接返回
            return obj
        if self._instance is None:
            self._instance = self()
        if isinstance(obj, property):
            # property 也可以直接透過類訪問
            return obj.fget(self._instance)
        elif isinstance(obj, FunctionType):
            # 函數綁定到實例
            return partial(obj, self._instance)
        else:
            return obj
    def _getAttrType(self, item):
        if hasattr(self, item):
            return type(super().__getattribute__(item))
        else:
            raise AttributeError(f"{self.__name__} has no attribute {item}")
    def _getOriginObj(self, item):
        return super().__getattribute__(item)

class Singleton(metaclass=SingletonMeta):
    '''單例模式基類, 繼承CrossModuleClass'''
    _instance = None # 實例

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # 如果沒有實例，生成實例
            if cls.__qualname__ == "Singleton":
                # 基類不可實例化
                raise Exception("Singleton can't be instantiated directly")
            cls._instance = super().__new__(cls)
            cls.__init__(cls._instance, *args, **kwargs)
            cls._instance.__init__ = lambda *args: None
            cls.__call__ = lambda *args: cls._instance
        return cls._instance

    def __getattribute__(self, item):
        # getattribute 時，返回類的屬性
        if item == '__class__':
            return super().__getattribute__(item)
        obj = getattr(self.__class__, item)
        if isinstance(obj, property):
            return obj.fget(self)
        return obj

    def __setattr__(self, key, value):
        # setattr 時，設置類的屬性
        if not hasattr(self.__class__, key):
            setattr(self.__class__, key, value)
        else:
            objType = self.__class__._getAttrType(key)
            if objType == property:
                self.__class__._getOriginObj(key).fset(self, value)
            else:
                setattr(self.__class__, key, value)

    @classmethod
    def instance(cls):
        '''如果没有实例，會自動生成'''
        return cls()
    @classmethod
    def ifInitOnDefine(cls)->bool:
        '''是否在定義時初始化，默認為True'''
        return True

__all__ = ["Singleton"]