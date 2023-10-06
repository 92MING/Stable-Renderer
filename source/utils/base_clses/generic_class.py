# -*- coding: utf-8 -*-
'''this module contains generic classes.'''
from .cross_module_class import CrossModuleClass

class SingleGenericClass(CrossModuleClass):
    _type = None
    _genericClses = {}

    def __class_getitem__(cls, item):
        if item not in cls._genericClses:
            newCls = type(f'{cls.__name__}{item}', (cls,), {'__slots__':()})
            newCls._type = item
            cls._genericClses[item] = newCls
        return cls._genericClses[item]

    @classmethod
    def type(cls):
        if cls._type is None:
            raise TypeError(f'{cls.__name__} has no generic type. Please use {cls.__name__}[type] to get a generic class')
        return cls._type

class MultiGenericClass(CrossModuleClass):
    _types = None
    _genericClses = {}

    def __class_getitem__(cls, item):
        if not isinstance(item, tuple):
            item = (item,)
        if item not in cls._genericClses:
            newCls = type(f'{cls.__name__}{item}', (cls,), {'__slots__':()})
            newCls._types = item
            cls._genericClses[item] = newCls
        return cls._genericClses[item]

    @classmethod
    def types(cls):
        if cls._types is None:
            raise TypeError(f'{cls.__name__} has no generic types. Please use {cls.__name__}[types] to get a generic class')
        return cls._types


__all__ = ['SingleGenericClass', 'MultiGenericClass']