# -*- coding: utf-8 -*-
'''this module contains generic classes.'''

class SingleGenericClass:
    _type = None
    _genericClses = {}

    DEFAULT_TYPE = None
    '''override this to set default type'''

    def __class_getitem__(cls, item):
        if not isinstance(item, type):
            raise TypeError(f'{cls.__name__} only accepts type as generic type')
        if item not in cls._genericClses:
            itemName = item.__qualname__.split('.')[-1] if hasattr(item, '__qualname__') else item.__name__.split('.')[-1]
            newCls = type(f'{cls.__name__}_{itemName}', (cls,), {'__slots__':()})
            newCls._type = item
            cls._genericClses[item] = newCls
        return cls._genericClses[item]

    @classmethod
    def type(cls):
        if cls._type is None:
            if cls.DEFAULT_TYPE is not None:
                return cls.DEFAULT_TYPE
            raise TypeError(f'{cls.__name__} has no generic type. Please use {cls.__name__}[type] to get a generic class')
        return cls._type

class MultiGenericClass:
    _types = None
    _genericClses = {}

    DEFAULT_TYPES = None
    '''override this to set default types, e.g. (int, int)'''

    def __class_getitem__(cls, item):
        item = tuple(item)
        if item not in cls._genericClses:
            newCls = type(f'{cls.__name__}{item}', (cls,), {'__slots__':()})
            newCls._types = item
            cls._genericClses[item] = newCls
        return cls._genericClses[item]

    @classmethod
    def types(cls):
        if cls._types is None:
            if cls.DEFAULT_TYPES is not None:
                return cls.DEFAULT_TYPES
            raise TypeError(f'{cls.__name__} has no generic types. Please use {cls.__name__}[types] to get a generic class')
        return cls._types


__all__ = ['SingleGenericClass', 'MultiGenericClass']