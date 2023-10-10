# -*- coding: utf-8 -*-
'''常用的類型相關函數，例如類型檢查等'''

from typing import *
from inspect import getmro

# region private functions
def _checkSubClassByName(clsName, largerCls):
    return clsName in [cls.__qualname__ for cls in getmro(largerCls)]
def _directCheckSubClass(cls, largerCls):
    if get_origin(cls) == Union:
        return any([_directCheckSubClass(arg, largerCls) for arg in get_args(cls)])
    if isinstance(cls, ForwardRef):
        return _checkSubClassByName(cls.__forward_arg__, largerCls)
    elif isinstance(cls, str):
        return _checkSubClassByName(cls, largerCls)
    else:
        try:
            return issubclass(cls, largerCls)
        except TypeError:
            return False
def _checkTypeByClsName(value, clsName):
    return _directCheckSubClass(clsName, type(value))
def _directCheckType(value, targetType):
    if isinstance(targetType, ForwardRef):
        return _checkTypeByClsName(value, targetType.__forward_arg__)
    elif isinstance(targetType, str):
        return _checkTypeByClsName(value, targetType)
    else:
        return isinstance(value, targetType)
# endregion

def subClassCheck(smallerCls, largerCls):
    '''
    檢查smallerCls是否是largerCls的子類，支持Union、Optional、Sequence、Iterable、Mapping、Callable、Literal等類型。
    e.g.
        subClassCheck(int, Union[int, str]) -> True
        subClassCheck(Literal[1, 2], Union[int, str]) -> True
    '''
    def _checkTypes(smallerTypes, largerTypes):
        if Any in largerTypes:
            return True
        for smallerType in smallerTypes:
            ok = False
            for largerType in largerTypes:
                if subClassCheck(smallerType, largerType):
                    ok = True
                    break
            if not ok:
                return False
        return True
    if largerCls == Any:
        return True
    elif smallerCls == Any:
        return False
    smallerTypeOrigin = get_origin(smallerCls)
    largerTypeOrigin = get_origin(largerCls)

    if smallerTypeOrigin == Literal:
        smallerValues = get_args(smallerCls)
        if largerTypeOrigin == Literal:
            largerValues = get_args(largerCls)
            for smallerValue in smallerValues:
                if smallerValue not in largerValues:
                    return False
            return True
        else:
            for smallerValue in smallerValues:
                if not valueTypeCheck(smallerValue, largerCls):
                    return False
            return True
    elif smallerTypeOrigin is None and largerTypeOrigin is None:
        return _directCheckSubClass(smallerCls, largerCls)
    elif smallerTypeOrigin is not None and largerTypeOrigin is None:
        return _directCheckSubClass(smallerTypeOrigin, largerCls)
    elif smallerTypeOrigin is None and largerTypeOrigin is not None:
        if not _directCheckSubClass(smallerCls, largerTypeOrigin):
            return False
        if len(get_args(largerCls)) == 0:
            return True
        return False
    else:
        # both not None
        if smallerTypeOrigin in (Union, Optional):
            if largerTypeOrigin not in (Union, Optional):
                return False
            smallerTypes = get_args(smallerCls)
            largerTypes = get_args(largerCls)
            return _checkTypes(smallerTypes, largerTypes)
        else:
            if not _directCheckSubClass(smallerTypeOrigin, largerTypeOrigin):
                return False
            if _directCheckSubClass(smallerTypeOrigin,Sequence) or _directCheckSubClass(smallerTypeOrigin, Iterable):
                smallerArgType = get_args(smallerCls)[0]
                largerArgType = get_args(largerCls)[0]
                return subClassCheck(smallerArgType, largerArgType)
            elif _directCheckSubClass(smallerTypeOrigin, Mapping):
                smallerKeyType, smallerValueType = get_args(smallerCls)
                largerKeyType, largerValueType = get_args(largerCls)
                return subClassCheck(smallerKeyType, largerKeyType) and subClassCheck(smallerValueType, largerValueType)
            elif _directCheckSubClass(smallerTypeOrigin, Callable):
                smallerClsArgs = get_args(smallerCls)
                largerClsArgs = get_args(largerCls)
                smallerAgrTypes, smallerReturnType = smallerClsArgs[0], smallerClsArgs[-1]
                largerArgTypes, largerReturnType = largerClsArgs[0], largerClsArgs[-1]
                if len(smallerAgrTypes) != len(largerArgTypes):
                    return False
                return _checkTypes(smallerAgrTypes, largerArgTypes) and subClassCheck(smallerReturnType, largerReturnType)
            else:
                raise Exception('unsupported types--- smaller:', smallerTypeOrigin, ' larger:',largerTypeOrigin)

def valueTypeCheck(value, targetType):
    '''
    檢查value是否是targetType的實例，支持Union、Optional、Sequence、Iterable、Mapping、Callable、Literal等類型。
    特殊地，對於simpleTypeCheck(None, None), 返回True。
    e.g:
        valueTypeCheck(1, 'int') -> True
        valueTypeCheck(1, int) -> True
        valueTypeCheck(1, Union[int, str]) -> True
        valueTypeCheck(1, Union[str, float]) -> False
        valueTypeCheck(1, Optional[int]) -> True
        valueTypeCheck((1,), Tuple[int]) -> True
        valueTypeCheck((1,), Tuple[int, ...]) -> True
        valueTypeCheck((1, 2, 3), Tuple[int, int]) -> False
        valueTypeCheck([1, 2, 3], Sequence[int]) -> True
        valueTypeCheck("a", Literal["a","b"]) -> True
        valueTypeCheck(None, None) -> True    # special case
    '''
    if targetType == callable:
        targetType = Callable
    if value is None and targetType is None:
        return True # special case
    if get_origin(targetType) is None:
        return _directCheckType(value, targetType)
    else:
        argTypes = get_args(targetType)
        origin = get_origin(targetType)
        if origin == Union or origin == Optional:
            for argType in argTypes:
                if valueTypeCheck(value, argType):
                    return True
            return False
        elif origin == Literal:
            return value in argTypes
        else:
            if not _directCheckType(value, origin):
                return False
            if _directCheckSubClass(origin, Sequence):
                # special case for Tuple
                if _directCheckSubClass(origin, tuple):
                    if argTypes[-1] != Ellipsis and len(value) != len(argTypes):
                        return False
                    if argTypes[-1] == Ellipsis:
                        for v in value:
                            if not valueTypeCheck(v, argTypes[0]):
                                return False
                    else:
                        for i, argType in enumerate(argTypes):
                            if not valueTypeCheck(value[i], argType):
                                return False
                    return True
                else:
                    # for List, Deque, etc.
                    for v in value:
                        if not valueTypeCheck(v, argTypes[0]):
                            return False
                    return True
            elif _directCheckSubClass(origin, Mapping):
                keyType, valueType = argTypes
                for key, val in value.items():
                    if not valueTypeCheck(key, keyType) or not valueTypeCheck(val, valueType):
                        return False
                return True
            elif _directCheckSubClass(origin, Callable):
                if len(argTypes) == 0: # no any type hints of target callable type, means any callable is ok
                    return True
                else:
                    callArgTypes, returnType = *argTypes[:-1], argTypes[-1]
                    valueArgTypes = get_type_hints(value)  # here, value should be a function
                    varNames = value.__code__.co_varnames
                    for varName in varNames:
                        if varName not in valueArgTypes:
                            continue # ignore unlabeled arguments
                        if not subClassCheck(valueArgTypes[varName], callArgTypes[varNames.index(varName)]):
                            return False
                    if 'return' in valueArgTypes:
                        if not subClassCheck(valueArgTypes['return'], returnType):
                            return False
                    return True
            elif _directCheckSubClass(origin, Iterable):
                argType = argTypes[0]
                for v in value:
                    if not valueTypeCheck(v, argType):
                        return False
                return True
            else:
                raise Exception('not supported checking type: ' + str(origin))

__all__ = ['subClassCheck', 'valueTypeCheck']
