# -*- coding: utf-8 -*-
'''overload decorator'''

from inspect import signature, getmro, Parameter
from collections import OrderedDict
from utils.type_utils import valueTypeCheck
from utils.global_utils import GetOrAddGlobalValue
_overloadFuncDict = GetOrAddGlobalValue("_overloadFuncDict", dict()) # {clsName: {funcName: [(prams,func, acceptArgs, accreptKws),]}}

def _getClsAndFuncName(func):
    funcName = func.__qualname__
    if '.' not in funcName:
        raise TypeError('The function must be a class method.')
    clsName, funcName = funcName.split('.')[-2:]
    return clsName, funcName

class _WrongInputError(Exception):
    pass
class NoSuchFunctionError(Exception):
    pass
class WrongCallingError(Exception):
    pass

class Overload:
    '''Class method overload. Must be used inside class definition.'''
    def __init__(self, func):
        self._instance = None
        clsname, funcname = _getClsAndFuncName(func)
        self._funcname = funcname
        if clsname not in _overloadFuncDict:
            _overloadFuncDict[clsname] = dict()
        if funcname not in _overloadFuncDict[clsname]:
            _overloadFuncDict[clsname][funcname] = []
        varnames = func.__code__.co_varnames
        if len(varnames)==0:
            raise TypeError('The function must be a class method.')
        prams:OrderedDict = signature(func).parameters.copy()
        prams.popitem(last=False) # remove 'self'
        acceptArgs = False
        acceptKws = False
        for key in tuple(prams.keys()):
            if prams[key].kind == Parameter.VAR_KEYWORD:
                prams.pop(key)
                acceptKws = True
            elif prams[key].kind == Parameter.VAR_POSITIONAL:
                prams.pop(key)
                acceptArgs = True
        _overloadFuncDict[clsname][funcname].append((prams, func, acceptArgs, acceptKws))

    def _getArgsAndKws(self, prams: OrderedDict, args, kwargs, acceptArgs, acceptKws):

        if len(args) > len(prams) and not acceptArgs:  # too many args provided
            raise _WrongInputError

        # fill args and kws
        pramNames = tuple(prams.keys())
        callingArgs = list(args)
        callingKws = kwargs.copy()
        if len(args) < len(prams):
            callingArgs.extend([pram.default for pram in prams.values()][len(args):])  # fill default args first
            for key in tuple(callingKws.keys()):
                if key in prams:
                    callingArgs[pramNames.index(key)] = callingKws.pop(key)  # then update callingArgs from callingKws
        if Parameter.empty in callingArgs:
            raise _WrongInputError # not enough args
        if len(callingKws) > 0 and not acceptKws:
            raise _WrongInputError # some kwargs not used, and not acceptKws

        # type check
        for i, pram in enumerate(prams.values()):
            thisArg = callingArgs[i]
            if pram.annotation != Parameter.empty: # annotation is not None
                if thisArg == pram.default:
                    continue # default value, no need to check

                if issubclass(type(pram.annotation), str):
                    # check type by class name
                    clsnames = [cls.__qualname__ for cls in getmro(thisArg.__class__)][:-1]
                    if pram.annotation not in clsnames:
                        raise _WrongInputError

                else: # check type normally
                    if not valueTypeCheck(thisArg, pram.annotation):
                        raise _WrongInputError

        return callingArgs, callingKws

    def __get__(self, instance, owner):
        self._instance = instance
        return self
    def __call__(self, *args, **kwargs):
        if self._instance is None:
            raise WrongCallingError('The function must be called by an instance of class.')
        clsnames = [cls.__qualname__ for cls in getmro(self._instance.__class__)][:-1]
        for clsname in clsnames:
            if clsname in _overloadFuncDict and self._funcname in _overloadFuncDict[clsname]:
                for prams, func, acceptArgs, acceptKws in _overloadFuncDict[clsname][self._funcname]:
                    try:
                        callingArgs, callingKws = self._getArgsAndKws(prams, args, kwargs, acceptArgs, acceptKws)
                        return func(self._instance, *callingArgs, **callingKws)
                    except _WrongInputError:
                        pass
        raise NoSuchFunctionError(f'No such function with args: {args} and kwargs:{kwargs}')

__all__ = ["Overload", "NoSuchFunctionError", "WrongCallingError"]