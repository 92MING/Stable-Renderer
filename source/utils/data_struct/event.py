# -*- coding: utf-8 -*-
'''事件类，用于实现事件机制。比PYQT原生Signal輕量，不需要QObject。當QObject被刪除後，invoke event時，對應的監聽方法在運行到RuntimeError的時候，會自動刪除。'''

import functools
from PySide2.QtCore import QObject, Signal
from types import FunctionType, MethodType
from typing import ForwardRef, get_origin, get_args, Union, Iterable, Literal
from inspect import getfullargspec, signature, getmro
from enum import Enum
from collections.abc import Callable

from utils.base_clses.cross_module_enum import CrossModuleEnum
from utils.type_utils import valueTypeCheck, subClassCheck

def _SignalWrapper(*args):
    class SignalWrapperCls(QObject):
        signal = Signal(*args)
    return SignalWrapperCls()
def _allArgsNotUnion(args):
    for arg in args:
        if get_origin(arg) is not None and get_origin(arg) == Union:
            return False
    return True
def _getHasUnionInput(possibleInputs:list):
    for possibleInput in possibleInputs:
        if not _allArgsNotUnion(possibleInput):
            return possibleInput
    return None
def _allPossibleInputsOk(possibleInputs):
    for possibleInput in possibleInputs:
        if not _allArgsNotUnion(possibleInput):
            return False
    return True
def _getMroClsNames(cls):
    return [cls.__qualname__ for cls in getmro(cls)]
def _mroDistance(cls, targetType):
    origin = get_origin(targetType)
    if origin is not None:
        targetType = origin
    mroClsNames = _getMroClsNames(cls)
    return mroClsNames.index(targetType.__qualname__)
def _totalMroDistance(values, targetTypes, acceptNone):
    totalDistance = 0
    for i, value in enumerate(values):
        if acceptNone and value is None:
            continue
        totalDistance += _mroDistance(type(value), targetTypes[i])
    return totalDistance
def _findMostSuitableInputType(possibleInputs, values, acceptNone):
    suitableInputs = []
    for possibleInput in possibleInputs:
        ok = True
        for i, argType in enumerate(possibleInput):
            if values[i] is None and acceptNone:
                continue
            if not valueTypeCheck(values[i], argType):
                ok = False
                break
        if ok:
            suitableInputs.append(possibleInput)
    mostSuitableDistances = [_totalMroDistance(values, suitableInput, acceptNone) for suitableInput in suitableInputs]
    mostSuitableDistance = min(mostSuitableDistances)
    return suitableInputs[mostSuitableDistances.index(mostSuitableDistance)]

class ListenerNotFoundError(Exception):
    pass
class NoneTypeNotSupportedError(Exception):
    pass

class Event:
    def __init__(self, *args, useQtSignal=False, acceptNone=False, noCheck=False):
        '''
        :param args: 事件的参数类型（1個或多個），可以是 类型 或 类型名稱 。支持所有utils.TypeUtils.simpleTypeCheck的類型。
        :param useQtSignal: 是否使用Qt的Signal作為事件機制。一般在QT需要跨線程時使用。如果使用此參數，則需要明確定義所有可能的參數類型,
                    e.g: A繼承B, 如果A和B都可能作為參數類型，則需要Event(Union[A, B], useQtSignal=True), 否則QT的Siganl會
                    導致類型坍塌為基類。
        :param acceptNone: 是否接受None作為參數。如果args中有None，則acceptNone會強制設置為True(僅限useQtSignal模式)。
        :param noCheck: addListener/invoke 時不檢查參數數量/類型是否正確
        '''
        args = list(args)
        for i, arg in enumerate(args):
            if arg is None:
                args[i] = type(None)
                if not acceptNone:
                    if not useQtSignal:
                        raise NoneTypeNotSupportedError("Event does not support NoneType as arg since 'acceptNone'=False")
                    else:
                        acceptNone = True # force acceptNone to True
            elif not isinstance(arg, (str, type, ForwardRef)):
                raise Exception("Event's arg must be type or type name, but got " + str(type(arg)))
        self._args = tuple(args)
        self._useQtSignal = useQtSignal
        self._acceptNone = acceptNone
        self._events = set()
        self._tempEvents = set()
        self._noCheck = noCheck
        self._hasEmitted = False  # to prevent sometimes Signal emit twice with unknown reason
        self._noneIndexes = set() # record the index of None in args. Since Signal is hard to emit None, we need to mark down the index of None and replace when _invoke
        if useQtSignal:
            signalArgs = []
            for arg in args:
                if subClassCheck(arg, CrossModuleEnum):
                    signalArgs.append(Enum)
                elif isinstance(arg, type):
                    signalArgs.append(arg)
                elif get_origin(arg) is not None:
                    origin = get_origin(arg)
                    if origin == Union:
                        signalArgs.append(arg)
                    elif origin != Literal:
                        signalArgs.append(origin)
                    else:
                        signalArgs.append(object)
                else:
                    signalArgs.append(object)
            possibleInputs = [signalArgs]
            while not _allPossibleInputsOk(possibleInputs):
                possibleInput = _getHasUnionInput(possibleInputs)
                possibleInputs.remove(possibleInput)
                for i, arg in enumerate(possibleInput):
                    if get_origin(arg) is not None and get_origin(arg) == Union:
                        for argType in get_args(arg): # for each type in Union
                            argTypeOrigin = get_origin(argType)
                            if argTypeOrigin is not None and argTypeOrigin != Union:
                                argType = argTypeOrigin
                            newPossibleInput = list(possibleInput)
                            newPossibleInput[i] = argType
                            possibleInputs.append(newPossibleInput.copy())
                        break
            self._possibleInputs = possibleInputs
            if len(possibleInputs) == 1:
                self._qtSignal = _SignalWrapper(*possibleInputs[0])
                self._qtSignal.signal.connect(self._invoke)
            else:
                self._qtSignal = _SignalWrapper(*possibleInputs)
                for possibleInput in possibleInputs:
                    self._qtSignal.signal.__getitem__(possibleInput).connect(self._invoke)
            if acceptNone:
                self._possibleInputs.append([type(None)] * len(signalArgs))

    def __iadd__(self, other):
        self.addListener(other)
        return self
    def __isub__(self, other):
        self.removeListener(other)
        return self
    def destroy(self):
        self._events.clear()
        self._tempEvents.clear()
        self._args = None
        self._events = None
        self._tempEvents = None
        if self._useQtSignal:
            try:
                self._qtSignal.disconnect()
            except RuntimeError:
                pass
            self._qtSignal.deleteLater()
            self._qtSignal = None
        del self

    @property
    def args(self):
        return self._args
    @property
    def argCount(self):
        return len(self._args)
    @property
    def events(self)->tuple:
        '''return a tuple of events'''
        return tuple(self._events)
    @property
    def argLength(self)->int:
        '''return the length of args'''
        return len(self.args)
    @property
    def tempEvents(self)->tuple:
        '''return a tuple of temp events'''
        return tuple(self._tempEvents)
    @property
    def acceptNone(self)->bool:
        return self._acceptNone
    @property
    def useQtSignal(self)->bool:
        return self._useQtSignal

    def _checkListener(self, listener:Callable):
        if self._noCheck:
            return
        try:
            isVarArgFunc = getfullargspec(listener).varargs is not None   # 是否是可变参数函数
            defaultsCount = len(getfullargspec(listener).defaults) if getfullargspec(listener).defaults is not None else 0
        except TypeError:
            isVarArgFunc = False
            defaultsCount = 0
        if isinstance(listener,FunctionType):
            argLength = listener.__code__.co_argcount
        elif isinstance(listener,MethodType):
            if issubclass(listener.__self__.__class__, Event):
                if listener.__qualname__.split('.')[-1] == 'invoke':
                    args = listener.__self__.args
                    for i, arg in enumerate(args):
                        if not subClassCheck(arg, self.args[i]):
                            raise Exception("Listener's arg must be type or type name")
                    argLength = len(args)
                else:
                    argLength = listener.__code__.co_argcount - 1
            else:
                argLength = listener.__code__.co_argcount - 1
        elif isinstance(listener, functools.partial):
            diff = 1 if isinstance(listener.func, MethodType) else 0
            if hasattr(listener.func, '__code__'):
                argLength = listener.func.__code__.co_argcount - len(listener.args) - len(listener.keywords) - diff
            else:
                return # really can't get arg length
        elif isinstance(listener, Callable):
            try:
                sig = signature(listener)
                argLength = len(sig.parameters)
            except ValueError: # really can't get arg length
                return
        else:
            raise Exception("Listener must be function or method. Got " + str(type(listener)))
        if not isVarArgFunc:
            possibleArgRange = range(argLength - defaultsCount, argLength + 1)
            if self.argLength not in possibleArgRange:
                raise Exception(f"Listener's arg length not match, self.argLength= {self.argLength}, but possible range is {possibleArgRange}")
        else:
            if self.argLength < (argLength - defaultsCount):
                raise Exception(f"Listener's arg length not match, self.argLength= {self.argLength}, but possible range is {argLength - defaultsCount}+")
    def _removeDestroyedListener(self, listener):
        try:
            self._events.remove(listener)
        except KeyError:
            pass
    def _removeDestroyedTempListener(self, listener):
        try:
            self._tempEvents.remove(listener)
        except KeyError:
            pass
    def addListener(self, listener:Union[callable, Iterable[callable]]):
        '''add a listener to event'''
        if isinstance(listener, Iterable):
            for l in listener:
                self.addListener(l)
        elif isinstance(listener, Callable):
            self._checkListener(listener)
            if isinstance(listener,MethodType):
                if isinstance(listener.__self__, QObject):
                    listener.__self__.destroyed.connect(functools.partial(self._removeDestroyedListener, listener))
            self._events.add(listener)
        else:
            raise TypeError("Listener must be callable, or iterable of callable")
    def addTempListener(self, listener:Callable):
        '''add a temp listener to event'''
        self._checkListener(listener)
        if isinstance(listener,MethodType):
            if isinstance(listener.__self__, QObject):
                listener.__self__.destroyed.connect(functools.partial(self._removeDestroyedTempListener, listener))
        self._tempEvents.add(listener)

    def removeListener(self, listener:Callable, throwError=True):
        '''remove a listener from event'''
        try:
            self._events.remove(listener)
        except KeyError:
            if throwError:
                raise ListenerNotFoundError
    def removeTempListener(self, listener:Callable, throwError=True):
        '''remove a temp listener from event'''
        try:
            self._tempEvents.remove(listener)
        except KeyError:
            if throwError:
                raise ListenerNotFoundError

    def _invoke(self, *args):
        if self._hasEmitted:
            return # when use QtSignal, sometimes it will emit twice with unknown reason
        self._hasEmitted = True
        if self.acceptNone and self.useQtSignal:
            args = list(args)
            for index in self._noneIndexes:
                args[index] = None
        for event in self.events:
            try:
                event(*args)
            except RuntimeError as e:
                if 'deleted' in str(e):
                    # C/C++ object has been deleted(usually occurs in PYQT)
                    self.removeListener(event)
                else:
                    raise e
        for event in self.tempEvents:
            try:
                event(*args)
            except RuntimeError as e:
                if 'deleted' in str(e):
                    # C/C++ object has been deleted(usually occurs in PYQT)
                    continue
                else:
                    raise e
        self._tempEvents.clear()
    def _check_invoke_params(self, *args):
        if len(args) < len(self.args):
            argNeeded = self.args[len(args):]
            outputStr = ""
            for arg in argNeeded:
                outputStr += f"{arg.__qualname__}, "
            outputStr = outputStr[:-2]
            raise Exception(f"Parameter: {outputStr[:-2]} are not provided")
        elif len(args) > len(self.args):
            raise Exception(f"Too many parameters")
        for i, arg in enumerate(args):
            if isinstance(self.args[i], str):
                # check type by name
                if type(arg).__qualname__ != self.args[i]:
                    raise Exception(f"Class of parameter: {arg} is not {self.args[i]}")
            elif not valueTypeCheck(arg, self.args[i]):
                if not (arg is None and self._acceptNone):
                    raise Exception(f"Parameter: {arg} is not valid for type'{self.args[i]}'")
    def invoke(self, *args):
        '''invoke all listeners'''
        if not self._noCheck:
            self._check_invoke_params(*args)
        self._hasEmitted = False
        if not self._useQtSignal:
            self._invoke(*args)
        else:
            for i, arg in enumerate(args):
                if arg is None:
                    if self.acceptNone:
                        self._noneIndexes.add(i)
                    else:
                        raise NoneTypeNotSupportedError('''None type is not supported since you have set "acceptNone" to False.
                                                        When you want to pass None in useQtSignal mode, you must use set "acceptNone" to True.''')
            if len(self._possibleInputs) == 0:
                self._qtSignal.signal.emit(*args)
            else:
                mostSuitableInput = _findMostSuitableInputType(self._possibleInputs, args, self.acceptNone)
                self._qtSignal.signal.__getitem__(mostSuitableInput).emit(*args)
            self._noneIndexes.clear()

    def eventsCount(self):
        '''return the count of normal events'''
        return len(self._events)
    def tempEventsCount(self):
        '''return the count of temp events'''
        return len(self._tempEvents)

    def clear(self):
        '''clear all events(both temp and normal)'''
        self._events.clear()
        self._tempEvents.clear()


__all__ = ["Event", "ListenerNotFoundError", "NoneTypeNotSupportedError"]
