# -*- coding: utf-8 -*-
'''
Event class, used to implement event mechanism. 
When the QObject is deleted, when the corresponding listener method is running and encounters a RuntimeError, it will be automatically deleted.
'''

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = 'common_utils.data_struct'   

import heapq, functools

from types import FunctionType, MethodType
from inspect import getfullargspec, signature, getmro
from enum import Enum
from functools import partial
from collections.abc import Callable
from PySide6.QtCore import QObject, Signal
from typing import ForwardRef, get_origin, get_args, Union, Iterable, Literal, Callable

from common_utils.type_utils import valueTypeCheck, subClassCheck


# region helper functions
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
# endregion

class ListenerNotFoundError(Exception):
    pass

class NoneTypeNotSupportedError(Exception):
    pass

class Event:
    
    def _init_params(self, *args, useQtSignal=False, acceptNone=False, noCheck=False):
        args = list(args)
        for i, arg in enumerate(args):
            if arg is None:
                args[i] = type(None)
                if not acceptNone:
                    if not useQtSignal:
                        raise NoneTypeNotSupportedError(
                            "Event does not support NoneType as arg since 'acceptNone'=False")
                    else:
                        acceptNone = True  # force acceptNone to True
            elif not isinstance(arg, (str, type, ForwardRef)):
                raise Exception("Event's arg must be type or type name, but got " + str(type(arg)))
        self._args = tuple(args)
        self._useQtSignal = useQtSignal
        self._acceptNone = acceptNone
        self._events = set()
        self._tempEvents = set()
        self._noCheck = noCheck
        self._hasEmitted = False  # to prevent sometimes Signal emit twice with unknown reason
        self._noneIndexes = set()  # record the index of None in args. Since Signal is hard to emit None, we need to mark down the index of None and replace when _invoke
    
    def _init_pyqt_signal(self, *args, acceptNone:bool=False):
        signalArgs = []
        for arg in args:
            if subClassCheck(arg, Enum):
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
            possibleInputs.remove(possibleInput)    # type: ignore
            for i, arg in enumerate(possibleInput):    # type: ignore
                if get_origin(arg) is not None and get_origin(arg) == Union:
                    for argType in get_args(arg):  # for each type in Union
                        argTypeOrigin = get_origin(argType)
                        if argTypeOrigin is not None and argTypeOrigin != Union:
                            argType = argTypeOrigin
                        newPossibleInput = list(possibleInput)    # type: ignore
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
    
    def __init__(self, *args, useQtSignal=False, acceptNone=False, noCheck=False):
        '''
        :param args: 
            types of event when invoke(can be multiple types, e.g: Event(int, str)). Supports type or typename(`str`)
        :param useQtSignal: 
            Whether to use Qt's Signal as the event mechanism. Generally used when crossing threads in QT. 
            If this parameter is used, all possible parameter types need to be explicitly defined, 
            e.g: A inherits from B, if both A and B can be used as parameter types, 
            then Event(Union[A, B], useQtSignal=True) is required, 
            
            otherwise the QT Signal will cause the type to collapse to the base class. 
        :param acceptNone: 
            Whether to accept None as a parameter. If None is present in args, acceptNone will be forced to True (only in useQtSignal mode). 
        :param noCheck: Do not check the number and types of parameters when adding listeners or invoking events
        '''
        self._init_params(*args, useQtSignal=useQtSignal, acceptNone=acceptNone, noCheck=noCheck)
        if useQtSignal:
            self._init_pyqt_signal(*args, acceptNone=acceptNone)
        self._add_listener_decorator = self._get_event_decorator(False)
        self._add_temp_listener_decorator = self._get_event_decorator(True)
        
    def __iadd__(self, other):
        self.addListener(other)
        return self
    
    def __isub__(self, other):
        self.removeListener(other)
        return self
    
    def __len__(self):
        '''return the total count of events & temp events'''
        return len(self._events) + len(self._tempEvents)
    
    def destroy(self):
        self._events.clear()
        self._tempEvents.clear()
        if self._useQtSignal:
            try:
                self._qtSignal.disconnect()
            except RuntimeError:
                pass
            self._qtSignal.deleteLater()
        setattr(self, '_qtSignal', None)
        setattr(self, '_events', None)
        setattr(self, '_tempEvents', None)
        setattr(self, '_args', None)
        setattr(self, '_possibleInputs', None)
        setattr(self, '_noneIndexes', None)
        setattr(self, '_hasEmitted', None)
        setattr(self, '_ignoreErr', None)
        setattr(self, '_acceptNone', None)
        setattr(self, '_useQtSignal', None)
        setattr(self, '_noCheck', None)
        setattr(self, '_add_listener_decorator', None)
        setattr(self, '_add_temp_listener_decorator', None)
        del self

    def _get_event_decorator(self_event: 'Event', temp_listener:bool):   # type: ignore
        
        class _event_decorator:
            
            def __init__(self, fn):
                self.fn = fn
                self._need_init_ins_method = False
                if isinstance(fn, FunctionType):
                    if '.' not in fn.__qualname__:
                        self_event.addListener(self.fn) if not temp_listener else self_event.addTempListener(self.fn)
                    else: # class method
                        self._need_init_ins_method = True
            
            def __set_name__(self, owner, name):
                setattr(owner, name, self.fn)
                if self._need_init_ins_method:
                    origin_init = owner.__init__ if hasattr(owner, '__init__') else lambda *args, **kwargs: None
                    def new_init(*args, **kwargs):
                        origin_init(*args, **kwargs)
                        self_event.addListener(partial(getattr(owner, name), args[0])) if not temp_listener else self_event.addTempListener(partial(getattr(owner, name), args[0]))
                    owner.__init__ = new_init
                    origin_del = owner.__del__ if hasattr(owner, '__del__') else lambda *args, **kwargs: None
                    def new_del(*args, **kwargs):
                        self_event.removeListener(partial(getattr(owner, name), args[0]), throwError=False) if not temp_listener else self_event.removeTempListener(partial(getattr(owner, name), args[0]), throwError=False)
                        origin_del(*args, **kwargs)
                    owner.__del__ = new_del
                else:
                    self_event.addListener(getattr(owner, name)) if not temp_listener else self_event.addTempListener(getattr(owner, name))
            
            def __call__(self, *args, **kwargs):
                '''call the origin function'''
                return self.fn(*args, **kwargs)
        
        return _event_decorator
    
    @property
    def register(self)->Callable:
        return self._add_listener_decorator
    
    @property
    def temp_register(self)->Callable:
        return self._add_temp_listener_decorator
    
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
            isVarArgFunc = getfullargspec(listener).varargs is not None   # whether the function is varargs
            defaultsCount = len(getfullargspec(listener).defaults) if getfullargspec(listener).defaults is not None else 0  # type: ignore
        except TypeError:
            isVarArgFunc = False
            defaultsCount = 0
        if isinstance(listener,FunctionType):
            argLength = listener.__code__.co_argcount
        elif isinstance(listener,MethodType):
            if issubclass(listener.__self__.__class__, Event):
                if listener.__qualname__.split('.')[-1] == 'invoke':
                    args = listener.__self__.args   # type: ignore
                    for i, arg in enumerate(args):
                        if not subClassCheck(arg, self.args[i]):
                            raise Exception("Listener's arg must be type or type name")
                    argLength = len(args)
                else:
                    argLength = listener.__code__.co_argcount - 1    # type: ignore
            else:
                argLength = listener.__code__.co_argcount - 1    # type: ignore
        elif isinstance(listener, functools.partial):
            diff = 1 if isinstance(listener.func, MethodType) else 0
            if hasattr(listener.func, '__code__'):
                argLength = listener.func.__code__.co_argcount - len(listener.args) - len(listener.keywords) - diff    # type: ignore
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
        
    def addListener(self, listener:Union[Callable, Iterable[Callable]]):
        '''add a listener to event'''
        if isinstance(listener, Iterable):
            for l in listener:
                self.addListener(l)
        elif isinstance(listener, Callable):
            self._checkListener(listener)
            if isinstance(listener,MethodType):
                if isinstance(listener.__self__, QObject):
                    listener.__self__.destroyed.connect(functools.partial(self._removeDestroyedListener, listener))     # type: ignore
            self._events.add(listener)
        else:
            raise TypeError("Listener must be Callable, or iterable of Callable")
        
    def addTempListener(self, listener:Union[Callable, Iterable[Callable]]):
        '''add a temp listener to event'''
        if isinstance(listener, Iterable):
            for l in listener:
                self.addTempListener(l)
        elif isinstance(listener, Callable):
            self._checkListener(listener)
            if isinstance(listener,MethodType):
                if isinstance(listener.__self__, QObject):
                    listener.__self__.destroyed.connect(functools.partial(self._removeDestroyedTempListener, listener))    # type: ignore
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
        ignoreErr = self._ignoreErr  # will be assigned in self.invoke
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
            except Exception as e:
                if not ignoreErr:
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
            except Exception as e:
                if not ignoreErr:
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
    
    def invoke(self, *args, ignoreErr=False):
        '''invoke all listeners'''
        if not self._noCheck:
            self._check_invoke_params(*args)
        self._hasEmitted = False
        self._ignoreErr = ignoreErr
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


class DelayEvent(Event):
    '''
    When invoke delay event, it will not be executed immediately, but params will be save and execute later by calling "release".
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_param = None

    def invoke(self, *args):
        self._check_invoke_params(*args)
        self._last_param = args

    def release(self):
        if self._last_param is not None:
            super().invoke(*self._last_param)
        self._last_param = None


class Tasks(Event):
    '''
    Tasks is a subclass of Event, which will clear all tasks after invoke.
    Type of tasks must be Callable.
    When create, the init params will be the input types of tasks.
    '''
    def _init_pyqt_signal(self, *args, acceptNone=False):
        signalArgs = []
        
        for arg in args:
            if subClassCheck(arg, Enum):
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
            possibleInputs.remove(possibleInput)    # type: ignore
            for i, arg in enumerate(possibleInput):   # type: ignore
                if get_origin(arg) is not None and get_origin(arg) == Union:
                    for argType in get_args(arg):  # for each type in Union
                        argTypeOrigin = get_origin(argType)
                        if argTypeOrigin is not None and argTypeOrigin != Union:
                            argType = argTypeOrigin
                        newPossibleInput = list(possibleInput)  # type: ignore
                        newPossibleInput[i] = argType   
                        possibleInputs.append(newPossibleInput.copy())
                    break
        self._possibleInputs = possibleInputs
        if len(possibleInputs) == 1:
            self._qtSignal = _SignalWrapper(*possibleInputs[0])
            self._qtSignal.signal.connect(self._execute)    # type: ignore
        else:
            self._qtSignal = _SignalWrapper(*possibleInputs)
            for possibleInput in possibleInputs:
                self._qtSignal.signal.__getitem__(possibleInput).connect(self._execute)   # type: ignore
        if acceptNone:
            self._possibleInputs.append([type(None)] * len(signalArgs))

    def addTasks(self, tasks:Union[Callable, Iterable[Callable]]):
        if isinstance(tasks, Iterable):
            for task in tasks:
                self.addTempListener(task)
        else:
            self.addTempListener(tasks)

    def removeTasks(self, tasks:Union[Callable, Iterable[Callable]]):
        if isinstance(tasks, Iterable):
            for task in tasks:
                self.removeTempListener(task)
        else:
            self.removeTempListener(tasks)
            
    def addForeverTasks(self, tasks:Union[Callable, Iterable[Callable]]):
        if isinstance(tasks, Iterable):
            for task in tasks:
                self.addListener(task)
        else:
            self.addListener(tasks)
            
    def removeForeverTasks(self, tasks:Union[Callable, Iterable[Callable]]):
        if isinstance(tasks, Iterable):
            for task in tasks:
                self.removeListener(task)
        else:
            self.removeListener(tasks)

    def invoke(self, *args, ignoreErr=False):
        raise Exception("invoke is not supported in Tasks, use 'execute' instead")

    def execute(self, *args, ignoreErr=True):
        if not self._noCheck:
            self._check_invoke_params(*args)
        self._hasEmitted = False
        self._ignoreErr = ignoreErr
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

    @property
    def register(self)->Callable:
        return self._add_temp_listener_decorator    # `Tasks` add temp listener by default

    @property
    def register_forever(self)->Callable:
        return self._add_listener_decorator # `Tasks` add forever listener by default

class AutoSortTask(Tasks):
    '''Tasks will be cleared after execute. Tasks will be sorted by order.'''
    
    def _init_pyqt_signal(self, *args, acceptNone=False):
        signalArgs = []
        for arg in args:
            if subClassCheck(arg, Enum):
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
            possibleInputs.remove(possibleInput)    # type: ignore
            for i, arg in enumerate(possibleInput):   # type: ignore
                if get_origin(arg) is not None and get_origin(arg) == Union:
                    for argType in get_args(arg):  # for each type in Union
                        argTypeOrigin = get_origin(argType)
                        if argTypeOrigin is not None and argTypeOrigin != Union:
                            argType = argTypeOrigin
                        newPossibleInput = list(possibleInput)  # type: ignore
                        newPossibleInput[i] = argType
                        possibleInputs.append(newPossibleInput.copy())
                    break
        self._possibleInputs = possibleInputs
        if len(possibleInputs) == 1:
            self._qtSignal = _SignalWrapper(*possibleInputs[0])
            self._qtSignal.signal.connect(self._execute)
        else:
            self._qtSignal = _SignalWrapper(*possibleInputs)
            for possibleInput in possibleInputs:
                self._qtSignal.signal.__getitem__(possibleInput).connect(self._execute)
        if acceptNone:
            self._possibleInputs.append([type(None)] * len(signalArgs))
            
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tempEvents = [] # for heapq
        self._events = [] # for heapq

    class TaskWrapper:
        def __init__(self, func, order):
            self._func = func
            self._order = order
        @property
        def func(self):
            return self._func
        @property
        def order(self):
            return self._order
        def __call__(self, *args, **kwargs):
            self._func(*args, **kwargs)
        def __lt__(self, other):
            return self._order < other._order
        def __gt__(self, other):
            return self._order > other._order
        def __eq__(self, other):
            if isinstance(other, int):
                return self._order == other
            elif not isinstance(other, self.__class__):
                return self._func == other
            else:
                return super().__eq__(other)
        def __le__(self, other):
            return self._order <= other._order
        def __ge__(self, other):
            return self._order >= other._order
        def __ne__(self, other):
            return self._order != other._order
        
    @classmethod
    def _TaskWrapper(cls, func, order):
        return cls.TaskWrapper(func, order)

    def addListener(self, listener:Union[Callable, Iterable[Callable]], order:int|float=0):
        self._checkListener(listener)   # type: ignore
        if listener not in self._events:
            heapq.heappush(self._events, self._TaskWrapper(listener, order))
            
    def addTempListener(self, listener:Union[Callable, Iterable[Callable]], order:int|float=0):
        self._checkListener(listener)    # type: ignore
        if listener not in self._tempEvents:
            heapq.heappush(self._tempEvents, self._TaskWrapper(listener, order))
            
    def removeListener(self, listener, throwError:bool=False):
        if isinstance(listener, Iterable):
            for l in listener:   # type: ignore
                self.removeListener(l)
        else:
            for task in self._events:
                if task == listener:
                    self._events.remove(task)
                    return
            if throwError:
                raise ListenerNotFoundError(f"listener {listener} not found")
            
    def removeTempListener(self, listener, throwError:bool=False):
        if isinstance(listener, Iterable):
            for l in listener:   # type: ignore
                self.removeTempListener(l)
        else:
            for task in self._tempEvents:
                if task == listener:
                    self._tempEvents.remove(task)
                    return
            if throwError:
                raise ListenerNotFoundError(f"listener {listener} not found")

    def addTask(self, task:Callable, order:int|float=0):
        self.addTempListener(task, order)
        
    def addTasks(self, tasks:Union[Callable, Iterable[Callable]], orders:Union[int, float, Iterable[int|float]]=0):
        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        if isinstance(orders, int):
            orders = [orders] * len(tasks)   # type: ignore
        for task, order in zip(tasks, orders):  # type: ignore
            self.addTask(task, order)
    
    def addForeverTasks(self, tasks:Union[Callable, Iterable[Callable]], orders:Union[int, float, Iterable[int|float]]=0):
        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        if isinstance(orders, int):
            orders = [orders] * len(tasks)   # type: ignore
        for task, order in zip(tasks, orders):  # type: ignore
            self.addForeverTask(task, order)
            
    def addForeverTask(self, task:Callable, order:int|float=0):
        self.addListener(task, order)   # type: ignore

    def _execute(self, *args):
        if self._hasEmitted:
            return
        self._hasEmitted = True
        ignoreErr = self._ignoreErr
        allTasks = []
        for task in self.events:
            heapq.heappush(allTasks, task)
        for temptask in self.tempEvents:
            heapq.heappush(allTasks, temptask)
        for task in allTasks:
            try:
                task(*args)
            except RuntimeError as e:
                if 'deleted' in str(e):
                    # C/C++ object has been deleted(usually occurs in PYQT)
                    self.removeTempListener(task)
                else:
                    raise e
            except Exception as e:
                if not ignoreErr:
                    raise e
        self._tempEvents.clear()
        
    def execute(self, *args, ignoreErr=True):
        if not self._noCheck:
            self._check_invoke_params(*args)
        self._hasEmitted = False
        self._ignoreErr = ignoreErr
        if not self._useQtSignal:
            self._execute(*args)
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

__all__ = ["Event", "ListenerNotFoundError", "NoneTypeNotSupportedError", "DelayEvent", "Tasks", "AutoSortTask"]


if __name__ == '__main__':  # for debugging
    task = AutoSortTask()
    task.addTask(lambda : print('x'), 1.1)
    task.addTask(lambda : print('y'), 1.0)
    task.execute()  # y, x