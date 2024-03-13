# -*- coding: utf-8 -*-
'''
Utils for building/reading globally unique values.
If ".env" exist, this module will load it and set the environment variables.
'''

import sys, os
from typing import Dict, Any, Callable


_needInit = True
_moduleName = __name__.split('.')[-1]
_globalValues: Dict[str, Any] = {}

for module in sys.modules.keys():
    if module == __name__:
        break
    modulename = module.split('.')[-1] if '.' in module else module
    if modulename == _moduleName:
        _globalValues: Dict[str, Any] = sys.modules[module]._globalValues
        _needInit = False
        break
    
if _needInit:
    from dotenv import load_dotenv
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    if os.path.exists(path):
        load_dotenv(path)
    if os.path.dirname(path) in ('source', 'src', 'code', 'codes', 'project', 'projects', 'workspace', 'script', 'scripts', 'tool', 'tools', 'utils', 'util', 'utils', 'util', 'common', 'commons', 'commonly'):
        # try to load .env from project root
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
        if os.path.exists(path):
            load_dotenv(path)

from typing import TypeVar, Type, Optional

T = TypeVar('T')

def GetEnv(key: str, default: Optional[T] = None, type: Type[T] = str)->Optional[T]:
    '''
    Get value from os.environ.
    :param key: the key of the value
    :param default: default value if key not found
    :param type: type of the value(try to convert to this type, if failed, return default)
    '''
    try:
        val = os.environ[key]
    except KeyError:
        return default
    try:
        return type(val)
    except ValueError:
        return default

def SetGlobalValue(key: str, value: object):
    _globalValues[key] = value

def GetGlobalValue(key: str, default=None):
    '''return default if key not found.'''
    try:
        return _globalValues[key]
    except KeyError:
        return default

def RemoveGlobalValue(key: str):
    try:
        _globalValues.pop(key)
    except KeyError:
        pass

def HasGlobalValue(key: str):
    return key in _globalValues

def ClearGlobalValue():
    _globalValues.clear()

def GetGlobalValueKeys():
    return tuple(_globalValues.keys())

def GetGlobalValueValues():
    return tuple(_globalValues.values())

def GetGlobalValueItems():
    return tuple(_globalValues.items())

def GetGlobalValueDict():
    '''Return a copy of global value dict.'''
    return _globalValues.copy()

def GetOrAddGlobalValue(key: str, defaultValue:object):
    if key in _globalValues:
        return _globalValues[key]
    else:
        _globalValues[key] = defaultValue
        return defaultValue
def GetOrCreateGlobalValue(key: str, creator:Callable, *args, **kwargs):
    if key in _globalValues:
        return _globalValues[key]
    else:
        _globalValues[key] = creator(*args, **kwargs)
        return _globalValues[key]

__all__ = [ 'SetGlobalValue', 'GetGlobalValue', 'RemoveGlobalValue', 'ClearGlobalValue', 'GetGlobalValueKeys', 'GetGlobalValueValues', 'GetGlobalValueItems', 'GetGlobalValueDict',
            'HasGlobalValue', 'GetOrAddGlobalValue', 'GetOrCreateGlobalValue']