# -*- coding: utf-8 -*-
'''
Utils for building/reading globally unique values.
If ".env" exist, this module will load it and set the environment variables.
'''

import sys, os
import json
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

def _env_type_convert(val_str: str, valType: type):
    if issubclass(valType, str):
        return valType(val_str)
    if issubclass(valType, bool):
        val_str = val_str.lower()
        val = (val_str == 'true' or val_str == '1' or val_str == 'yes' or val_str ==
               'y' or val_str == 'on' or val_str.startswith('enable') or val_str == 'ok')
        return valType(val)
    if issubclass(valType, (float, int)):
        return valType(val_str)
    if issubclass(valType, (list, tuple)):
        try:
            val = json.loads(val_str)
            return valType(val)
        except:
            try:
                val = eval(val_str)
                return valType(val)
            except:
                try:
                    val = re.split(r'[\s,]+', val_str)
                    return valType(val)
                except:
                    return valType([val_str,])
    if issubclass(valType, dict):
        try:
            val = json.loads(val_str)
            return valType(val)
        except:
            try:
                val = eval(val_str)
                return valType(val)
            except:
                raise ValueError(f'Cannot convert {val_str} to {valType.__name__}')
    return valType(val_str)

T = TypeVar('T')

# region env
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
        return _env_type_convert(val)   # type: ignore
    except ValueError:
        return default
    except TypeError:
        return default

def is_game_mode()->bool:
    '''
    Game mode means the engine should be ran without editor window.
    
    Default is True.
    '''
    if 'GAME_MODE' in os.environ:
        mode: bool = GetEnv('GAME_MODE', False, bool)
        if not mode:
            if 'EDITOR_MODE' in os.environ:
                return not GetEnv('EDITOR_MODE', False, bool)
        return mode
    elif 'EDITOR_MODE' in os.environ:
        return not GetEnv('EDITOR_MODE', False, bool)
    return True # default is game mode

def is_editor_mode():
    '''
    Editor mode means the engine in running with editor window.
    
    Default is False.
    '''
    return not is_game_mode()

def is_release_mode():
    '''
    Release mode means the application is built and released.
    
    Note: 
        Release mode != game mode/editor mode. 
        There has no direct relationship between them.
        
    Default is False.
    '''
    if 'RELEASE_MODE' in os.environ:
        mode: bool = GetEnv('RELEASE_MODE', False, bool)
        if not mode:
            if 'DEV_MODE' in os.environ:    # if dev mode is set, release mode will be overrided
                return not GetEnv('DEV_MODE', False, bool)
        return mode
    elif 'DEV_MODE' in os.environ:
        return not GetEnv('DEV_MODE', False, bool)
    return False # default is not release mode
 
def is_dev_mode():
    '''
    Development mode makes the engine is running with more debug information, etc.
    
    Note: 
        Debug mode != game mode/editor mode. 
        There has no direct relationship between them.
    
    Default is True.
    '''
    return not is_release_mode()

__all__ = ['GetEnv', 'is_game_mode', 'is_editor_mode', 'is_release_mode', 'is_dev_mode']
# endregion

# region global values
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

__all__.extend(['SetGlobalValue', 'GetGlobalValue', 'RemoveGlobalValue', 'ClearGlobalValue', 
                'GetGlobalValueKeys', 'GetGlobalValueValues', 'GetGlobalValueItems', 'GetGlobalValueDict',
                'HasGlobalValue', 'GetOrAddGlobalValue', 'GetOrCreateGlobalValue'])

# endregion