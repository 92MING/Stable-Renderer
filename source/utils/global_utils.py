# -*- coding: utf-8 -*-
'''常用的全局变量操作, 無論如何import此模塊, 都會得到同一個全局變量字典。不涉及global()或者os.environ. 如果有.env, 會自動讀取。'''

import sys, os

_needInit = True
_moduleName = __name__.split('.')[-1]
for module in sys.modules.keys():
    if module == __name__:
        break
    modulname = module.split('.')[-1] if '.' in module else module
    if modulname == _moduleName:
        _globalValues = sys.modules[module]._globalValues
        _needInit = False
        break
if _needInit:
    _globalValues = {}
    from dotenv import load_dotenv
    load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')))

def SetGlobalValue(key: str, value: object):
    _globalValues[key] = value
def GetGlobalValue(key: str):
    try:
        return _globalValues[key]
    except KeyError:
        return None
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

__all__ = [ 'SetGlobalValue', 'GetGlobalValue', 'RemoveGlobalValue', 'ClearGlobalValue', 'GetGlobalValueKeys', 'GetGlobalValueValues', 'GetGlobalValueItems', 'GetGlobalValueDict',
            'HasGlobalValue', 'GetOrAddGlobalValue' ]