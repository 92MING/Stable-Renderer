import os
import sys
import logging
import colorama
import warnings

from typing import ClassVar, Union
from functools import partial
from concurrent_log_handler import ConcurrentTimedRotatingFileHandler as _RotatingFileHandler

if __name__ == '__main__':  # for debugging
    import sys, os
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(_proj_path)
    __package__ = 'common_utils'

from .global_utils import GetEnv, is_dev_mode, is_editor_mode


# region internal
if os.name == "nt":
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

_warning_level: str = GetEnv('WARNING_LEVEL', 'default')  # type: ignore
warnings.filterwarnings(_warning_level) # type: ignore

logging.addLevelName(100, 'SUCCESS')    # to ensure SUCCESS level is added
logging.SUCCESS = 100   # type: ignore

# log settings
# TODO: modify settings with YAML instead of using env
_default_log_format: str = GetEnv('DEFAULT_LOG_FORMAT', '[%(levelname)s](%(name)s) %(asctime)s | %(message)s') # type: ignore
_default_date_format: str = GetEnv('DEFAULT_LOG_DATE_FORMAT', '%H:%M:%S')  # type: ignore
_default_formatter = logging.Formatter(fmt=_default_log_format, datefmt=_default_date_format)

_root_logger = logging.getLogger()

class _PassToRootLogger(logging.Logger):
    '''
    Logger that do nothing when calling, and pass a formatted record to root logger.
    
    No matter it is propagated or not, its formatted msg will always pass to root logger.
    '''
    
    _DEFAULT_FORMATTER = logging.Formatter(fmt="%(message)s")
    
    def addHandler(self, hdlr: logging.Handler) -> None:
        if len(self.handlers) >0:
            raise ValueError('PassToRootLogger can only have one handler')
        return super().addHandler(hdlr)
    
    def handle(self, record: logging.LogRecord) -> None:
        if self.disabled:
            return
        maybe_record = self.filter(record)
        if not maybe_record:
            return
        if isinstance(maybe_record, logging.LogRecord):
            record = maybe_record
        if record.args:
            if len(self.handlers)>0:
                record.msg = self.handlers[0].format(record)
            else:
                record.msg = self._DEFAULT_FORMATTER.format(record)
            record.args = tuple() # no args is allowed in root logger
        record.msg = record.msg.strip()
        self.callHandlers(record)
        
    def callHandlers(self, record:logging.LogRecord) -> None:
        '''Will just pass the record to root logger's handlers.'''
        root = logging.getLogger()
        for handler in root.handlers:   # pass to root logger directly
            handler.handle(record)

logging.setLoggerClass(_PassToRootLogger)   # make all loggers pass to root logger

class _RootHandler(logging.Handler):
    
    _Formatter: ClassVar[logging.Formatter] = _default_formatter
    '''Formatter for root handler'''
    
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()  # return msg only, cuz other info is formatted in child handlers
    
    def _format(self, record):
        '''real formatter'''
        return self._Formatter.format(record)

class _ColorStreamRootHandler(logging.StreamHandler, _RootHandler):
    
    _ColorDict = {
        'DEBUG': colorama.Fore.WHITE,
        'INFO': colorama.Fore.BLUE,
        'WARNING': colorama.Fore.YELLOW,
        'ERROR': colorama.Fore.RED,
        'CRITICAL': colorama.Fore.RED + colorama.Style.BRIGHT,
        'SUCCESS': colorama.Fore.GREEN + colorama.Style.BRIGHT,
    }
    
    def emit(self, record:logging.LogRecord):
        levelname = record.levelname
        level_color = self._ColorDict.get(levelname, None)
        record.args = tuple()
        if level_color:
            record.msg = colorama.Style.RESET_ALL + record.msg
        record.msg = self._format(record)
        if level_color:
            record.msg = level_color + record.msg
        super().emit(record)
        
class _ConcurrentTimedRotatingFileHandler(_RotatingFileHandler, _RootHandler):
    
    def emit(self, record:logging.LogRecord):
        record.args = tuple()
        record.msg = self._format(record)
        super().emit(record)

if is_dev_mode():
    if is_editor_mode():
        _root_handler = _ColorStreamRootHandler()    # TODO: logs should print on UI directly
    else:
        _root_handler = _ColorStreamRootHandler()
else:
    from datetime import datetime
    rot_hr_interval: int = GetEnv('LOG_ROTATION_INTERVAL_HOUR', 1)  # type: ignore
    backup_count: int = GetEnv('LOG_BACKUP_COUNT', 5, type=int)  # type: ignore
    zip_old_logs: bool = GetEnv('LOG_ZIPPING', False, type=bool)  # type: ignore
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs', datetime.now().strftime("%Y_%m_%d-%H_%M_%S")))
    os.makedirs(log_dir, exist_ok=True)
    _root_handler = _ConcurrentTimedRotatingFileHandler(filename=os.path.join(log_dir, 'log.log'),  
                                                       backupCount=backup_count, encoding='utf-8',
                                                       use_gzip=zip_old_logs, 
                                                       interval=rot_hr_interval,
                                                       when='h')

_log_level: str = GetEnv('LOG_LEVEL', 'DEBUG' if is_dev_mode() else 'INFO')  # type: ignore
_root_handler.setLevel(_log_level)

_root_logger.handlers = [_root_handler, ]

class _ModifiedLogger(logging.Logger):
    '''just a type hint for logger with success method.'''
    def success(self, msg, *args, **kwargs): ...
    
    def print(self, *args, level:Union[str, int]=logging.DEBUG, sep:str=" "): ...
    
def _log_success(self: logging.Logger, message: str, *args, **kws):
    if self.isEnabledFor(logging.SUCCESS):  # type: ignore
        self._log(logging.SUCCESS, message, args, **kws)    # type: ignore

def _log_print(self: logging.Logger, *args, level:Union[str, int]=logging.DEBUG, sep:str=" "):
    if isinstance(level, str):
        level = get_log_level_by_name(level)
    if self.isEnabledFor(level):  # type: ignore
        self._log(level, sep.join(map(str, args)), (), {})    # type: ignore
    
def _logger_modify(logger: logging.Logger)->_ModifiedLogger:
    logger.success = partial(_log_success, logger)  # type: ignore
    logger.print = partial(_log_print, logger)  # type: ignore
    return logger   # type: ignore
# endregion

def get_log_level_by_name(name:str)->int:
    '''Get log level by name'''
    return getattr(logging, name.upper())

EditorLogger: _ModifiedLogger = _logger_modify(logging.getLogger("Engine.Editor"))
'''Logger for editor(UI). It will be ignore in some modes, e.g. release mode'''

EngineLogger: _ModifiedLogger = _logger_modify(logging.getLogger("Engine.Runtime"))
'''Logger for rendering engine. It will only be ignore when u are running the ComfyUI directly.'''

ComfyUILogger: _ModifiedLogger = _logger_modify(logging.getLogger("Engine.ComfyUI"))
'''Logger specifically for comfyUI'''

EditorLogger.setLevel(_log_level)
EngineLogger.setLevel(_log_level)
ComfyUILogger.setLevel(_log_level)

DefaultLogger: _ModifiedLogger = _logger_modify(_root_logger)
'''Default logger. You can use it for any purpose.'''


__all__ = ['get_log_level_by_name', 'EditorLogger', 'EngineLogger', 'ComfyUILogger', 'DefaultLogger']