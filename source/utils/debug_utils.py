import logging
import colorama
from functools import partial
from .global_utils import GetEnv

# region internal
logging.addLevelName(100, 'SUCCESS')    # to ensure SUCCESS level is added
logging.SUCCESS = 100   # type: ignore


_default_log_format: str = GetEnv('DEFAULT_LOG_FORMAT', '[%(levelname)s](%(name)s) %(asctime)s - %(message)s') # type: ignore
_default_date_format: str = GetEnv('DEFAULT_LOG_DATE_FORMAT', '%H:%M:%S')  # type: ignore
_default_formatter = logging.Formatter(fmt=_default_log_format, datefmt=_default_date_format)
class _ColorStreamHandler(logging.StreamHandler):
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
        
    def _format(self, record:logging.LogRecord):
        return _default_formatter.format(record)

class _SuccessLogger(logging.Logger):
    def success(self, msg, *args, **kwargs):
        pass

def _log_success(self, message: str, *args, **kws):
    if self.isEnabledFor(logging.SUCCESS):  # type: ignore
        self._log(logging.SUCCESS, message, args, **kws)    # type: ignore

def _logger_modify(logger: logging.Logger)->_SuccessLogger:
    logger.success = partial(_log_success, logger)  # type: ignore
    return logger   # type: ignore

# endregion

EngineEditorLogger: _SuccessLogger = _logger_modify(logging.getLogger("Engine.Editor"))
'''Logger which is active during editor mode. Will be ignore in release mode.'''

EngineRuntimeLogger: _SuccessLogger = _logger_modify(logging.getLogger("Engine.Runtime"))
'''Logger is active during runtime. It will not be ignore in any mode.'''

DefaultLogger: _SuccessLogger = _logger_modify(logging.getLogger())
'''Default logger'''

DefaultLogger.setLevel(logging.INFO)
DefaultLogger.handlers = [_ColorStreamHandler()]  # type: ignore


__all__ = ['EngineEditorLogger', 'EngineRuntimeLogger', 'DefaultLogger']