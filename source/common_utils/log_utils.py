import datetime


class ANSI:
    BLACK = '\033[30m'  # basic colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_BLACK = '\033[90m'  # bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    F = '\033[F'
    K = '\033[K'
    NEWLINE = F + K


def gettime(day: bool = True, time: bool = True, sep: str = ' - ', daysep: str = '/', timesep: str = ':'):
    now = datetime.now()
    d = f"{now.year}{daysep}{now.month}{daysep}{now.day}"
    t = f"{now.hour}{timesep}{now.minute}{timesep}{now.second}"
    if day and time:
        return d + sep + t
    elif day:
        return d
    elif time:
        return t


def stylize(string: str, *ansi_styles, format_spec: str = "", newline: bool = False) -> str:
    """
    Stylize a string by a list of ANSI styles.
    """
    if not isinstance(string, str):
        string = format(string, format_spec)
    if len(ansi_styles) == 0:
        return string
    ansi_styles = ''.join(ansi_styles)
    if newline:
        ansi_styles = ANSI.NEWLINE + ansi_styles
    return ansi_styles + string + ANSI.RESET


def warn(msg: str, *args, **kwargs):
    print('[WARNING] ' + stylize(msg, ANSI.YELLOW), *args, **kwargs)


def debug(msg: str, *args, **kwargs):
    print('[DEBUG] ' + stylize(msg, ANSI.BOLD, ANSI.BRIGHT_WHITE), *args, **kwargs)


def info(msg: str, *args, **kwargs):
    print('[INFO] ' + stylize(msg, ANSI.BRIGHT_CYAN), *args, **kwargs)


def error(msg: str, *args, **kwargs):
    print('[ERROR] ' + stylize(msg, ANSI.BRIGHT_RED), *args, **kwargs)


def success(msg: str, *args, **kwargs):
    print('[INFO] ' + stylize(msg, ANSI.BRIGHT_GREEN), *args, **kwargs)
