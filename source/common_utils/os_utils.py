import platform

def is_windows():
    return platform.system() == 'Windows'

def is_linux():
    return platform.system() == 'Linux'

def is_mac():
    return platform.system() == 'Darwin'


__all__ = ['is_windows', 'is_linux', 'is_mac']