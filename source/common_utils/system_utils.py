import os
import socket
import platform

def is_windows():
    return platform.system() == 'Windows'

def is_linux():
    return platform.system() == 'Linux'

def is_mac():
    return platform.system() == 'Darwin'

__all__ = ['is_windows', 'is_linux', 'is_mac']

def get_available_port():
    '''Return a port that is available now'''
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port

def check_port_is_using(port:int)->bool:
    '''Check if the port is using'''
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

__all__.extend(['get_available_port', 'check_port_is_using'])