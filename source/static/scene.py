import os.path

from utils.base_clses import NamedObj
import inspect
from typing import Literal

class Scene(NamedObj):

    def prepare(self):
        '''Override this method to prepare your scene.'''
        raise NotImplementedError

    @classmethod
    def Load(cls, path):
        '''Load a scene from a file.'''
        basename = os.path.basename(path)
        name = os.path.splitext(basename)[0]
        with open(path, 'r') as f:
            source = f.read()
        scene = cls(name)
        scene.deserialize(source)
        return scene

    def serialize(self, format:Literal['python','json']='python')->str:
        '''Serialize this scene to a file.'''
        if format == 'python':
            code = inspect.getsource(self.prepare)
            codeLines = [line.strip() for line in code.split('\n')][2:]
            codeLines.insert(0, '# Python code')
            code = '\n'.join(codeLines)
            return code
        else:
            # TODO: implement this
            raise NotImplementedError
    def deserialize(self, source:str):
        '''Deserialize this scene from source str.'''
        lines = source.split('\n')
        if 'Python' in lines[0]:
            self.prepare = lambda : exec(source)
        elif 'JSON' in lines[0]:
            # TODO: implement this
            raise NotImplementedError
        else:
            raise Exception('Unknown format')
    def save(self, path, overwrite=False):
        '''Save this scene to a file.'''
        if os.path.exists(path):
            if os.path.isdir(path):
                path = os.path.join(path, f'{self.name}.scene')
                if os.path.exists(path) and not overwrite:
                    raise Exception(f'File {path} already exists.')
                with open(path, 'w') as f:
                    f.write(self.serialize())
            elif not overwrite:
                raise Exception(f'File {path} already exists.')
            else:
                with open(path, 'w') as f:
                    f.write(self.serialize())
        else:
            with open(path, 'w') as f:
                f.write(self.serialize())

    def __repr__(self):
        return f"<Scene {self.name}>"
    def __str__(self):
        return self.__repr__()
