from .manager import Manager

class SceneManager(Manager):
    def __init__(self, mainScene=None, scenes=()):
        super().__init__()
        self._mainScene = mainScene
        self._scenes = list(scenes)
        if mainScene is not None and mainScene not in self._scenes:
            self._scenes.append(mainScene)
        if mainScene is None and len(self._scenes)!=0:
            self._mainScene = self._scenes[0] # the first scene is the main scene if mainScene is not specified

    @property
    def MainScene(self):
        return self._mainScene
    def SwitchToScene(self, scene):
        if scene == self.MainScene:
            return
        # TODO: switch to scene
        pass

    def prepare(self):
        if self.MainScene is not None:
            self.MainScene.prepare()

__all__ = ['SceneManager']