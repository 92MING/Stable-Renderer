from runtime.component import Component

class Transform(Component):
    '''記載物件的位置、旋轉、縮放等資訊、操作'''
    def __init__(self, gameObj:'GameObject'):
        super().__init__(gameObj)
    @property
    def enable(self):
        return True
    @enable.setter
    def enable(self, value):
        if not value:
            raise Exception('Transform can not be disabled.')

    @property
    def parent(self)->'Transform':
        '''return the transform on parent gameObject'''
        if self.gameObj