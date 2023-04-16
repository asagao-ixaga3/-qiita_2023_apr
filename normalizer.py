
from environment import MsdSystem

class Normalizer(object):

    def __init__(self, env: MsdSystem) -> None:
        self.env = env

    def convert(self, Y_raw):
        # Y_raw: (..., nx)
        Y = (Y_raw  - self.env.y_center)/self.env.y_scale # (..., nx)
        return Y
    
    def invert(self, Y):
        # Y: (..., nx)
        Y_raw = Y * self.env.y_scale + self.env.y_center # (..., nx)
        return Y_raw