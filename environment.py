import numpy as np

class MsdSystem(object):
    '''
    Mass spring damper system
    m\ddot{x} = -kx -c \dot{x} + f
    Use the parameters in the article:
    https://qiita.com/code0327/items/10fb56090a1e56046fa4
    '''

    nx = 2
    y_center = np.array((0., 0.)) # (nx,)
    y_scale = np.array((0.23, 0.71)) # (nx,)
    
    def __init__(self, m=1, k=10, c=1, r_state = np.random.RandomState(), dt=0.1, h=1e-2) -> None:
        self.m = m
        self.k = k
        self.c = c
        self.y = None
        self.t = None
        self.r_state = r_state
        self.dt = dt
        self.h = h

        self.log = {"Time": [], "Position": [], "Velocity": [], "Force": []}

    def _record(self, f):
        self.log["Time"].append(self.t)
        self.log["Position"].append(self.y[0])
        self.log["Velocity"].append(self.y[1])
        self.log["Force"].append(f)
    
    def reset(self, y_initial = None):
        # y[0]: position
        # y[1]: velocity

        for key in self.log:
            self.log[key].clear()

        if y_initial is None:
            r = (2*self.r_state.rand(2)-1)*3 # (2,)
            self.y = self.y_scale * r + self.y_center # (2,)
        else:
            self.y = np.array(y_initial)
        self.t = 0

        self._record(np.nan)

    def step(self, f):
        # y[0]: position
        # y[1]: velocity
        
        def diff_eq(y):
            return np.array([y[1], -self.k*y[0]-self.c*y[1]+f])
        
        def rk4(y,g,h):
            k1 = g(y)
            k2 = g(y+k1*h/2)
            k3 = g(y+k2*h/2)
            k4 = g(y+k3*h)

            return y + (k1+2*k2+2*k3+k4)/6*h

        for _ in range(round(self.dt/self.h)):
            self.y = rk4(self.y,diff_eq,self.h)
            self.t += self.h
            self._record(f)

