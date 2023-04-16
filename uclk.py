import numpy as np
from scipy.stats import multivariate_normal

from utils import (fit_mixture_model, integral_two_gaussian_functions,
                   train_gaussian_process)

class UclkAgent(object):

    def __init__(self, nx, na, p_scale = 0.5, p_dim = 2**7, **kwargs) -> None:

        # feature map parameters
        self.na = na
        self.p_scale = p_scale
        self.p_dim = p_dim
        self.Theta = np.ones((p_dim,na))/p_dim # (p_dim,na)

        Fg = np.random.randn(p_dim,nx,nx+1)/np.sqrt(nx+1)
        self.F = Fg[:,:,:-1] # (p_dim,nx,nx)
        self.g = Fg[:,:,-1] # (p_dim,nx,)

        self.Theta = np.ones((p_dim,na))/p_dim # (p_dim,na)

        self.q_gp = None
        self.v_gp = None

        self.V = None
        self.v_dim = None
        self.S_tld = None
        self.v_scale = None

    def feat_map(self, X, X0, scale):
        # X: (..., nx)
        # X0: (v_dim, nx)
        nx = X.shape[-1]
        v_dim = X0.shape[0]
        return multivariate_normal(mean=np.zeros(nx), cov=scale**2).pdf(
            X[...,None,:]-X0[None,:,:]).reshape(*X.shape[:-1], v_dim) 
            # (..., v_dim)

    def inner_product(self, S, S0, scale):
        # S: (N, nx)
        # S0: (v_dim, nx)
        # output: (N, p_dim, v_dim+1)

        mu1 = S @ self.F  + self.g[:,None,:] 
        # (N,nx) @ (p_dim,nx,nx) -> (p_dim,N,nx)
        # (p_dim,N,nx) + (p_dim,1,nx) -> (p_dim,N,nx)
        mu2 = S0[:,None,:]
        # (N,nx) - (v_dim,1,nx) -> (v_dim,N,nx)

        s1 = self.p_scale
        s2 = scale

        tmp = integral_two_gaussian_functions(
            mu1[:,None,...],s1,mu2[None,...],s2)
        # (p_dim,1,N,nx), (), (1,v_dim,N,nx), () -> (p_dim,v_dim,N,nx)
        psi_main = np.prod(tmp,axis=-1) # (p_dim,v_dim,N)
        psi_main = np.transpose(psi_main,(2,0,1)) # (N,p_dim,v_dim)
        psi_bias = np.ones(shape=(*psi_main.shape[:-1],1)) # (N,p_dim,1)
        return np.concatenate((psi_main, psi_bias), axis=-1) # (N,p_dim,v_dim+1)

    def predict_q(self, S):
        # S: (*,nx)
        if len(S.shape) > 1:
            return self.q_gp.predict(S) # (N,na)
        elif len(S.shape) == 1:
            return self.q_gp.predict(S.reshape(1,-1)).reshape(-1) # (na,)
        else:
            raise NotImplementedError()        

    def predict_v(self, S):
        # S: (*,nx)
        if len(S.shape) > 1:
            return self.v_gp.predict(S) # (N,)
        elif len(S.shape) == 1:
            return self.v_gp.predict(S.reshape(1,-1)).squeeze() # (,)
        else:
            raise NotImplementedError()        

    def predict_next_state(self, S, A):

        if len(S.shape) == 2 and len(A.shape) == 1:
            # S: (N, nx), A: (N,)
            mu1 = S @ self.F  + self.g[:,None,:] 
            # (N,nx) @ (p_dim,nx,nx) -> (p_dim,N,nx)
            # (p_dim,N,nx) + (p_dim,1,nx) -> (p_dim,N,nx)
        elif len(S.shape) == 1 and len(A.shape) == 0:
            # S: (N, ), A: (,)
            mu1 = S @ self.F  + self.g 
            # (nx) @ (p_dim,nx,nx) -> (p_dim,nx)
            # (p_dim,nx) + (p_dim,nx) -> (p_dim,nx)
        else:
            raise NotImplementedError()

        return np.sum(self.Theta[:,A,None] * mu1, axis=0) # (N,nx)
        # Theta[:,A,None]: (p_dim,N,1)
        # self.Theta[:,A,None] * mu1: (p_dim,N,nx)
        # np.sum(self.Theta[:,A,None] * mu1, axis=0): (nx,N)

    def sample_next_state(self, S, A, n_sample = 1):

        if len(S.shape) == 2 and len(A.shape) == 1:
            raise NotImplementedError()
        elif len(S.shape) == 1 and len(A.shape) == 0:
            # S: (N, ), A: (,)
            i = np.random.choice(np.arange(self.p_dim), p=self.Theta[:,A], 
                             size=(n_sample)) # (n_sample,)  
            mu = S @ self.F[i,...]  + self.g[i,...] # (n_sample,nx)
            S_sampled = mu + np.random.randn(*mu.shape) * self.p_scale
            return S_sampled
        else:
            raise NotImplementedError()

    def predict_v_at_next_step(self, S, A):
        # S: (N, nx), A: (N,)
        # V_next: (N,) predicted value at the next state of S driven by A
        # , namely Exp[V(s(t+1))|s(t),a(t)].

        if len(S.shape) == 1 and len(A.shape) == 0:
            return self.predict_v_at_next_step(S.reshape(1,-1), A.reshape(-1))[0] #(,)

        Psi = self.inner_product(S, self.S_tld, self.v_scale)
        # (N, p_dim, v_dim+1)
        Q_next = (Psi @ self.V) @ self.Theta
        # ((N, p_dim, v_dim+1) @ (v_dim+1,)) @ (p_dim, na) -> (N, na) 
        V_next = Q_next[np.arange(len(A)), A] # (N,)
        return V_next # (N,)

    def fit(self,S0, A0, S1, S_tld, R_tld, n_rounds=1, gamma = 0.9, 
            n_itr = 2**4, **kwargs):
        # S0,S1: (N, nx), A0: (N,), S_tld: (N_tld,nx), R_tld: (N_tld, na)
        
        p_dim = self.p_dim
        na = self.na

        N, nx = S0.shape
        N_tld = S_tld.shape[0]
        assert S1.shape == (N, nx)
        assert S_tld.shape == (N_tld, nx)
        assert A0.shape == (N,)
        assert na > 1

        V = np.zeros(N_tld+1) # (N_tld+1,)
        Q = np.zeros((N_tld+1,na)) # (N_tld+1,na)
        Theta = np.ones((p_dim,na))/N_tld # (p_dim,na)

        v_scale = self.p_scale
                
        for _ in range(n_itr):
            # run EVI process            
            for _ in range(n_rounds):                

                Psi_tld = self.inner_product(S_tld, S_tld, v_scale) 
                # (N_tld,p_dim,N_tld+1)
                Y = R_tld + gamma * (Psi_tld @ V) @ Theta # (N_tld,na)     
                coef_, intercept_, _, q_gp = \
                    train_gaussian_process(S_tld, Y)
                # coef_: (N_tld, na), intercept_: (na,)
                Q[:-1,:] = coef_ # (N_tld, na)
                Q[-1,:] = intercept_ # (na,)
                if na > 1:
                    tol = 1e-8
                    if np.any(np.abs(Q[:,0] - Q[:,-1]) < max(tol * np.max(np.abs(Q)), tol)):
                        print("Warning: the values of Q function equals over actions.")
                Y = np.max(q_gp.predict(S_tld), axis=-1) # (N_tld,)
                coef_, intercept_, v_scale, v_gp = \
                    train_gaussian_process(S_tld, Y)
                V[:-1] = coef_ # (N_tld,)
                V[-1] = intercept_ # (,)

            # Solve theta
            Y = v_gp.predict(S1) # (N,)
            Psi0 = self.inner_product(S0, S_tld, v_scale) 
            # (N_tld,p_dim,N_tld+1)
            X = Psi0 @ V # (N,p_dim)
            for a in range(na):                
                Ya = Y[A0==a] # (*,)
                Xa = X[A0==a,:] # (*,p_dim)
                theta, res = fit_mixture_model(Xa, Ya)
                assert res.success, res.message
                Theta[:,a] = theta[:]

        self.Theta[:,:] = Theta[:,:] # (p_dim,na)

        self.v_gp = v_gp
        self.q_gp = q_gp

        self.V = V
        self.S_tld = S_tld.copy()
        self.v_scale = v_scale
        self.v_dim = N_tld