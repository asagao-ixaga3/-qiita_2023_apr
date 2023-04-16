import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


def integral_two_gaussian_functions(mu1,s1,mu2,s2):
    """
    from sympy import *

    x = Symbol("x")
    mu1 = Symbol("mu1", real = True)
    s1 = Symbol("s1", real = True, positive = True)

    mu2 = Symbol("mu2", real = True)
    s2 = Symbol("s2", real = True, positive = True)

    f = exp(-(x-mu1)**2/(2*s1**2))/s1/sqrt(2*pi)
    g = exp(-(x-mu2)**2/(2*s2**2))/s2/sqrt(2*pi)

    e = simplify(integrate(f * g, (x,-oo, oo)))
    """
    from numpy import exp, pi, sqrt
    e = sqrt(2)*exp((-mu1**2 + 2*mu1*mu2 - mu2**2 + s2**2*(mu1 - mu2)**2/(s1**2 + s2**2))/(2*s1**2))/(2*sqrt(pi)*sqrt(s1**2 + s2**2))
    return e 

def fit_mixture_model(X, y):
    # X: (n_sample, d)
    # y: (n_sample,)    

    def objective(theta):
        # theta: (d,)
        err = y - X @ theta # (*,)
        return np.sum(err**2) / max(n_sample, 1)

    def jacobian(theta):
        # theta: (d,)        
        jac = (2*(X.T@X)@theta - 2 * X.T@y)/max(n_sample, 1) # (d,)
        return jac

    def constraint(theta):
        return np.sum(theta) - 1.

    def constraint_jac(theta):
        return np.ones(theta.shape) # (d,)

    n_sample, d = X.shape
        
    bounds = [(0.,1.0,) for _ in range(d)]

    x0 = np.ones(d)/d

    problem = {'fun': objective, 'jac': jacobian, 'args': (), 'constraints': 
               {'type': 'eq', 'fun': constraint, 'jac': constraint_jac}, 
               'bounds': bounds}

    result = minimize(**problem, method='SLSQP', options={'disp': False}, x0=x0)

    return result.x, result

def train_gaussian_process(X, y):
    # X: (n_sample, n_feat), y: (n_sample, n_target) or (n_sample,)
    
    kernel = ConstantKernel() \
        + ConstantKernel() * RBF(length_scale_bounds = (1e-5,1e+1)) \
        + WhiteKernel(noise_level=1.0)
    gaussian_process = GaussianProcessRegressor(kernel=kernel)
    gaussian_process.fit(X, y)

    alpha_sq, beta_sq, scale = [gaussian_process.kernel_.get_params()[key] 
                                for key in ("k1__k1__constant_value", 
                                            "k1__k2__k1__constant_value", 
                                            "k1__k2__k2__length_scale") ]
    weight = gaussian_process.alpha_ # (n_samsple,n_target) or (n_samsple,)
    n_feat = gaussian_process.n_features_in_

    coef_ = weight * (np.sqrt(2*np.pi)*scale)**n_feat * beta_sq 
    # (n_samsple,n_target) or (n_samsple,)
    intercept_ = np.sum(weight, axis=0) * alpha_sq # (n_target,)

    return coef_, intercept_, scale, gaussian_process
    # g = lambda x: multivariate_normal(
    #       mean=np.zeros(n_feat), cov=scale**2).pdf(
    #       x[:,None,:]-X[None,:,:]) @ coef_ + intercept_
    # => g(X_given) == gaussian_process.predict(X_given)