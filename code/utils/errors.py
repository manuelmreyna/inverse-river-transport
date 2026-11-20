import numpy as np
import forward.laplace
    
def RMSE_func(t,bt0,bt1):
    deltat = np.gradient(t)
    P = bt0/np.sum(bt0*deltat)
    Q = bt1/np.sum(bt1*deltat)
    return np.sqrt(np.sum((P-Q)**2*deltat))

def RMSD_func(t,bt0,bt1,return_a = False):
    P = bt0
    Q = bt1
    a = np.trapezoid(P * Q,t) / np.trapezoid(Q**2,t)
    if return_a:
        return np.sqrt(np.trapezoid((P-a*Q)**2,t)/np.trapezoid(P**2,t)),a
    else:
        return np.sqrt(np.trapezoid((P-a*Q)**2,t)/np.trapezoid(P**2,t))

def KL_div_func(t,bt0,bt1):
    epsilon = 10**-30
    t,bt0,bt1 = t[(bt0>epsilon)&(bt1>epsilon)], bt0[(bt0>epsilon)&(bt1>epsilon)],bt1[(bt0>epsilon)&(bt1>epsilon)]
    if len(t)>1:
        deltat = np.gradient(t)
        P = bt0*deltat/np.sum(bt0*deltat)
        Q = bt1*deltat/np.sum(bt1*deltat)
        return np.sum(P*np.log(P/Q))
    else:
        return np.nan

def compute_errors(t, bt0, x, params, memory_func, bound_cond):
    btc_est = forward.laplace.ADEMF_1D(t[t>0], [x], params[0], params[1], params[2], [1,params[3]], memory_func, bound_cond)[:,-1]
    return RMSE_func(t[t>0],bt0[t>0],btc_est), KL_div_func(t[t>0],bt0[t>0],btc_est)