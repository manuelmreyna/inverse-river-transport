import scipy
import numpy as np
def ADEMF_1D(t, btc, x, memory_func, bound_cond, params0 = None, bounds = None, reg_v = None, v_ref = None):
    """
    Estimates parameters of measured breakthrough curves by fitting in Laplace domain.
    - t (array): Times of measurement of concentrations
    - btc (array): Measured concentrations (same length as t)
    - memory_func (str): Either 'first order' or 'power law', others can be added editing beta_g_est_lap
    - bound_cond (str): Either 'semi-infinite-conc', 'semi-infinite-mixed', or 'infinite'
                'semi-infinite-conc' is a domain x>0 with vc(x=0,t)=M_0 delta(t)
                'semi-infinite-mixed' is a domain x>0 with vc(x=0,t)- 2 D dc/dx (x=0,t) =M_0 delta(t)
                'infinite' is a domain x>0 with vc(x=0,t)-2 D dc/dx (x=0,t) =M_0 delta(t). The solution to this problem is equivalent to the solution in the infinite domain
    - params0 (array of length 4): Initial guess of parameters. Default is v_ref and geometric mean of bounds. If bounds are 0 or infinity, it is 1.0.
    - bounds (tuple): Lower and upper bounds for the parameters (each is an array of length 4). Default is [v_ref*0.9,v_ref*1.5], and all zeros and infinity for first order, but 0.01 to 1.0 for beta*alpha, and 0.1 to 1.0 for 1-gamma.
    :returns: Array of estimated independent parameters [v,Pe,beta*k_f/v*x,k_r/v*x] for first order and [v,Pe,beta*alpha/v*x,1-gamma] for power law (all parameters but the velocity are dimensionless)
    """
    if v_ref is None:
        v_ref = x/t[np.argmax(btc)]
        
    if bounds is None:
        if memory_func == 'first order':
            bounds = (np.array([v_ref*0.9,20.0,0.0,0.0]),np.array([v_ref*1.5,20000.0,np.inf,np.inf]))
        if memory_func == 'power law':
            bounds = (np.array([v_ref*0.9,20.0,0.01,0.1]),np.array([v_ref*1.5,20000.0,1.0,1.0]))
    
    if params0 is None:
        params0 = np.concatenate(([v_ref],np.where((bounds[0][1:]>0.0) & (bounds[1][1:]<np.inf), np.sqrt(bounds[0][1:]*bounds[1][1:]), np.ones(3))))
        
    step = np.min(t[1:]-t[:-1])
    s = np.linspace(1/t[-1],1/(step),10000)
    meas_bt_lap = np.trapz(btc[:,None]*np.exp(-s*t[:,None]),x= t,axis = 0)
    meas_bt_lap_0 = np.trapz(btc[:,None]*np.exp(-0*t[:,None]),x= t,axis = 0)
    log_meas_bt_lap = (np.log(meas_bt_lap)-np.log(meas_bt_lap_0))
    f_meas = log_meas_bt_lap
    if memory_func == 'first order':
        beta_g_est_lap = lambda v,beta_prime,param_g: beta_prime/(param_g+s*x/v)
    if memory_func == 'power law':
        beta_g_est_lap = lambda v,beta_prime,param_g: beta_prime*(s*x/v)**(-param_g)
    if bound_cond == 'semi-infinite-conc':
        log_B = lambda v,Pe,beta_prime,param_g: 0
    if bound_cond == 'semi-infinite-mixed':
        log_B = lambda v,Pe,beta_prime,param_g: np.log(1/(1/2+1/(2*np.sqrt(Pe))*np.sqrt(4*(s*x/v+(s*x/v)*beta_g_est_lap(v,beta_prime,param_g))+Pe)))
    if bound_cond == 'infinite':
        log_B = lambda v,Pe,beta_prime,param_g: np.log(np.sqrt(Pe)/np.sqrt(4*(s*x/v+(s*x/v)*beta_g_est_lap(v,beta_prime,param_g))+Pe))
    f_est_lap = lambda v,Pe,beta_prime,param_g: 1/2*(Pe-np.sqrt(Pe*(4*(s*x/v+(s*x/v)*beta_g_est_lap(v,beta_prime,param_g))+Pe)))+log_B(v,Pe,beta_prime,param_g)
    if reg_v is None:
        res = lambda params: (1/np.sum(np.abs(f_meas))*(f_meas-f_est_lap(params[0],params[1],params[2],params[3])))
    else:
        res = lambda params: np.concatenate((1/np.sum(np.abs(f_meas))*(f_meas-f_est_lap(params[0],params[1],params[2],params[3])),[reg_v*(params[0]-v_ref)/v_ref]))
    params_laplace = scipy.optimize.least_squares(res,params0,bounds=bounds).x
    return params_laplace