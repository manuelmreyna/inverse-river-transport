import numpy as np
import scipy as sp
from forward.laplace import ADEMF_1D
def generate(seed,size,t_,memory_func,bound_cond,mean_log_params,cov_log_params,bounds = None):
    """
    Generate a synthetic dataset of breakthrough curves (BTCs) using a memory function model,
    parameterized by log-normal random variables. Perform Karhunen–Loève (KL) decomposition 
    to reduce the dimensionality of the dataset.

    Parameters:
    - seed (int): Random seed for reproducibility.
    - size (int): Number of samples in the synthetic dataset.
    - t_ (array): Dimensionless times at which the breakthrough curves are computed.
    - memory_func (str): Either 'first order' or 'power law', others can be added editing beta_g_est_lap
    - bound_cond (str): Either 'semi-infinite-conc', 'semi-infinite-mixed', or 'infinite'
                        'semi-infinite-conc' is a domain x>0 with vc(x=0,t)=M_0 delta(t)
                        'semi-infinite-mixed' is a domain x>0 with vc(x=0,t)- 2 D dc/dx (x=0,t) =M_0 delta(t)
                        'infinite' is a domain x>0 with vc(x=0,t)-2 D dc/dx (x=0,t) =M_0 delta(t). The solution to this problem is equivalent to the solution in the infinite domain
    - mean_log_params (array): Mean of the log-parameters (log-transformed).
    - cov_log_params (array): Covariance matrix of the log-parameters.
    - bounds (tuple): Lower and upper bounds for the parameters (each is an array).
    - n_lmbds (int): Number of KL modes to retain.

    Returns:
    - params (array): Dimensionless parameters used to create the synthetic breakthrough curves.
    - btcs (2D array): Synthetic breakthrough curves
    """
    cov_log_params_sqrt = sp.linalg.sqrtm(cov_log_params)
    if bounds == None:
        if memory_func == 'first order':
            bounds = (np.maximum(np.array([20.0,0.0,0.0]),np.exp(np.diagonal(mean_log_params-cov_log_params_sqrt*4.0))),np.minimum(np.array([20000.0,np.inf,np.inf]),np.exp(np.diagonal(mean_log_params+cov_log_params_sqrt*4.0))))
        if memory_func == 'power law':
            bounds = (np.maximum(np.array([20.0,0.01,0.1]),np.exp(np.diagonal(mean_log_params-cov_log_params_sqrt*4.0))),np.minimum(np.array([20000.0,1.0,1.0]),np.exp(np.diagonal(mean_log_params+cov_log_params_sqrt*4.0))))
    
    
    np.random.seed(seed)
    
    # Preallocate arrays to store synthetic BTCs and their parameters
    btcs = np.zeros((size, len(t_)))
    params = np.zeros((size,len(mean_log_params)))
    
    for i in range(size):
        # Sample parameters within bounds
        in_range = False
        while not(in_range):
            random_log_param = sp.stats.multivariate_normal.rvs(mean = mean_log_params,cov = cov_log_params)
            random_param = np.exp(random_log_param)
            if sum(random_param>bounds[0])==len(mean_log_params) and sum(random_param<bounds[1])==len(mean_log_params):
                in_range = True
        
        # Store the valid parameter set
        params[i] = random_param
        
        # Compute the BTC using the Laplace-based solver
        btcs[i] = ADEMF_1D(t_,[1.0],1.0,random_param[0],random_param[1],[1,random_param[2]],memory_func,bound_cond)[:,-1]

    return params, btcs

def generate_dist(params):
    """
    Generates a multivariate log-normal distribution from parameter samples, excluding outliers.
    
    The function iteratively identifies and excludes outliers based on Mahalanobis distance in 
    log-parameter space. It computes the mean and covariance matrix of the log-transformed parameters, 
    then returns these along with the matrix square root of the covariance for later sampling or analysis.
    
    Parameters:
        params (np.ndarray): A 2D array of shape (n_samples, n_parameters) containing positive parameter samples.
    
    Returns:
        mean_log_params (np.ndarray): Mean of the log-transformed parameters, excluding outliers.
        cov_log_params (np.ndarray): Covariance matrix of the log-transformed parameters.
        cov_log_params_sqrt (np.ndarray): Matrix square root of the covariance matrix (via matrix square root).
    """
    is_outlier = np.array([False for i_btc in range(len(params))])
    for i in range(10):
        mean_log_params = np.mean(np.log(params)[~is_outlier], axis=0)
        cov_log_params = np.cov(np.log(params)[~is_outlier], rowvar=0)
        cov_log_params_sqrt = sp.linalg.sqrtm(cov_log_params)
        is_outlier = np.array([np.linalg.norm(np.abs(np.linalg.solve(cov_log_params_sqrt,(np.log(params)-mean_log_params)[i_btc])))>4.0 for i_btc in range(len(params))]) # 4.0 corresponds to a 0.1% probability of being outside of the range
    bounds = (np.exp(np.diagonal(mean_log_params-cov_log_params_sqrt*4.0)),np.exp(np.diagonal(mean_log_params+cov_log_params_sqrt*4.0)))
    return mean_log_params, cov_log_params, cov_log_params_sqrt   