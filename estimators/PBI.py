import numpy as np
import scipy as sp

def test_err_rand_KL_coeffs(t, btc, x, btcs_mean, phis_interp, lambdas, v_epsilons, reg_Zs, ratio_drop = 0.2, n_repeats = 5):
    test_errors = np.zeros((len(reg_Zs),n_btcs,n_vs))
    for i_reg in range(len(reg_Zs)):
        reg_Z = reg_Zs[i_reg]
        btc = btc/(np.sum(btc)*np.sqrt(len(btc)))
        v_ests = x/t[np.argmax(btc)]*v_epsilons
        for i in range(n_vs):
            v_est = v_ests[i]
            btcs_mean_interp = np.interp(t/x*v_est, t_, btcs_mean, left=0.0, right=0.0)
            scaling_factor = 1/(np.sum(btcs_mean_interp)*np.sqrt(len(btcs_mean_interp)))
            btcs_mean_interp = btcs_mean_interp*scaling_factor
            phis_interp  = np.array([np.interp(t/x*v_est, t_, phi, left=0.0, right=0.0) for phi in phis])*scaling_factor
            index_not_drop = [np.sort(np.random.choice(np.arange(len(btc)),size = int(len(btc)*(1-ratio_drop)),replace=False)) for j in range(n_repeats)]
            index_drop = [np.delete(np.arange(len(btc)),index_not_drop[j]) for j in range(n_repeats)]
            vec_bs = [np.concatenate((btc[index_not_drop[j]]-btcs_mean_interp[index_not_drop[j]],np.zeros(len(lambdas)))) for j in range(n_repeats)]
            matrix_Cs = [np.concatenate((phis_interp[:,index_not_drop[j]].T,reg_Z*np.diag(1/np.sqrt(lambdas)))) for j in range(n_repeats)]
            Z_js =  [sp.linalg.lstsq(matrix_Cs[j],vec_bs[j])[0] for j in range(n_repeats)]
            rec_btcs = [btcs_mean_interp[index_drop[j]]+phis_interp[:,index_drop[j]].T@Z_js[j] for j in range(n_repeats)]
            test_error = np.sum([np.sum((btc[index_drop[j]]-rec_btcs[j])**2) for j in range(n_repeats)])
            test_errors[i_reg,i_btc,i] = test_error
    return np.exp(np.mean(np.mean(np.log(test_errors),axis = -1),axis =-1))

def KL_coeffs(t, btc, x, btcs_mean, phis_interp, lambdas, v_epsilons,reg_Z_opt=10**-(4.5)):
    Zs_meas_v = np.zeros((n_vs,n_lmbds))
    btc = btc/(np.sum(btc)*np.sqrt(len(btc)))
    v_ests = x/t[np.argmax(btc)]*epsilons
    for i in range(n_vs):
        v_est = v_ests[i]
        btcs_mean_interp = np.interp(t/x*v_est, t_, btcs_mean, left=0.0, right=0.0)
        scaling_factor = 1/(np.sum(btcs_mean_interp)*np.sqrt(len(btcs_mean_interp)))
        btcs_mean_interp = btcs_mean_interp*scaling_factor
        phis_interp  = np.array([np.interp(t/x*v_est, t_, phi, left=0.0, right=0.0) for phi in phis[:n_lmbds]])*scaling_factor        
        vec_b = np.concatenate((btc-btcs_mean_interp,np.zeros(n_lmbds)))
        matrix_C = np.concatenate((phis_interp.T,reg_Z_opt*np.diag(1/np.sqrt(lambdas[:n_lmbds]))))
        Zs_meas_v[i_btc][i] = sp.linalg.lstsq(matrix_C,vec_b)[0]
    return Zs_meas_v
    
def NNI(t, btc, x,Zs_meas_v,Zs, v_epsilons):
    v_ests = x/t[np.argmax(btc)]*v_epsilons
    diff_Zs_v = np.sum((Zs_v[:,None,:]-Zs[None,:,:])**2,axis = -1)+reg_v*(v_epsilons[:,None]-1)**2
    v_NNI = v_ests[np.argmin(np.min(diff_Zs_v, axis = -1),axis = -1)]
    params_NNI = np.concatenate(([v_NNI],params_rand[np.argmin(np.min(diff_Zs_v,axis = -2),axis = -1)]))
    return params_NNI