import numpy as np
import scipy as sp

def test_err_rand_KL_coeffs(ts_m, btcs_m, xs_m, t_, btcs_mean, phis, lambdas, v_epsilons, reg_Zs, ratio_drop = 0.2, n_repeats = 5):
    """
    Evaluate test error for a range of regularization values using random subsampling.

    Parameters:
        ts_m (list of np.ndarray): List of time arrays for each BTC.
        btcs_m (list of np.ndarray): List of measured BTCs.
        xs_m (list of float): List of distances for each BTC.
        btcs_mean (np.ndarray): Mean BTC.
        phis_interp (list of np.ndarray): Interpolated eigenfunctions.
        lambdas (np.ndarray): Eigenvalues corresponding to KL expansion.
        v_epsilons (np.ndarray): Perturbation factors for velocity estimation.
        reg_Zs (list of float): List of regularization parameters to test.
        ratio_drop (float, optional): Fraction of data to drop during testing. Default is 0.2.
        n_repeats (int, optional): Number of repetitions for subsampling. Default is 5.

    Returns:
        np.ndarray: Mean log test error across all BTCs and regularizations.
    """    
    n_vs = len(v_epsilons)
    n_btcs = len(ts_m)
    test_errors = np.zeros((len(reg_Zs),n_btcs,n_vs))
    for i_btc in range(n_btcs):
        t, btc, x = ts_m[i_btc], btcs_m[i_btc], xs_m[i_btc]
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
    reg_Z_opt = np.exp(np.mean(np.mean(np.log(test_errors),axis = -1),axis =-1))
    return reg_Z_opt

def KL_coeffs(t, btc, x, t_, btcs_mean, phis, lambdas, v_epsilons,reg_Z_opt=10**-(4.5)):
    """
    Estimate KL coefficients (Zs) from a given BTC by projecting onto interpolated eigenfunctions.

    Parameters:
        t (np.ndarray): Time vector of the BTC.
        btc (np.ndarray): Measured breakthrough curve.
        x (float): Distance from injection to measurement.
        btcs_mean (np.ndarray): Mean BTC.
        phis_interp (list of np.ndarray): Interpolated eigenfunctions.
        lambdas (np.ndarray): Eigenvalues.
        v_epsilons (np.ndarray): Perturbation factors for velocity.
        reg_Z_opt (float, optional): Regularization coefficient. Default is 10**-4.5.

    Returns:
        np.ndarray: Array of estimated KL coefficients for each velocity perturbation.
    """
    n_lmbds = phis.shape[0]
    n_vs = len(v_epsilons)
    Zs_meas_v = np.zeros((n_vs,n_lmbds))
    btc = btc/(np.sum(btc)*np.sqrt(len(btc)))
    v_ests = x/t[np.argmax(btc)]*v_epsilons
    for i in range(n_vs):
        v_est = v_ests[i]
        btcs_mean_interp = np.interp(t/x*v_est, t_, btcs_mean, left=0.0, right=0.0)
        scaling_factor = 1/(np.sum(btcs_mean_interp)*np.sqrt(len(btcs_mean_interp)))
        btcs_mean_interp = btcs_mean_interp*scaling_factor
        phis_interp  = np.array([np.interp(t/x*v_est, t_, phi, left=0.0, right=0.0) for phi in phis[:n_lmbds]])*scaling_factor        
        vec_b = np.concatenate((btc-btcs_mean_interp,np.zeros(n_lmbds)))
        matrix_C = np.concatenate((phis_interp.T,reg_Z_opt*np.diag(1/np.sqrt(lambdas[:n_lmbds]))))
        Zs_meas_v[i] = sp.linalg.lstsq(matrix_C,vec_b)[0]
    return Zs_meas_v, v_ests
    
def estimate_params_NNI(params_rand, Zs, v_ests, Zs_v, reg_v = 0, v_ref = None):
    """
    Estimate transport parameters using Nearest Neighbor Interpolation (NNI) in KL coefficient space.

    Parameters:
        params_rand (np.ndarray): Randomly sampled parameters.
        Zs (np.ndarray): KL coefficients for the random samples.
        v_ests (np.ndarray): Velocity estimates.
        Zs_v (np.ndarray): KL coefficients estimated from BTC.
        reg_v (float, optional): Regularization on velocity deviation. Default is 0.
        v_epsilons (np.ndarray, optional): Velocity perturbation factors.

    Returns:
        np.ndarray: Estimated parameters using NNI.
    """
    if v_ref is None:
        v_ref = np.mean(v_ests)
    diff_Zs_v = np.sum((Zs_v[:,None,:]-Zs[None,:,:])**2,axis = -1)+reg_v*((v_ests/v_ref)[:,None]-1)**2
    v_NNI = v_ests[np.argmin(np.min(diff_Zs_v, axis = -1),axis = -1)]
    params_NNI = np.concatenate(([v_NNI],params_rand[np.argmin(np.min(diff_Zs_v,axis = -2),axis = -1)]))
    return params_NNI

def proj(v,u):
    """
    Project vector v onto vector u.

    Parameters:
        v (np.ndarray): Vector to project.
        u (np.ndarray): Vector to project onto.

    Returns:
        np.ndarray: Projection of v onto u.
    """
    return (v@u)/(u@u)*u

def gs(vectors):
    """
    Perform Gram-Schmidt orthonormalization on a list of vectors.

    Parameters:
        vectors (list of np.ndarray): Input vectors.

    Returns:
        list of np.ndarray: Orthonormal basis vectors.
    """
    basis = []
    for vec in vectors:
        u = vec - sum([proj(vec,e) for e in basis])
        basis.append(u/np.linalg.norm(u))
    return basis

def get_projecting_basis(projecting_points):
    """
    Compute the projecting basis using Gram-Schmidt from the projecting points.

    Parameters:
        projecting_points (np.ndarray): Points used for projection.

    Returns:
        list: Orthonormal basis for projection.
    """
    return gs(projecting_points[1:]-projecting_points[0])

def get_projected_points(projecting_points,projecting_basis,Z):
    """
    Project a point Z onto the affine space defined by projecting_points and basis.

    Parameters:
        projecting_points (np.ndarray): Points defining the space.
        projecting_basis (list): Basis for the projection.
        Z (np.ndarray): Query point.

    Returns:
        np.ndarray: Projected point.
    """
    return projecting_points[0]+sum([proj(Z-projecting_points[0],e) for e in projecting_basis])

def get_vertices_basis(projecting_points,projecting_basis):
    """
    Express the vertices of a simplex in the basis coordinates.

    Parameters:
        projecting_points (np.ndarray): Vertices of the simplex.
        projecting_basis (list): Basis vectors.

    Returns:
        np.ndarray: Coordinates of the vertices in the basis.
    """
    return np.array([[(projecting_points[i]-projecting_points[0])@e for e in projecting_basis]  for i in range(len(projecting_points))])

def get_projected_points_basis(projected_points,projecting_points,projecting_basis):
    """
    Express the projected point in the simplex basis coordinates.

    Parameters:
        projected_points (np.ndarray): Projected point.
        projecting_points (np.ndarray): Vertices of the simplex.
        projecting_basis (list): Basis vectors.

    Returns:
        np.ndarray: Coordinates of the projected point in the basis.
    """
    return np.array([(projected_points-projecting_points[0])@e for e in projecting_basis])

def drop_points_in_simplex(projecting_indices,projecting_points,projecting_basis,projected_points,vertices_basis,projected_points_basis,Z,Zs):
    """
    Adjust the simplex to ensure that the projected point lies inside it (all barycentric coordinates are positive) by iteratively dropping vertices.

    Parameters:
        projecting_indices (list): Indices of the projecting points.
        projecting_points (np.ndarray): Coordinates of projecting points.
        projecting_basis (list): Basis vectors.
        projected_points (np.ndarray): Projected point.
        vertices_basis (np.ndarray): Vertices in basis coordinates.
        projected_points_basis (np.ndarray): Projected point in basis coordinates.
        Z (np.ndarray): Query point.

    Returns:
        tuple: Updated barycentric coordinates, projecting indices, projecting points, basis, etc.
    """
    if len(vertices_basis)>1:
        T = (vertices_basis[:-1]-vertices_basis[-1]).T
        bar_coord = np.linalg.solve(T,projected_points_basis-vertices_basis[-1])
        bar_coord = np.concatenate((bar_coord,[1-np.sum(bar_coord)]))
        while np.any(bar_coord<0) and len(projecting_indices)>1:
            projecting_indices = projecting_indices[:-1]
            projecting_points = Zs[projecting_indices]
            projecting_basis = gs(projecting_points[1:]-projecting_points[0])
            projected_points = projecting_points[0]+sum([proj(Z-projecting_points[0],e) for e in projecting_basis])
            if len(vertices_basis)>=1:
                vertices_basis = np.array([[(projecting_points[i]-projecting_points[0])@e for e in projecting_basis] for i in range(len(projecting_points))])
                projected_points_basis = np.array([(projected_points-projecting_points[0])@e for e in projecting_basis])
                T = (vertices_basis[:-1]-vertices_basis[-1]).T
                bar_coord = np.linalg.solve(T,projected_points_basis-vertices_basis[-1])
                bar_coord = np.concatenate((bar_coord,[1-np.sum(bar_coord)]))
            else:
                bar_coord = np.array([1])
                break
    else:
        bar_coord = np.array([1])
    return bar_coord, projecting_indices,projecting_points,projecting_basis,projected_points,vertices_basis,projected_points_basis

def estimate_params_PBI(params_rand, Zs, v_ests, Zs_v, dim, reg_v = 0, v_ref = None):
    """
    Estimate parameters using Projected Barycentric Interpolation (PBI) from KL coefficients.

    Parameters:
        params_rand (np.ndarray): Random parameter samples.
        Zs (np.ndarray): KL coefficients of samples.
        v_ests (np.ndarray): Velocity estimates.
        Zs_v (np.ndarray): KL coefficients of measured BTC.
        dim (int): Target dimension for projection.
        reg_v (float, optional): Velocity regularization. Default is 0.
        v_epsilons (np.ndarray, optional): Velocity perturbation factors.

    Returns:
        np.ndarray: Estimated parameters using PBI.
    """
    if v_ref is None:
        v_ref = np.mean(v_ests)
    sorted_args = np.argsort(np.sum((Zs_v[:,None,:]-Zs[None,:,:])**2,axis = -1),axis = -1)
    
    projecting_indices = [sorted_args[j][:dim+1] for j in range(len(v_ests))]
    projecting_points = [Zs[projecting_indices[j]]  for j in range(len(v_ests))]
    projecting_basis = [get_projecting_basis(projecting_points[j]) for j in range(len(v_ests))]
    projected_points = np.array([get_projected_points(projecting_points[j],projecting_basis[j],Zs_v[j]) for j in range(len(v_ests))])
    vertices_basis = [get_vertices_basis(projecting_points[j],projecting_basis[j]) for j in range(len(v_ests))]
    projected_points_basis= [get_projected_points_basis(projected_points[j],projecting_points[j],projecting_basis[j]) for j in range(len(v_ests))]
    
    bar_coords = []
    for j in range(len(v_ests)):
        bar_coord,projecting_indices[j],projecting_points[j],projecting_basis[j],projected_points[j],vertices_basis[j],projected_points_basis[j] = drop_points_in_simplex(projecting_indices[j],projecting_points[j],projecting_basis[j],projected_points[j],vertices_basis[j],projected_points_basis[j],Zs_v[j],Zs)
        bar_coords.append(bar_coord)

    arg_PBI = np.argmin(np.linalg.norm(projected_points-Zs_v, axis = -1)+np.sqrt(reg_v*(v_ests/v_ref-1)**2))
    Z_PBI = Zs_v[arg_PBI]
    v_PBI = v_ests[arg_PBI]
    bar_coords_best = bar_coords[arg_PBI]
    projecting_indices_best = projecting_indices[arg_PBI]

    params_PBI = np.concatenate(([v_PBI],np.exp(np.average(np.log(params_rand)[projecting_indices_best],weights = bar_coords_best, axis = 0))))
    
    return params_PBI