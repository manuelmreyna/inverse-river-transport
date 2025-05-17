import scipy
import numpy as np

def TSM_moms_analyt(params,x,bound_cond):
    """
    Calculates analytical moments as derived Aghababaei, M., & Ginn, T. R. (2023). Temporal moments of one-dimensional advective-dispersive transport with exchange represented via memory function models: Application to river corridor transport. Advances in Water Resources, 172, 104383. https://doi.org/10.1016/j.advwatres.2023.104383
    The formulation for different boundary conditions is original to this work.
    params = List of independent parameters [v,Pe,beta*k_f/v*x,k_r/v*x] (all parameters but the velocity are dimensionless)
    x = Float. Sampling location.
    bound_cond = Either 'semi-infinite-conc', 'semi-infinite-mixed', or 'infinite'
                'semi-infinite-conc' is a domain x>0 with vc(x=0,t)=M_0 delta(t)
                'semi-infinite-mixed' is a domain x>0 with vc(x=0,t)- 2 D dc/dx (x=0,t) =M_0 delta(t)
                'infinite' is a domain x>0 with vc(x=0,t)-2 D dc/dx (x=0,t) =M_0 delta(t). The solution to this problem is equivalent to the solution in the infinite domain
    :returns: Array of first 4 analytical kth central moments to the power 1/k, such that they all have the same dimension [T].
    """
    if bound_cond == 'semi-infinite-conc':
        n_BC = 0
    if bound_cond == 'semi-infinite-mixed':
        n_BC = 1
    if bound_cond == 'infinite':
        n_BC = 2
    
    A_1 = lambda beta,kr: 1+beta
    A_2 = lambda beta,kr: 2*beta/kr
    A_3 = lambda beta,kr: 6*beta/kr**2
    A_4 = lambda beta,kr: 24*beta/kr**3

    a_11 = lambda v,D,beta,kr: A_1(beta,kr)/v
    mu1 = lambda x,v,D,beta,kr: a_11(v,D,beta,kr)*(x+D/v*n_BC)

    a_21 = lambda v,D,beta,kr: 2*A_1(beta,kr)**2*D/v**3+(A_2(beta,kr)+2*A_1(beta,kr)*mu1(0,v,D,beta,kr))/v
    a_22 = lambda v,D,beta,kr: a_11(v,D,beta,kr)**2
    mu2 = lambda x,v,D,beta,kr: a_21(v,D,beta,kr)*(x+D/v*n_BC)+a_22(v,D,beta,kr)*(x**2)

    a_31 = lambda v,D,beta,kr: (A_3(beta,kr)+3*A_1(beta,kr)*mu2(0,v,D,beta,kr)+3*A_2(beta,kr)*mu1(0,v,D,beta,kr))/v +6*(A_2(beta,kr)*A_1(beta,kr)*D+A_1(beta,kr)**2*D*mu1(0,v,D,beta,kr))/v**3 +12*A_1(beta,kr)**3*D**2/v**5
    a_32 = lambda v,D,beta,kr: 3*A_1(beta,kr)*(A_2(beta,kr)+A_1(beta,kr)*mu1(0,v,D,beta,kr))/v**2 +6*A_1(beta,kr)**3*D/v**4
    a_33 = lambda v,D,beta,kr: a_11(v,D,beta,kr)**3
    mu3 = lambda x,v,D,beta,kr: a_31(v,D,beta,kr)*(x+D/v*n_BC)+a_32(v,D,beta,kr)*(x**2)+a_33(v,D,beta,kr)*(x**3)

    a_41 = lambda v,D,beta,kr: (A_4(beta,kr)+4*A_1(beta,kr)*mu3(0,v,D,beta,kr)+6*A_2(beta,kr)*mu2(0,v,D,beta,kr)+4*A_3(beta,kr)*mu1(0,v,D,beta,kr))/v +(8*A_1(beta,kr)*A_3(beta,kr)*D+6*A_2(beta,kr)**2*D+24*A_1(beta,kr)*A_2(beta,kr)*D*mu1(0,v,D,beta,kr)+12*A_1(beta,kr)**2*D*mu2(0,v,D,beta,kr))/v**3 + (72*A_1(beta,kr)**2*A_2(beta,kr)*D**2+48*A_1(beta,kr)**3*D**2*mu1(0,v,D,beta,kr))/v**5 +120*A_1(beta,kr)**4*D**3/v**7
    a_42 = lambda v,D,beta,kr: ((3*A_2(beta,kr)**2+4*A_1(beta,kr)*A_3(beta,kr))+12*A_1(beta,kr)*A_2(beta,kr)*mu1(0,v,D,beta,kr)+6*A_1(beta,kr)**2*mu2(0,v,D,beta,kr))/v**2 + (36*A_1(beta,kr)**2*A_2(beta,kr)*D+24*A_1(beta,kr)**3*D*mu1(0,v,D,beta,kr))/v**4 + 60*A_1(beta,kr)**4*D**2/v**6
    a_43 = lambda v,D,beta,kr: (6*A_1(beta,kr)**2*A_2(beta,kr)+4*A_1(beta,kr)**3*mu1(0,v,D,beta,kr))/v**3 + 12*A_1(beta,kr)**4*D/v**5
    a_44 = lambda v,D,beta,kr: a_11(v,D,beta,kr)**4
    mu4 = lambda x,v,D,beta,kr: a_41(v,D,beta,kr)*(x+D/v*n_BC)+a_42(v,D,beta,kr)*(x**2)+a_43(v,D,beta,kr)*(x**3)+a_44(v,D,beta,kr)*(x**4)

    m1_analytical =lambda x,v,D,beta,kr: a_11(v,D,beta,kr)*(x+D/v*n_BC)
    m2_analytical =lambda x,v,D,beta,kr: a_21(v,D,beta,kr)*(x+D/v*n_BC)-a_11(v,D,beta,kr)**2*(2*x*D/v*n_BC+(D/v*n_BC)**2)
    m3_analytical =lambda x,v,D,beta,kr: mu3(x,v,D,beta,kr)-3*mu1(x,v,D,beta,kr)*mu2(x,v,D,beta,kr)+2*mu1(x,v,D,beta,kr)**3
    m4_analytical =lambda x,v,D,beta,kr: mu4(x,v,D,beta,kr)-4*mu1(x,v,D,beta,kr)*mu3(x,v,D,beta,kr)+6*mu2(x,v,D,beta,kr)*mu1(x,v,D,beta,kr)**2-3*mu1(x,v,D,beta,kr)**4

    mi_analytical_t_dim = [lambda x,v,D,beta,kr: m1_analytical(x,v,D,beta,kr),
                         lambda x,v,D,beta,kr: (m2_analytical(x,v,D,beta,kr))**(1/2),
                         lambda x,v,D,beta,kr: (m3_analytical(x,v,D,beta,kr))**(1/3),
                         lambda x,v,D,beta,kr: (m4_analytical(x,v,D,beta,kr))**(1/4)]
    
    return np.array([mi_analytical_t_dim[i](x,params[0],x*params[0]/params[1],params[2]/params[3],params[3]*params[0]/x) for i in range(4)])

def TSM_moms_meas(t,btc):
    """
    Computes the first 4 kth central moments to the power 1/k of the measured breakthrough curve, using the trapezoidal rule of integration.
    t = Times of measurement of concentrations
    btc = Measured concentrations (same length as t)
    :returns: Array of first 4 measured kth central moments to the power 1/k, such that they all have the same dimension [T].
    """
    mu_i = [np.trapz(btc*t**j,x=t)/np.trapz(btc,x=t) for j in range(5)]
    m_i = [mu_i[1],
           mu_i[2]-mu_i[1]**2,
           mu_i[3]-3*mu_i[1]*mu_i[2]+2*mu_i[1]**3,
           mu_i[4]-4*mu_i[1]*mu_i[3]+6*mu_i[2]*mu_i[1]**2-3*mu_i[1]**4]
    m_i_t_dim = np.array([m_i[0],m_i[1]**(1/2),np.sign(m_i[2])*np.abs(m_i[2])**(1/3),m_i[3]**(1/4)])
    return m_i_t_dim

def TSM(t,btc,x,bound_cond, params0 = None, bounds = None, v_ref = None):
    """
    Estimates parameters of the first order memory function problem (equivalent to transient storage model) by moment matching.
    t = Times of measurement of concentrations
    btc = Measured concentrations (same length as t)
    x = Float. Distance from the point of release
    bound_cond = Either 'semi-infinite-conc', 'semi-infinite-mixed', or 'infinite'
                'semi-infinite-conc' is a domain x>0 with vc(x=0,t)=M_0 delta(t)
                'semi-infinite-mixed' is a domain x>0 with vc(x=0,t)- 2 D dc/dx (x=0,t) =M_0 delta(t)
                'infinite' is a domain x>0 with vc(x=0,t)-2 D dc/dx (x=0,t) =M_0 delta(t). The solution to this problem is equivalent to the solution in the infinite domain
    params0 = Array of length 4. Initial guess of parameters.
    bounds = 2-tuple of arrays of length 4. Lower and upper bounds of parameters.
    :returns: Array of estimated independent parameters [v,Pe,beta*k_f/v*x,k_r/v*x] (all parameters but the velocity are dimensionless)
    """
    if v_ref is None:
        v_ref = x/t[np.argmax(btc)]
    
    if bounds is None:
        bounds = (np.array([v_ref*0.9,20.0,0.0,0.0]),np.array([v_ref*1.5,20000.0,np.inf,np.inf]))
    
    if params0 is None:
        params0 = np.concatenate(([v_ref],np.where((bounds[0][1:]>0.0) & (bounds[1][1:]<np.inf), np.sqrt(bounds[0][1:]*bounds[1][1:]), np.ones(3))))

    res = lambda params: TSM_moms_analyt(params,x,bound_cond)-TSM_moms_meas(t,btc)
    params_moments = scipy.optimize.least_squares(res,params0,bounds=bounds).x
    return params_moments