
import numpy as np
from utils.invlap import invlap

def ADEMF_1D(ts, xs, v, Pe, beta, g_params, memory_func, bound_cond):
    """
    Solves the memory function advection-dispersion-immobile-exchange model with constant coefficients
    ts = Array. Times after release
    xs = Array. Distance from the point of release
    v = Float. Advection velocity
    Pe = Float. Peclet number
    beta = Float. Ratio between the effective immobile phase area and the mobile phase area
    g_params = List of the parameters of the corresponding memory function
                'first order' takes [k_f,k_r]
                'power law' takes [alpha,1-gamma]   
    memory_func = Either 'first order' or 'power law', others can be added editing c_lap_space
    bound_cond = Either 'semi-infinite-conc', 'semi-infinite-mixed', or 'infinite'
                'semi-infinite-conc' is a domain x>0 with vc(x=0,t)=M_0 delta(t)
                'semi-infinite-mixed' is a domain x>0 with vc(x=0,t)- 2 D dc/dx (x=0,t) =M_0 delta(t)
                'infinite' is a domain x>0 with vc(x=0,t)-2 D dc/dx (x=0,t) =M_0 delta(t). The solution to this problem is equivalent to the solution in the infinite domain
    """
    def sol_lap_space(s, x):
        """
        Auxilitary function for c_lap_solver
        """
        if memory_func == 'first order':
            G = lambda s,g_params: g_params[0]/(g_params[1] + s)
        if memory_func == 'power law':
            G = lambda s,g_params: g_params[0]*s**(-g_params[1])
        b = s + beta * s * G(s,g_params)
        if bound_cond == 'semi-infinite-conc':
            c = np.exp(x*(Pe- np.sqrt(Pe*(4*b+Pe)))/2)
        if bound_cond == 'semi-infinite-mixed':
            c = 1/(1/2+np.sqrt(4*b+Pe)/(2*np.sqrt(Pe)))*np.exp(x*(Pe- np.sqrt(Pe*(4*b+Pe)))/2)
        if bound_cond == 'infinite':
            c = np.sqrt(Pe/(4*b+Pe)) * np.exp((Pe*x- np.sqrt(Pe*x**2*(4*b+Pe)))/2)
        return c
    cis = np.array([invlap(sol_lap_space, ts*v/xs[-1], [x/xs[-1]]) for x in xs])*v/xs[-1]
    cis = np.nan_to_num(np.array(cis).T, 0.0)
    return cis