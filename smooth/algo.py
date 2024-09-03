import numpy as np
from smooth.utils import get_cauchy4_rv_secure, get_cauchy6_rv, get_cauchy6_rv_secure, get_kde_smooth_analytic, get_cauchy4_rv, func_theta, get_threshold_smooth_analytic, func_twtau2, get_twthreshold_smooth_analytic2, get_standardt_secure

def smooth_mechanism_c(x, eps, func, sens_func, nu=3, pg=0.2):
    gamma = pg*eps/nu
    nu1 = nu**(nu/(nu+1.0))
    eta = (1.0-pg)*eps/(nu1)
    if nu==3:
        Z = get_cauchy4_rv()
    else:
        Z = get_cauchy6_rv()
    sens = sens_func(x, gamma)
    y = func(x)+sens/eta*Z
    return y


def smooth_mechanism_c_secure(x, eps, func, sens_func, nu=3, pg=0.2):
    gamma = pg*eps/nu
    nu1 = nu**(nu/(nu+1.0))
    eta = (1.0-pg)*eps/(nu1)
    if nu==3:
        Z = get_cauchy4_rv_secure()
    else:
        Z = get_cauchy6_rv_secure()
    sens = sens_func(x, gamma)
    y = func(x)+sens/eta*Z
    return y


def smooth_mechansim_cvec(x, eps, func, sens_func, mreps, nu=3, pg=0.2):
    gamma = pg*eps/nu
    nu1 = nu**(nu/(nu+1.0))
    eta = (1.0-pg)*eps/(nu1)
    sens = sens_func(x, gamma)
    if nu==3:
        Z = get_cauchy4_rv(mreps)
    else:
        Z = get_cauchy6_rv(mreps)
    y = func(x)+sens/eta*Z
    return y


def smooth_mechansim_cvec_secure(x, eps, func, sens_func, mreps, nu=3, pg=0.2):
    gamma = pg*eps/nu
    nu1 = nu**(nu/(nu+1.0))
    eta = (1.0-pg)*eps/(nu1)
    sens = sens_func(x, gamma)
    if nu==3:
        Z = get_cauchy4_rv_secure(mreps)
    else:
        Z = get_cauchy6_rv_secure(mreps)
    y = func(x)+sens/eta*Z
    return y


def smooth_mechanism_t(x, eps, func, sens_func, nu=3, pg=1.0/3.0):
    gamma = pg*eps/nu
    nu1 = (nu+1.)/2./np.sqrt(nu)
    eta = (1.0-pg)*eps/(nu1)

    Z = np.random.standard_t(nu)
    sens = sens_func(x, gamma)
    y = func(x)+sens/eta*Z
    return y

def smooth_mechanism_t_secure(x, eps, func, sens_func, nu=3, pg=1.0/3.0):
    gamma = pg*eps/nu
    nu1 = (nu+1.)/2./np.sqrt(nu)
    eta = (1.0-pg)*eps/(nu1)
    Z = get_standardt_secure(nu)[0]
    sens = sens_func(x, gamma)
    y = func(x)+sens/eta*Z
    return y

def smooth_mechanism_tvec(x, eps, func, sens_func, mreps, nu=3, pg=1.0/3.0):
    gamma = pg*eps/nu
    nu1 = (nu+1.)/2./np.sqrt(nu)
    eta = (1.0-pg)*eps/(nu1)
    sens = sens_func(x, gamma)
    Z = np.random.standard_t(nu,size=mreps)
    y = func(x)+sens/eta*Z
    return y

def smooth_mechanism_tvec_secure(x, eps, func, sens_func, mreps, nu=3, pg=1.0/3.0):
    gamma = pg*eps/nu
    nu1 = (nu+1.)/2./np.sqrt(nu)
    eta = (1.0-pg)*eps/(nu1)
    sens = sens_func(x, gamma)
    Z = get_standardt_secure(nu,size=mreps)
    y = func(x)+sens/eta*Z
    return y

def kde_smooth_vec(arr_x, eps, func, t, h, mreps, smooth_mech_vec=smooth_mechanism_tvec, nu=3, pg=1.0/3.0):
    sens_gk= lambda z, gamma: get_kde_smooth_analytic(z-t, gamma, h)

    arr_y = [smooth_mech_vec(x, eps, func, sens_func=sens_gk, mreps=mreps, nu=nu, pg=pg) for x in arr_x]

    return arr_y


def thres_smooth_vec(arr_x, eps, thres, theta, mreps, smooth_mech_vec=smooth_mechanism_tvec, nu=3, pg=1.0/3.0):
    func_thres = lambda x: func_theta(x, thres, theta)
    sens_thres = lambda x, gamma: get_threshold_smooth_analytic(x-thres, gamma, theta)

    arr_y = [smooth_mech_vec(x, eps, func_thres, sens_func=sens_thres, mreps=mreps, nu=nu, pg=pg) for x in arr_x]

    return arr_y


def twthres_smooth_vec(arr_x, eps, T1, T2, tau, mreps, smooth_mech_vec=smooth_mechanism_tvec, nu=3, pg=1.0/3.0):
    func_thres = lambda x: func_twtau2(x, T1, T2, tau)
    sens_thres = lambda x, gamma: get_twthreshold_smooth_analytic2(x, gamma, T1, T2, tau)

    arr_y = [smooth_mech_vec(x, eps, func_thres, sens_func=sens_thres, mreps=mreps, nu=nu, pg=pg) for x in arr_x]

    return arr_y


