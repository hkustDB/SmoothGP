import numpy as np
from scipy.stats import laplace
from random import SystemRandom

def ldp_lap_vec(arr_x, func, epsL, mreps, sens):
    y_tilde = [func(x)+np.random.laplace(0,1,size=mreps)*sens/epsL for x in arr_x]
    return y_tilde

def get_laplace_secure(size=1):
    rand = SystemRandom()
    rv = laplace()
    Zs = []
    for i in range(size):
        U = rand.random()
        Z = rv.ppf(U)
        Zs.append(Z)
    return np.array(Zs)

def ldp_lap_vec_secure(arr_x, func, epsL, mreps, sens):
    Z = get_laplace_secure(size=mreps)
    y_tilde = [func(x)+Z*sens/epsL for x in arr_x]
    return y_tilde


