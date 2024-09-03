import numpy as np
from baseline.utils import get_laplacian, get_laplacian_2d, get_laplacian_2d_vec, get_laplacian_2d_vec_secure, get_laplacian_vec, get_laplacian_vec_secure, polar_to_spherical, polar_to_spherical_vec, get_laplace_secure

def GP_loc(x, eps):
    if not(isinstance(x,(list, tuple, np.ndarray))):
        Z = np.random.laplace(0,1)
        x_tilde = x+Z/eps
        return x_tilde
    d = len(x)
    if d == 2:
        (r, theta) = get_laplacian_2d(eps)
        Z = np.array([r*np.cos(theta),r*np.sin(theta)])
        x_tilde = x + Z
    else:
        (r, thetas) = get_laplacian(eps, d)
        Z = np.array(polar_to_spherical(r,thetas,d))
        x_tilde = x + Z
    return x_tilde

def GP_loc_vec(x, eps, mreps):
    if not(isinstance(x,(list, tuple, np.ndarray))):
        Z = np.random.laplace(0,1,mreps)
        x_tilde = x+1.0/eps*Z
        return x_tilde
    d = len(x)
    if d == 2:
        (r, theta) = get_laplacian_2d_vec(eps, mreps)
        Z = np.array([r*np.cos(theta),r*np.sin(theta)])
        x_tilde = x+np.transpose(Z)
    else:
        (r, thetas) = get_laplacian_vec(eps, d, mreps)
        Z = np.array(polar_to_spherical_vec(r,thetas,d))
        x_tilde = x+np.transpose(Z)
    return x_tilde        

def GP_loc_vec_secure(x, eps, mreps):
    if not(isinstance(x,(list, tuple, np.ndarray))):
        Z = get_laplace_secure(size=mreps)
        x_tilde = x+1.0/eps*Z
        return x_tilde
    d = len(x)
    if d == 2:
        (r, theta) = get_laplacian_2d_vec_secure(eps, mreps)
        Z = np.array([r*np.cos(theta),r*np.sin(theta)])
        x_tilde = x+np.transpose(Z)
    else:
        (r, thetas) = get_laplacian_vec_secure(eps, d, mreps)
        Z = np.array(polar_to_spherical_vec(r,thetas,d))
        x_tilde = x+np.transpose(Z)
    return x_tilde   

def GP_func(x, func, eps, sens):#, d=1):
    y = func(x)
    Z = np.random.laplace(0,1)#,size=d)
    y = y + Z*sens/eps
    return y

def GP_func_vec(x, func, eps, sens, mreps):
    y = func(x)
    Z = np.random.laplace(0,1,size=mreps)
    y = y + Z*sens/eps
    return y

def GP_func_vec_secure(x, func, eps, sens, mreps):
    y = func(x)
    Z = get_laplace_secure(size=mreps)
    y = y + Z*sens/eps
    return y

def GP_post_func(arr_x, eps, func_priv, sens_priv, func_post):
    y_post = [func_post(GP_func(x, func_priv, eps, sens_priv)) for x in arr_x]
    return y_post

def GP_post_func_vec(arr_x, eps, func_priv, sens_priv, func_post, mreps):
    func_post_vec = lambda arr_y: [func_post(y) for y in arr_y]
    y_post = [func_post_vec(GP_func_vec(x, func_priv, eps, sens_priv, mreps)) for x in arr_x]
    return y_post

def GP_post_func_vec_secure(arr_x, eps, func_priv, sens_priv, func_post, mreps):
    func_post_vec = lambda arr_y: [func_post(y) for y in arr_y]
    y_post = [func_post_vec(GP_func_vec_secure(x, func_priv, eps, sens_priv, mreps)) for x in arr_x]
    return y_post