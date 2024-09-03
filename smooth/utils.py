import numpy as np
from scipy.optimize import root_scalar, fsolve
from scipy.special import lambertw
from scipy.stats import t
from random import SystemRandom

def get_cauchy4_cdf_inv(p):
    ff = lambda a: np.sqrt(2.0)*a+1.0
    H = lambda a: np.log((a*a+ff(a))/(a*a+ff(-a)))+2.0*(np.arctan(ff(a))-np.arctan(ff(-a)))
    F = lambda a: 0.5 + np.sign(a)*H(abs(a))/4.0/np.pi
    func_root = lambda a: F(a)-p
    if p < 0.5:
        r = 0
        l = -4.0
        while func_root(l) > 0:
            l = l*1.5
        root_a = root_scalar(func_root, bracket=[l,r])
    else:
        l = 0
        r = 4.0
        while func_root(r) < 0:
            r = r*1.5
        root_a = root_scalar(func_root, bracket=[l,r])
    return root_a.root

def get_cauchy4_rv(m=1):
    if m==1:
        U = np.random.uniform()
        Z = get_cauchy4_cdf_inv(U)
        return Z
    else:
        U = np.random.uniform(size=m)
        Z = np.array([get_cauchy4_cdf_inv(u) for u in U])
        return Z

def get_cauchy6_cdf_inv(p):
    fl = lambda a: np.sqrt(3.0)*a+1.0
    ft = lambda a: np.sqrt(3.0)+2.0*a
    H = lambda a: np.sqrt(3.0)*np.log((a*a+fl(a))/(a*a+fl(-a)))+2.0*(np.arctan(ft(a))-np.arctan(ft(-a)))+4.0*np.arctan(a)
    F = lambda a: 0.5 + np.sign(a)*H(abs(a))/8.0/np.pi
    func_root = lambda a: F(a)-p
    if p < 0.5:
        r = 0
        l = -4.0
        while func_root(l) > 0:
            l = l*1.5
        root_a = root_scalar(func_root, bracket=[l,r])
    else:
        l = 0
        r = 4.0
        while func_root(r) < 0:
            r = r*1.5
        root_a = root_scalar(func_root, bracket=[l,r])
    return root_a.root

def get_cauchy6_rv(m=1):
    if m==1:
        U = np.random.uniform()
        Z = get_cauchy6_cdf_inv(U)
        return Z
    else:
        U = np.random.uniform(size=m)
        Z = np.array([get_cauchy6_cdf_inv(u) for u in U])
        return Z


def func_theta(x, thres, theta):
    xd = x - thres
    if xd < -0.5*theta:
        return 0.
    elif xd > 0.5*theta:
        return 1.
    else:
        return xd/theta + 0.5

def get_threshold_smooth_analytic(x, gamma, theta):
    sens = 0.0
    xd  = abs(x) #assume threshold has already been subtracted
    if xd <= 0.5*theta:
        sens = 1./theta
    else:
        sens = max(1./(xd+0.5*theta),np.exp(-gamma*(xd-0.5*theta))/theta)
    return sens   

def get_twoway_region(x, c0, T1, T2, R1, R2, tau):
    len_xc = np.linalg.norm(x-c0)
    if x[0] > T1 and x[1] > T2:
        if x[0] > T1+R1 and x[1] > T2+R1:
            return 'S13'
        elif x[0] > T1+R1:
            return 'S12'
        elif x[1] > T2+R1:
            return 'S11'
        elif len_xc < R1:
            return 'S1a'
        else:
            return 'Sta'
    elif x[0] <= T1+R1 and x[1] <= T2+R1:
        if len_xc <= R2 and len_xc >= R1:
            return 'Sta'
        else:
            return 'S0a'
    elif x[1] > T2+R1:
        if x[0] < T1-tau:
            return 'S01'
        else:
            return 'St1'
    elif x[0] > T1+R1:
        if x[1] < T2-tau:
            return 'S02'
        else:
            return 'St2'
    else:
        return ''


def get_twoway_region2(x, c0, T1, T2, R1, R2, tau):
    len_xc = np.linalg.norm(x-c0)
    if x[0] > T1+0.5*tau and x[1] > T2+0.5*tau:
        if x[0] > T1+R1+0.5*tau and x[1] > T2+R1+0.5*tau:
            return 'S13'
        elif x[0] > T1+R1+0.5*tau:
            return 'S12'
        elif x[1] > T2+R1+0.5*tau:
            return 'S11'
        elif len_xc < R1:
            return 'S1a'
        else:
            return 'Sta'
    elif x[0] <= T1+R1+0.5*tau and x[1] <= T2+R1+0.5*tau:
        if len_xc <= R2 and len_xc >= R1:
            return 'Sta'
        else:
            return 'S0a'
    elif x[1] > T2+R1+0.5*tau:
        if x[0] < T1-0.5*tau:
            return 'S01'
        else:
            return 'St1'
    elif x[0] > T1+R1+0.5*tau:
        if x[1] < T2-0.5*tau:
            return 'S02'
        else:
            return 'St2'
    else:
        return ''
    

def func_twtau2(x, T1, T2, tau):
    R1 = tau/np.sqrt(2.0)
    R2 = R1+tau
    c0 = np.array([T1+R1+0.5*tau, T2+R1+0.5*tau])
    sreg = get_twoway_region2(x, c0, T1, T2, R1, R2, tau)
    y = 0.
    if sreg[:-1] == 'S0':
        y = 0.
    elif sreg[:-1] == 'S1':
        y = 1.
    elif sreg == 'Sta':
        len_xc = np.linalg.norm(x-c0)
        y = (R2-len_xc)/tau
    elif sreg == 'St1':
        y = (x[0]-(T1-0.5*tau))/tau
    elif sreg == 'St2':
        y = (x[1]-(T2-0.5*tau))/tau
    return y

def get_twthreshold_smooth_analytic2(x, gamma, T1, T2, tau):
    R1 = tau/np.sqrt(2.0)
    R2 = R1+tau
    c0 = np.array([T1+R1+0.5*tau, T2+R1+0.5*tau])
    sreg = get_twoway_region2(x, c0, T1, T2, R1, R2, tau)
    assert(not(sreg==''))
    sens = 0.0
    match sreg:
        case 'S01':
            sens = max(1.0/(T1+0.5*tau-x[0]),np.exp(-gamma*(T1-0.5*tau-x[0]))/tau)
        case 'S02':
            sens = max(1.0/(T2+0.5*tau-x[1]),np.exp(-gamma*(T2-0.5*tau-x[1]))/tau)
        case 'S0a':
            len_xc = np.linalg.norm(x-c0)
            sens = max(1.0/(len_xc-R1),np.exp(-gamma*(len_xc-R2))/tau)
        case 'S11':
            sens = max(1.0/(x[0]-(T1-0.5*tau)),np.exp(-gamma*(x[0]-(T1+0.5*tau)))/tau)
        case 'S12':
            sens = max(1.0/(x[1]-(T2-0.5*tau)),np.exp(-gamma*(x[1]-(T2+0.5*tau)))/tau)
        case 'S13':
            sens = max(max(1.0/(x[0]-(T1-0.5*tau)),np.exp(-gamma*(x[0]-(T1+0.5*tau)))/tau),max(1.0/(x[1]-(T2-0.5*tau)),np.exp(-gamma*(x[1]-(T2+0.5*tau)))/tau))
        case 'S1a':
            len_xc = np.linalg.norm(x-c0)
            sens = max(1.0/(R2-len_xc),np.exp(-gamma*(R1-len_xc))/tau)
        case 'St1':
            sens = 1.0/tau
        case 'St2':
            sens = 1.0/tau
        case 'Sta':
            sens = 1.0/tau
        case _:
            print(sreg+' not implemented')
            raise
    return sens



def get_root_safe(func_root,l,r):
    root_z = r
    try:
        root_z = root_scalar(func_root,[l,r]).root
    except:
        try:
            root_z = fsolve(func_root,r)[0]
        except:
            root_z = fsolve(func_root,l)[0]
    return root_z

# computes root of exp(-(w-a)^2/2)-(1+a*w)*exp(-w^2/2)
def get_root_kde_phi0(w,c=1.0):
    func_root = lambda a: np.exp(-0.5*(w-a)*(w-a)*c*c)-(1+a*w*c*c)*np.exp(-0.5*w*w*c*c)
    u = w*c
    y = -u*u*np.exp(-u*u)
    if (y < -np.exp(-1.0) or abs(y)<1e-12):
        return -1
    a0 = lambertw(y).real
    a1 = lambertw(y,k=-1).real
    y0 = np.sqrt(abs(a0/u/u))
    y1 = np.sqrt(abs(a1/u/u))
    if (abs(1.0-y0)<1e-12):
        yy = y1
    else:
        assert(abs(1.0-y1)<1e-12)
        yy = y0
    if yy < 1.0:
        b1 = 1-yy
    else:
        b1 = -(yy-1)
    if 0 < b1 and b1 < 1:
        l = b1*w
        r = w
        if (func_root(l)*func_root(r) < 0):
            root_z = root_scalar(func_root,bracket=[l,r])
        else:
            return -1
    else:
        l = b1*w*2.0
        r = b1*w
        assert(func_root(r)<0)
        while(func_root(l)<0):
            l =l*1.5
        root_z = root_scalar(func_root,bracket=[l,r])
    return w - root_z.root

def get_root_kde_phi1(c, z=1.0):
    func_root = lambda a: np.exp(-0.5*z*z*c*c)-(1+c*c*a*(z+a))*np.exp(-0.5*(z+a)*(z+a)*c*c)
    if c < 1.0:
        a0 = 1.0-c
    else:
        a0 = -(c-1.0)
    if a0 > 0:
        l = a0/c
        r = 2.0*l
        while(func_root(r) < 0):
            r = r*1.5
        if (np.sign(l)==np.sign(r)):
            while(func_root(l)>1e-16):
                l = 0.1*l
        root_z = get_root_safe(func_root,l,r)
    else:
        r = a0/c
        l = -1
        assert(func_root(l) < 0)
        root_z = get_root_safe(func_root,l,r)
    return root_z + z



def get_kde_smooth_c1(c, gamma, h):
    func_phi1 = lambda g: h*(g+np.sqrt(g*g+4.0/h/h))/2.0/c

    func_obj1 = lambda w, z: c/h*w*np.exp(-0.5*w*w*c*c)*np.exp(-gamma*abs(z-1)*c*h)
    func_obj2 = lambda w: abs((np.exp(-0.5*w*w*c*c)-np.exp(-0.5*c*c))/(w-1))/c/h
    func_obj3 = lambda z: c/h*z*np.exp(-0.5*z*z*c*c)*np.exp(-gamma*abs(z-1)*c*h)
    cand4 = c/h*np.exp(-0.5*c*c)

    S1 = []
    w11 = func_phi1(-gamma)
    z11 = get_root_kde_phi0(w11,c)
    if (z11 > 1.0):
        S1.append((w11,z11))
    w12 = func_phi1(gamma)
    z12 = get_root_kde_phi0(w12,c)
    if (z12 > 0 and z12 < 1.0):
        S1.append((w12,z12))

    S2 = []
    w21 = get_root_kde_phi1(c)
    if (w21 > 0):
        S2.append(w21)
    S2.append(0)

    S3 = []
    z31 = func_phi1(-gamma)
    if (z31 > 1.0):
        S3.append(z31)
    z32 = func_phi1(gamma)
    if (z32 > 0 and z32 < 1.0):
        S3.append(z32)

    cand1 = 0
    cand2 = 0
    cand3 = 0
    if len(S1) > 0:
        cand1 = np.max([func_obj1(tup[0],tup[1]) for tup in S1])
    if len(S2) > 0:
        cand2 = np.max([func_obj2(tup) for tup in S2])
    if len(S3)> 0:
        cand3 = np.max([func_obj3(tup) for tup in S3])

    return max(cand1, cand2, cand3, cand4)


def get_root_kde_lw_c0(u=-0.5*np.exp(-0.5)):
    assert(u >= -np.exp(-1))
    y0 = lambertw(u).real
    y1 = lambertw(u,k=-1).real
    roots = []
    if -y0 > 0.5:
        l0 = np.sqrt(-2.0*y0-1)
        roots.append(l0)
    if -y1 > 0.5:
        l1 = np.sqrt(-2.0*y1-1)
        roots.append(l1)
    return roots

def get_root_kde_phi0_c0(func_phi1,gamma, h):
    func_root = lambda w: np.exp(-0.5*func_phi1(w)**2)-(1+w*(w-1)*func_phi1(w)**2)*np.exp(-0.5*w*w*func_phi1(w)**2)
    l = 0.01*h*gamma
    r = 2.0*h*gamma
    root_z = root_scalar(func_root,bracket=[l,r])

    return root_z.root


def get_kde_smooth_c0(gamma, h):
    func_phi1 = lambda w: h*(-gamma+np.sqrt(gamma*gamma+4.0*w*w/h/h))/2.0/w/w

    func_obj1 = lambda w, z: z/h*w*np.exp(-0.5*w*w*z*z)*np.exp(-gamma*z*h)
    func_obj2 = lambda l: abs(np.exp(-0.5*l*l)-1)/l/h
    func_obj3 = lambda z: z/h*np.exp(-0.5*z*z)*np.exp(-gamma*z*h)

    cand1 = 0
    S1 =[]
    w11 = get_root_kde_phi0_c0(func_phi1,gamma,h)
    z11 = func_phi1(w11)
    S1.append((w11,z11))
    if len(S1) > 0:
        cand1 = np.max([func_obj1(tup[0],tup[1]) for tup in S1])

    cand2 = 0
    S2 = get_root_kde_lw_c0()
    if len(S2) > 0:
        cand2 = np.max([func_obj2(tup) for tup in S2])

    z3 = func_phi1(1.0)
    cand3 = func_obj3(z3)

    return max(cand1, cand2, cand3)

def get_kde_smooth_analytic(x, gamma, h):
    c = np.linalg.norm(x)/h#np.linalg.norm(x-t)/h
    if (c > 0):
        sens = get_kde_smooth_c1(c, gamma, h)
    else:
        sens = get_kde_smooth_c0(gamma, h)
    return sens


def get_cauchy4_rv_secure(m=1):
    rand = SystemRandom()
    if m==1:
        U = rand.random()
        Z = get_cauchy4_cdf_inv(U)
        return Z
    else:
        Zs = []
        for i in range(m):
            U = rand.random()
            Zs.append(get_cauchy4_cdf_inv(U))
        return np.array(Zs)

def get_cauchy6_rv_secure(m=1):
    rand = SystemRandom()
    if m==1:
        U = rand.random()
        Z = get_cauchy6_cdf_inv(U)
        return Z
    else:
        Zs = []
        for i in range(m):
            U = rand.random()
            Zs.append(get_cauchy6_cdf_inv(U))
        return np.array(Zs)
        
def get_standardt_secure(nu,size=1):
    rand = SystemRandom()
    rv = t(nu)
    Zs = []
    for i in range(size):
        U = rand.random()
        Z = rv.ppf(U)
        Zs.append(Z)
    return np.array(Zs)
