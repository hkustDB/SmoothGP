import numpy as np
from scipy.stats import lognorm
from scipy.optimize import curve_fit
from utils import extract_loc_data, get_hhincome_dataCALI, get_weighted_perc


def test_var_bymech(data, vars, mech, name_mech, aggr_func, seed=0,folder_out='',header=''):#, filelog=None):
    np.random.seed(seed=seed)
    var_res = []
    for var in vars:
        y = mech(data, var)
        res = aggr_func(y)
        var_res.append(res)
    if not(folder_out==''):
        np.savetxt(folder_out+'/results_'+name_mech+'.txt', var_res, delimiter=';',  header=header)
    return var_res

def get_hhincome_paramsCALI(ncount,seed=0):
    hhinc_data, wt_data = get_hhincome_dataCALI()
    wt_total = np.sum(wt_data)
    wts = np.array(wt_data)/wt_total

    np.random.seed(seed=seed)
    samp = np.random.choice(hhinc_data,ncount,p=wts)
    return samp

def get_hhincome_debt_paramsCALI(ncount,seed,rho=0.4,s1=1.791,bTrain=True):
    if bTrain:
        inc_data, wt_data = get_hhincome_dataCALI()
        wt_data = wt_data/np.sum(wt_data)
        ps = [0.1, 0.3, 0.6, 0.8, 0.9]
        pts = get_weighted_perc(inc_data,wt_data,ps)
        func_pdf = lambda x, mu, ss: lognorm.ppf(x,s=ss,scale=np.exp(mu))
        popt, pcov = curve_fit(func_pdf, xdata=ps, ydata=pts, p0=[np.log(3000), 1.0])
        (mu0, sc0) = popt
    else:
        mu0 = 11.49399503122476
        sc0 = 0.8290709468895202
    mu1 = np.log(s1*np.exp(mu0+0.5*sc0*sc0))-0.5*sc0*sc0
    cov = np.array([[sc0*sc0, rho*sc0*sc0], [rho*sc0*sc0, sc0*sc0]])
    np.random.seed(seed=seed)
    xx = np.random.multivariate_normal([mu0,mu1],cov,size=ncount)
    yy = np.exp(xx)
    return yy

def get_loc_data_params(mcell,ncount=0,seed=0):
    nymvc_file = './data/ny_mvcdata.csv'
    loc_tups = np.array(extract_loc_data(nymvc_file))
    n = len(loc_tups)
    if ncount > 0 :
        np.random.seed(seed=seed)
        inds = np.random.randint(low=0,high=n,size=ncount)
        loc_tups = loc_tups[inds]
    v_max = np.max(loc_tups,axis=0)
    v_min = np.min(loc_tups,axis=0)
    w = v_max-v_min
    margin = (2*mcell-(w%mcell))/2
    v_max = v_max+margin
    v_min = v_min-margin
    return loc_tups, v_max, v_min
