import numpy as np
from time import time
from baseline.algo import GP_loc_vec, GP_post_func_vec
from ldp.algo import ldp_lap_vec
from smooth.algo import smooth_mechanism_tvec, thres_smooth_vec
from smooth.utils import func_theta
from utils import check_folder
from test_functions import get_hhincome_paramsCALI, test_var_bymech


seeds = [128406,692964,318577,380019,230331,546042,160819,971124,845930,947327,456960,118776,142428,569987,622965,326371,794499,513438,299986,380574,749065,580182,480143,756660,247362,160729,794232,833580,267433,746601,831759,583124,530721,565167,565833,516521,627884,140305,756739,478991,560742,756344,926802,94969,799689,852752,281791,28904,324295,580111,988123,933278,399592,992941,812511,666806,82273,309811,435303,441094,357977,568865,667822,311231,636435,400118,981846,841217,857127,9269,71390,476525,480924,387781,39319,384686,547019,467627,401362,791414,681323,80120,945400,545977,741272,728817,94056,411718,154896,402090,662495,493673,478616,723388,535931,664312,168217,797170,932312,686987]
params = 'folder_out;jrep;varname;n;eps;thres;pg3;epsL;seed\n'
params = params + '<folder_out>;<jrep>;<varname>;<n>;<eps>;<thres>;<pg3>;<epsL>;<seed>'
varname = 'thres'
name_np = 'thres_np'
name_base0 = 'thres_base0'
name_base2 = 'thres_base2'
name_smootht3 = 'thres_smootht3'
name_ldp_lap = 'thres_lap'
name_taufunc = 'thres_taufunc'
filelog = None
folder_prefix = './results/test_thres_<thres>_<eps>_calihh<jrep>'
jrep = 0#1#2#3#4
mreps = 100

seed_data = seeds[-1]
ns = [400000,800000,1600000,3200000,6400000]
kvars = len(ns)
samps = {}
for j in range(kvars):
    n = ns[j]
    samp = get_hhincome_paramsCALI(n,seed=seed_data)
    samps[n] = samp

tol = 800#1200#2000
#48000#80000#32000#
#40000#16000#24000#
#20000#12000#8000#
eps = 1.0/tol#
thres = 10000#
#1000000
#500000
#100000
tau = min(0.2*thres,2.0*tol)
tol_fix = 1200#24000#12000#48000#
epsL = eps*tol_fix

print('n: ', ns)
print('eps: ', eps)
print('thres: ', thres)
print('varname: ', varname)

func = lambda x: (x>thres)
func_vec = lambda x: (x>np.full(mreps,thres))
np_func = lambda arr_x, n: [np.full(mreps, func(x)) for x in arr_x[n]]

func_vec_tau = lambda arr_x, n: [np.full(mreps, func_theta(x, thres, tau)) for x in arr_x[n]]
func_tau_post = lambda arr_y: [func_theta(y, thres, tau) for y in arr_y]

GP_func2 = lambda x: func_theta(x, thres, tau)
post_func2 = lambda y: y
sens2 = 1.0/tau

sensL = 1.0

pg3 = 1.0/3.0
mech_base0 = lambda arr_x, n: [func_tau_post(GP_loc_vec(x, eps, mreps=mreps)) for x in arr_x[n]]
mech_base2 = lambda arr_x, n: GP_post_func_vec(arr_x[n], eps, GP_func2, sens2, post_func2, mreps)
mech_smootht3 = lambda arr_x, n: thres_smooth_vec(arr_x[n], eps, thres, tau, mreps=mreps, smooth_mech_vec=smooth_mechanism_tvec,nu=3,pg=pg3)
mech_lap = lambda arr_x, n: ldp_lap_vec(arr_x[n], GP_func2, epsL, mreps, sensL)
aggr_func = lambda y: np.sum(y,axis=0)/len(y)


if not(folder_prefix==''):
      folder_out = folder_prefix.replace('<thres>','t'+str(thres)).replace('<eps>','eps'+str(eps)).replace('<jrep>','j'+str(jrep))
if not(folder_out==''):
        folder_out = check_folder(folder_out)
        filelog = open(folder_out+'/params.txt', 'w+')
        lineout = params.replace('<folder_out>', folder_out).replace('<varname>',varname).replace('<n>',str(ns)).replace('<jrep>',str(jrep))
        lineout = lineout.replace('<eps>',str(eps)).replace('<thres>',str(thres)).replace('<seed>',str(seed_data))
        lineout = lineout.replace('<pg3>',str(pg3)).replace('<epsL>',str(epsL))
        filelog.write(lineout)
        filelog.close()

res_data = {}
tick = time()
res_data[name_np] = test_var_bymech(samps, ns, np_func, name_np, aggr_func=aggr_func,seed=seeds[jrep])
res_data[name_base0] = test_var_bymech(samps, ns, mech_base0, name_base0, aggr_func=aggr_func,seed=seeds[jrep])
res_data[name_base2] = test_var_bymech(samps, ns, mech_base2, name_base2, aggr_func=aggr_func,seed=seeds[jrep])
res_data[name_smootht3] = test_var_bymech(samps, ns, mech_smootht3, name_smootht3, aggr_func=aggr_func,seed=seeds[jrep])
res_data[name_ldp_lap] = test_var_bymech(samps, ns, mech_lap, name_ldp_lap, aggr_func=aggr_func,seed=seeds[jrep])
res_data[name_taufunc] = test_var_bymech(samps, ns, func_vec_tau, name_taufunc, aggr_func=aggr_func,seed=seeds[jrep])
if not(folder_out==''):
    header = ';'.join(['rep '+str(j) for j in range(mreps)])
    for name in res_data.keys():
        np.savetxt(folder_out+'/results_'+name+'.txt', res_data[name], delimiter=';',  header=header)
print('Finished. Time elapsed: ',time() - tick)
print('Results saved to: ', folder_out)
