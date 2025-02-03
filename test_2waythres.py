import numpy as np
from time import time
from baseline.algo import GP_loc_vec, GP_post_func_vec
from ldp.algo import ldp_lap_vec
from smooth.algo import smooth_mechanism_tvec, twthres_smooth_vec
from smooth.utils import func_twtau2
from utils import check_folder
from test_functions import get_hhincome_debt_paramsCALI, test_var_bymech


seeds = [128406,692964,318577,380019,230331,546042,160819,971124,845930,947327,456960,118776,142428,569987,622965,326371,794499,513438,299986,380574,749065,580182,480143,756660,247362,160729,794232,833580,267433,746601,831759,583124,530721,565167,565833,516521,627884,140305,756739,478991,560742,756344,926802,94969,799689,852752,281791,28904,324295,580111,988123,933278,399592,992941,812511,666806,82273,309811,435303,441094,357977,568865,667822,311231,636435,400118,981846,841217,857127,9269,71390,476525,480924,387781,39319,384686,547019,467627,401362,791414,681323,80120,945400,545977,741272,728817,94056,411718,154896,402090,662495,493673,478616,723388,535931,664312,168217,797170,932312,686987]
params = 'folder_out;jrep;varname;n;eps;T1;T2;epsL;seed\n'
params = params + '<folder_out>;<jrep>;<varname>;<n>;<eps>;<T1>;<T2>;<epsL>;<seed>'
varname = 'twthres'
name_np = 'twthres_np'
name_base0 = 'twthres_base0'
name_base2 = 'twthres_base2'
name_smootht3 = 'twthres_smootht3'
name_thetafunc = 'twthres_taufunc'
name_ldp_lap = 'twthres_lap'
filelog = None
folder_prefix = './results/twgrid/test_twthresh_<T1>_<eps>_calihhincdebt<trange>'
mreps = 50
gcell = 33

seed_data = seeds[-1]
n =1600000#6400000#400000#

samp = get_hhincome_debt_paramsCALI(n,seed=seed_data,rho=0.4)


tol = 12000#8000#20000
eps = 1.0/tol
tol_fix = 12000
epsL = eps*tol_fix

T1s = np.array([10000+j*30000 for j in range(gcell+1)])
T2s = np.array([10000+j*30000 for j in range(gcell+1)])
Ts = np.array([[[T1s[i],T2s[j]] for i in range(gcell+1)] for j in range(gcell+1)])
Ts = np.reshape(Ts,((gcell+1)*(gcell+1),2))
kvars = len(Ts)


dict_taus = {}
for T in Ts:
      w = np.sqrt(T[0]*T[0]+T[1]*T[1])
      tau = min(0.2*w,2.0*tol)
      T_name = str(T[0])+'_'+str(T[1])
      dict_taus[T_name] = tau

print('n: ', n)
print('eps: ', eps)
print('T1: ', T1s)
print('T2: ', T2s)
print('varname: ', varname)


func = lambda x, T: (x[0]> T[0])*(x[1] > T[1])
np_func = lambda arr_x, T: [np.full(mreps, func(x,T)) for x in arr_x]

func_vec_tau = lambda arr_x, T: [np.full(mreps, func_twtau2(x, T1=T[0], T2=T[1], tau=dict_taus[str(T[0])+'_'+str(T[1])])) for x in arr_x]

GP_func2 = lambda x, T: func_twtau2(x, T1=T[0], T2=T[1], tau=dict_taus[str(T[0])+'_'+str(T[1])])
post_func2 = lambda y: y
sens_func2 = lambda T: 1.0/dict_taus[str(T[0])+'_'+str(T[1])]

sensL = 1.0

pg3 = 1.0/3.0
mech_base0 = lambda arr_x, T: [func(np.transpose(GP_loc_vec(x, eps, mreps=mreps)),T) for x in arr_x]
mech_base2 = lambda arr_x, T: GP_post_func_vec(arr_x, eps, (lambda x: GP_func2(x,T)), sens_func2(T), post_func2, mreps)
mech_smootht3 = lambda arr_x, T: twthres_smooth_vec(arr_x, eps, T1=T[0], T2=T[1], tau=dict_taus[str(T[0])+'_'+str(T[1])], mreps=mreps, smooth_mech_vec=smooth_mechanism_tvec,nu=3,pg=pg3)
mech_lap = lambda arr_x, T: ldp_lap_vec(arr_x, (lambda x: func(x,T)), epsL, mreps, sensL)
aggr_func = lambda y: np.sum(y,axis=0)/len(y)

bsize = 100
jstart = 0
jend = jstart+int(kvars/bsize)
var_range = range(jstart,jend)
tick = time()
for jrep in var_range:
    js = jrep*bsize
    je = min(js+bsize,kvars)
    vs = Ts[js:je]
    if not(folder_prefix==''):
        folder_out = folder_prefix.replace('<T1>','t'+str(int(T1s[0]/1000))+'k-'+str(int(T1s[-1]/1000))+'k').replace('<eps>','eps'+str(eps)).replace('<trange>','t'+str(js+1)+'-'+str(je))
    if not(folder_out==''):
        folder_out = check_folder(folder_out)
        filelog = open(folder_out+'/params.txt', 'w+')
        lineout = params.replace('<folder_out>', folder_out).replace('<varname>',varname).replace('<n>',str(n)).replace('<jrep>',str(jrep))
        lineout = lineout.replace('<eps>',str(eps)).replace('<T1>',str(T1s[0])).replace('<T2>',str(T1s[-1])).replace('<seed>',str(seed_data)).replace('<epsL>',str(epsL))
        filelog.write(lineout)
        filelog.close()
        header = ';'.join(['rep '+str(j) for j in range(mreps)])
        np.savetxt(folder_out+'/results_'+varname+'.txt', vs, delimiter=';',  header=header)
        test_var_bymech(samp, vs, np_func, name_np, aggr_func=aggr_func,seed=seeds[jrep],folder_out=folder_out,header=header)
        test_var_bymech(samp, vs, mech_base0, name_base0, aggr_func=aggr_func,seed=seeds[jrep],folder_out=folder_out,header=header)
        test_var_bymech(samp, vs, mech_base2, name_base2, aggr_func=aggr_func,seed=seeds[jrep],folder_out=folder_out,header=header)
        test_var_bymech(samp, vs, mech_smootht3, name_smootht3, aggr_func=aggr_func,seed=seeds[jrep],folder_out=folder_out,header=header)
        test_var_bymech(samp, vs, mech_lap, name_ldp_lap, aggr_func=aggr_func,seed=seeds[jrep],folder_out=folder_out,header=header)
        test_var_bymech(samp, vs, func_vec_tau, name_thetafunc, aggr_func=aggr_func,seed=seeds[jrep],folder_out=folder_out,header=header)
print('Finished. Time elapsed: ',time() - tick)
print('Results saved to: ', folder_out)
