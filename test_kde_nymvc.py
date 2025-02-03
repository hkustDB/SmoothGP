import numpy as np
from time import time
from baseline.algo import GP_loc_vec, GP_post_func_vec
from ldp.algo import ldp_lap_vec
from smooth.algo import smooth_mechanism_tvec, kde_smooth_vec
from test_functions import get_loc_data_params, test_var_bymech
from utils import check_folder

seeds = [128406,692964,318577,380019,230331,546042,160819,971124,845930,947327,456960,118776,142428,569987,622965,326371,794499,513438,299986,380574,749065,580182,480143,756660,247362,160729,794232,833580,267433,746601,831759,583124,530721,565167,565833,516521,627884,140305,756739,478991,560742,756344,926802,94969,799689,852752,281791,28904,324295,580111,988123,933278,399592,992941,812511,666806,82273,309811,435303,441094,357977,568865,667822,311231,636435,400118,981846,841217,857127,9269,71390,476525,480924,387781,39319,384686,547019,467627,401362,791414,681323,80120,945400,545977,741272,728817,94056,411718,154896,402090,662495,493673,478616,723388,535931,664312,168217,797170,932312,686987]
params = 'folder_out;jrep;varname;n;eps;h;gcell;pg3;epsL;seed\n'
params = params + '<folder_out>;<jrep>;<varname>;<n>;<eps>;<h>;<gcell>;<pg3>;<epsL>;<seed>'
varname = 't'
name_np = 'kde_np'
name_base0 = 'kde_base0'
name_base1 = 'kde_base1'
name_base2 = 'kde_base2'
name_smootht3 = 'kde_smootht3'
name_ldp_lap = 'kde_lap'
filelog = None
folder_prefix = './results/test_nymvc1w/test_kde_<gcell>_<eps>_n<n>k<trange>'
mreps = 10
gcell = 60

seed_data = seeds[-1]
n = 200000#100000#400000

samp, v_max, v_min = get_loc_data_params(gcell,ncount=n,seed=seed_data)
tx = np.linspace(v_min[0],v_max[0],gcell+1)
ty = np.linspace(v_min[1],v_max[1],gcell+1)
ts = np.array([[[tx[i],ty[j]] for i in range(gcell+1)] for j in range(gcell+1)])
ts = np.reshape(ts,((gcell+1)*(gcell+1),2))
kvars = len(ts)

x_cell = tx[1]-tx[0]
y_cell = ty[1]-ty[0]
w = np.sqrt(x_cell*x_cell+y_cell*y_cell)
h = np.floor(1.0*w)#np.floor(1.2*w)#np.floor(0.8*w)
eps = 1.0/1000#1.0/500#1.0/2000#
epsL = eps*1000

print('n: ', n)
print('eps: ', eps)
print('gcell: ', gcell)
print('mreps: ', mreps)
print('h: ',h)

func = lambda x, t: np.exp(-0.5*np.linalg.norm(x-t,axis=0)**2/h/h)
np_func = lambda arr_x, t: [np.full(mreps, func(x,t)) for x in arr_x]

GP_func1 = lambda x, t: np.linalg.norm(x-t)
post_func1 = lambda l: np.exp(-0.5*l*l/h/h)
sens1 = 1.0

GP_func2 = func
post_func2 = lambda y: y
sens2 = np.exp(-0.5)/h

sensL = 1.0

pg3 = 1.0/3.0
mech_base0 = lambda arr_x, t: [func(np.transpose(GP_loc_vec(x, eps, mreps=mreps)),np.reshape(t,(2,1))) for x in arr_x]
mech_base1 = lambda arr_x, t: GP_post_func_vec(arr_x, eps, (lambda x: GP_func1(x,t)), sens1, post_func1, mreps)
mech_base2 = lambda arr_x, t: GP_post_func_vec(arr_x, eps, (lambda x: GP_func2(x,t)), sens2, post_func2, mreps)
mech_smootht3 = lambda arr_x, t: kde_smooth_vec(arr_x, eps, (lambda x: func(x,t)), t, h, mreps=mreps, smooth_mech_vec=smooth_mechanism_tvec,nu=3,pg=pg3)
mech_lap = lambda arr_x, t: ldp_lap_vec(arr_x, (lambda x: func(x,t)), epsL, mreps, sensL)
aggr_func = lambda y: np.sum(y,axis=0)/len(y)

bsize = 50
jstart = 0
jend = jstart+int(np.ceil(kvars/bsize))
var_range = range(jstart,jend)
tick = time()
for jrep in var_range:
    js = jrep*bsize
    je = min(js+bsize,kvars)
    vs = ts[js:je]
    if not(folder_prefix==''):
        folder_out = folder_prefix.replace('<gcell>','cell'+str(gcell)).replace('<eps>','eps'+str(eps)).replace('<trange>','t'+str(js+1)+'-'+str(je)).replace('<n>',str(int(n/1000)))
    if not(folder_out==''):
        folder_out = check_folder(folder_out)
        filelog = open(folder_out+'/params.txt', 'w+')
        lineout = params.replace('<folder_out>', folder_out).replace('<varname>',varname).replace('<n>',str(n)).replace('<jrep>',str(jrep))
        lineout = lineout.replace('<eps>',str(eps)).replace('<gcell>',str(gcell)).replace('<seed>',str(seed_data))
        lineout = lineout.replace('<h>',str(h)).replace('<pg3>',str(pg3)).replace('<epsL>',str(epsL))
        filelog.write(lineout)
        filelog.close()
        header = ';'.join(['rep '+str(j) for j in range(mreps)])
        np.savetxt(folder_out+'/results_'+varname+'.txt', vs, delimiter=';',  header=header)
        test_var_bymech(samp, vs, np_func, name_np, aggr_func=aggr_func,seed=seeds[jrep],folder_out=folder_out,header=header)
        test_var_bymech(samp, vs, mech_base0, name_base0, aggr_func=aggr_func,seed=seeds[jrep],folder_out=folder_out,header=header)
        test_var_bymech(samp, vs, mech_base1, name_base1, aggr_func=aggr_func,seed=seeds[jrep],folder_out=folder_out,header=header)
        test_var_bymech(samp, vs, mech_base2, name_base2, aggr_func=aggr_func,seed=seeds[jrep],folder_out=folder_out,header=header)
        test_var_bymech(samp, vs, mech_lap, name_ldp_lap, aggr_func=aggr_func,seed=seeds[jrep],folder_out=folder_out,header=header)
        test_var_bymech(samp, vs, mech_smootht3, name_smootht3, aggr_func=aggr_func,seed=seeds[jrep],folder_out=folder_out,header=header)
print('Finished. Time elapsed: ',time() - tick)
print('Results saved to: ', folder_out)
