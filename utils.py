import numpy as np
import pandas as pd
from os import makedirs
from os.path import isdir


def deg_to_rad(deg):
    rad = deg/360.*2*np.pi
    return rad

def convert_coord(lat, long, long0=0, R=6371000.0):
    x = R*(long-long0)
    y = R*np.log(np.tan(0.25*np.pi+0.5*lat))
    return [x, y]


def extract_loc_data(filename,R=6371000.0,loc_header='LOCATION',maxcount=0,seed=0,xmin=-8.3e+6,xmax=-8.19e+6,ymin=4.91e+6,ymax=5.0e+6):
    loc_data = pd.read_csv(filename,delimiter=',',usecols=[loc_header],header=0,skip_blank_lines=True)
    loc_data = loc_data[loc_header]
    n = len(loc_data)
    ncount = n
    if maxcount > 0:
        ncount = maxcount
    loc_tups = []
    count = 0
    if maxcount > 0 and seed > 0:
        np.random.seed(seed=seed)
        inds = np.random.randint(low=0,high=n,size=2*ncount)
        loc_data = loc_data[inds]
    for loc_txt in loc_data:
        if count >= ncount:
            break
        if not(pd.isna(loc_txt)):
            loc = eval(loc_txt)
            if loc[0]>0 and loc[1]<0:
                loc_coord = convert_coord(deg_to_rad(loc[0]),deg_to_rad(loc[1]),R=R)
                if (loc_coord[0] <= xmax) and (loc_coord[0] >= xmin) and (loc_coord[1] <= ymax) and (loc_coord[1] >= ymin):
                    loc_tups.append(loc_coord)
                    count = count + 1
    return loc_tups
            
def check_folder(folder):
    if not isdir(folder):
        makedirs(folder)
        return folder
    else:
        if folder[-1] == ')':
            j = int(folder[-2:-1])+1
            foldernew = folder[:-3]+'('+str(j)+')'
            foldernew = check_folder(foldernew)
            return foldernew
        else:
            foldernew = folder+'('+str(1)+')'
            foldernew = check_folder(foldernew)
            return foldernew

def get_weighted_perc(data, weights, ps):
    inds = np.argsort(data)
    data_sorted = data[inds]
    wts_sorted = weights[inds]
    cdf_wt = np.cumsum(wts_sorted)
    pts = []
    for p in ps:
        istart = 0
        iend = len(cdf_wt)
        while (istart < iend):
            im = int(np.floor(0.5*(istart+iend)))
            if cdf_wt[im] <= p and cdf_wt[im+1] > p:
                ifound = im
                break
            elif cdf_wt[im] > p:
                iend = im
            else:
                istart = im
        if istart>=iend:
            ifound = istart
        pts.append(data_sorted[ifound])
    return pts


def get_hhincome_dataCALI(hhinc_header='HHINCOME', hwt_header='HHWT', sta_header='STATEICP', na_vals=[9999998,9999999],states=[71]):
    filename = './data/usa_00006.csv'
    data = pd.read_csv(filename,delimiter=',',skip_blank_lines=True)
    hhinc_data0 = data[hhinc_header].to_numpy()
    wt_data0 = data[hwt_header].to_numpy()
    sta_data0 = data[sta_header].to_numpy()
    if len(states) > 0:
        inc_states = states
    else:
        inc_states = list(set(sta_data0))
    n = len(hhinc_data0)
    hhinc_data = []
    wt_data = []
    for i in range(n):
        f = hhinc_data0[i]
        w = wt_data0[i]
        s = sta_data0[i]
        if (s in inc_states) and (f not in na_vals):
            hhinc_data.append(f)
            wt_data.append(w)
    return np.array(hhinc_data), np.array(wt_data)
