import numpy as np
import pandas as pd

amu_c = 1.66*10**-27
Vpac = 7000
cm_c = 10**6
qV = Vpac*1.6*10**-19
di = {'tof0':50,'tof1':22.5,'tof2':27.7}


# Filters for the ToF checksum, and for golden triples. 
# Additionally, filters for a maximum speed, providing min ToF for each of the legs
def tof_filter(df_gt,maxv= 1.4,tof_dims = {'tof0':50,'tof1':22.5,'tof2':27.7}):

    speed_max = np.logical_and.reduce([tof_dims[nam]/df_gt[nam]< maxv for nam in tof_dims])
    checksum = abs(df_gt['tof0']+df_gt['tof3']-df_gt['tof2']-df_gt['tof1'])<2
    loco = np.logical_and.reduce([checksum,
                                 speed_max,
                                 df_gt['type'] == 0])
    # di = tof_dims
    # loco = ((abs(df_gt['tof0']+df_gt['tof3']-df_gt['tof2']-
    #              df_gt['tof1'])<2)&(df_gt['type'] == 0) &
    #          (di['tof0']/df_gt['tof0']< maxv) &(di['tof1']/df_gt['tof1']<\
    #           maxv)&(di['tof2']/df_gt['tof2']< maxv))
    # loco = abs(df_gt['tof0']+df_gt['tof3']-df_gt['tof2']-df_gt['tof1'])<2
    return(df_gt.loc[loco])

def remove_delay(df):
    df_nd = df.copy()
    df_nd['tof0'] =df_nd['tof0']+df_nd['tof3']/2 
    df_nd['tof1'] =df_nd['tof1']-df_nd['tof3']/2 
    return(df_nd)

def remove_delay_post_cal(df):
    df_nd = df.copy()
    df_nd['tof0'] =df_nd['tof0']+df_nd['tof3']/2 
    df_nd['tof1'] =df_nd['tof1']-df_nd['tof3']/2 
    return(df_nd)

def tof_speeds(df_gt,tof_dims = {'tof0':50,'tof1':22.5,'tof2':27.7}):
    mindt = 0

    di = tof_dims

    vs = {}
    for lab in di:
        vs[lab] = di[lab]/df_gt[lab]
    vs['tof1'] = di['tof1']/(df_gt['tof1']-df_gt['tof3'])
    return(vs)


def calc_v0_2(v1_2,v2_2,a,pm):
    return(1/2*(pm*(a*(v1_2 - v2_2)*np.sqrt(a**2*v1_2**2 - 2*a**2*v1_2*v2_2 +
                                      a**2*v2_2**2 + 8*v1_2**2 + 8*v1_2*v2_2))/(v1_2 + v2_2)
            + (4*a**2*v1_2**2)/(v1_2 + v2_2) - 3*a**2*v1_2 + a**2*v2_2 + 2*v1_2))
    # return((v1_2+np.sqrt(5*v1_2**2-2*v1_2*v2_2+v2_2+v2_2**2)-v2_2)/(2*np.sqrt(v1_2)))

def calc_v0shift(v1,v2,a):
    v1_2 = v1**2
    v2_2 = v2**2
    v2_sh_2 = shift_v2(v1,v2)**2
    E_loss1 = (v1_2-v2_2)
    E_loss_sh = (v1_2-v2_sh_2)

    v0_2 = calc_v0_2(v1_2,v2_sh_2,a,1)
    v00_2 = calc_v0_2(v1_2,v2_2,a,1)
    E_loss0 = (v00_2-v1_2)

    swap_cond = E_loss_sh<0
    v0_2[swap_cond] = calc_v0_2(v1_2[swap_cond],v2_sh_2[swap_cond],abs(a[swap_cond]),-1)
    return(np.sqrt((v0_2)))

def calc_v0(v1,v2,a,a2):
    v1_2 = v1**2
    v2_2 = v2**2
    E_loss1 = (v1_2-v2_2)

    v0_2 = calc_v0_2(v1_2,v2_2,a,1)
    E_loss0 = (v0_2-v1_2)

    swap_cond = E_loss1<0
    v0_2[swap_cond] = calc_v0_2(v1_2[swap_cond],v2_sh_2[swap_cond],abs(a[swap_cond]),-1)
    
    swap_cond = np.logical_and(E_loss1/E_loss0<1,E_loss1>0)
    v0_2[swap_cond] = calc_v0_2(v1_2[swap_cond],v2_sh_2[swap_cond],
                                abs(a[swap_cond]**2),1)

    return(np.sqrt(abs(v0_2)))

    
def get_m(v1,v2):
    a1=v_to_v0((v1+v2)/2)-1
    v0 = calc_v0shift(v1,v2,a1)
    m = qV/(v0*cm_c)**2/amu_c*2
    return(m)


def v_00(m,Vinc=7000,q = 1):
    qVinc = Vinc*1.6*10**-19*q
    return(np.sqrt(qVinc/m*2/amu_c)/cm_c)

def tof_expected(m,ke,tof_dims = {'tof0':50,'tof1':22.5,'tof2':27.7},quadrant = 0,include_delay = True):
    # print('Expected ToF [nS]: Quadrant %d'%quadrant)
    # print('%5s  %7s,  %7s '%('','No Delay','Delay'))
    tof3_expected = quadrant*4
    tof_out = {}
    for lab,val in tof_dims.items():
        tof_offset = 0
        tof = val/v_00(m,ke)
        if include_delay == True:
#         if lab=='tof2':
#             tof_offset = 0 
            if lab == 'tof0':
                toff_offset = -tof3_expected/2
            elif lab == 'tof1':
                toff_offset = tof3_expected/2
        tof_out[lab] = tof+toff_offset
            # print('%.5s:  %3.4f,  %3.4f '%(lab,tof,tof+toff_offset))   



def get_a(v1,v2):
    v1_h = np.mean(v1)
    mid_v = v_00(1)-v_00(16)
    mid_v0 = (v_00(1)+v_00(16))/2
    mid_v1 = mid_v-v_00(1)+v1_h


    v1_o = np.median(v1[v1<(mid_v-v_00(1)+v1_h)])

    def calc_a2(v):
        v_h = np.mean(v)
        mid_v = mid_v0-v_00(1)+v_h
        v_o = np.median(v[v<mid_v])
        return(np.sqrt(abs(((v-v_o)/(v_h-v_o)*(v_00(1)-v_00(16)))+v_00(16))/(v_00(1))))
    
    def calc_a(v):
        # return(np.sqrt((((v-v1_o)/(v1_h-v1_o)*(v_00(1)-v_00(16)))+v_00(16))/(v_00(1))))
        # return(np.sqrt((((v-v1_o)/(v1_h-v1_o)*(v_00(1)-v_00(16)))+v_00(16))))
        return(((((v-v1_o)/(v1_h-v1_o)*(v_00(1)-v_00(16)))+v_00(16))))
    a1 = calc_a((v1+v2)/2)
    a2 = calc_a(v2)
    return(a1,a2)


def shift_v2(v1,v2):
    def get_ho(v):
        h = np.mean(v)
        o = np.mean(v[v<h/np.sqrt(16)])
        return(h,o)
    v1_h,v1_o = get_ho(v1)
    v2_h,v2_o = get_ho(v2)
    return((v2-v2_o)/(v2_h-v2_o)*(v1_h-v1_o)+v1_o)



def v_to_v0(v):
    v1_h = np.mean(v)
    # mid_v = v_00(1)-v_00(16)
    mid_v = (v_00(1)+v_00(16))/2
    mid_v1 = mid_v-v_00(1)+v1_h
    v1_o = np.median(v[v<(mid_v-v_00(1)+v1_h)])
    return((((v-v1_o)/(v1_h-v1_o)*(v_00(1)-v_00(16)))+v_00(16))/v)



def define_species(df_tof,h_range = [0,5],o_range = [12,30]):

    df_gt = tof_filter(df_tof)
    vf = tof_speeds(df_gt)
    v1 = vf['tof2']
    v2 = vf['tof1']
    m = get_m(v1,v2)
    e_loss = (v1**2-v2**2)/(2*v1**2)
    logm = e_loss>0

    species = np.chararray((len(m)))
    species[:] = 'z'
    species[np.logical_and.reduce([m<h_range[1],m>h_range[0],
                                               logm])] = 'H'
    species[np.logical_and.reduce([m<o_range[1],m>o_range[0],
                                               logm])] = 'O'
    for nam,vals in zip(['m','species','v1','v2'],[m,species.decode('utf8'),v1,v2]):
        df_gt.insert(df_gt.columns.get_loc('tof3'),nam,vals)
    return(df_gt)
    

def lut_species(df_tof):

    df_gt = df_tof[tof_checksum(df_tof)]
    tofs = {'tof0+':df_gt['tof0']+df_gt['tof3']/2,'tof2':df_gt['tof2']}
    # tof_selects = {'tof2': {'H':[13,40],'O':[75,200]},'tof0+':{'H':[30,70],'O':[150,300]}}
    tof_selects = {'H':{'tof2':[13,40],'tof0+':[30,70]},'O':{'tof2':[75,200],'tof0+':[150,300]}}

    species = np.chararray((df_gt.shape[0]))
    species[:] = 'z'
    # thing = {}
    for spec,ts in tof_selects.items():
        # for t in ts:print(ts[t])
        species[np.logical_and.reduce([np.logical_and(tofs[t]>ts[t][0],
                                                      tofs[t]<ts[t][1]) for t in ts])] = spec
    # species[~tof_checksum(df_gt)] = '2'
    for nam,vals in zip(['species'],[species.decode('utf8')]):
        df_gt.insert(df_gt.columns.get_loc('tof3'),nam,vals)
    return(df_gt)


def tof_checksum(df_gt,tof3max = 15,check_max = 1,min_tof=0,tof_log = {}):

    # speed_max = np.logical_and.reduce([tof_dims[nam]/df_gt[nam]< maxv for nam in tof_dims])
    checksum = abs(df_gt['tof0']+df_gt['tof3']-df_gt['tof2']-df_gt['tof1'])<check_max
    # checksum = (abs(df_gt['tof0']+tof3_offset(df_gt)-df_gt['tof2']-df_gt['tof1']+.4).values)<check_max
    select = []
    # for lab in ['tof1','tof2','tof0']:
    for lab in ['tof2']:
        select.append(df_gt[lab]>min_tof)
    selector = []
    for lab,app_filt in tof_log.items():
        selector.append(app_filt[0](df_gt[lab],app_filt[1]))

    loco = np.logical_and.reduce([checksum,
                                 df_gt['type'] == 0,
                                 df_gt['tof3'] > 0,
                                 df_gt['tof3'] < tof3max,
                                 np.logical_and.reduce(select)])
    # di = tof_dims
    # loco = ((abs(df_gt['tof0']+df_gt['tof3']-df_gt['tof2']-
    #              df_gt['tof1'])<2)&(df_gt['type'] == 0) &
    #          (di['tof0']/df_gt['tof0']< maxv) &(di['tof1']/df_gt['tof1']<\
    #           maxv)&(di['tof2']/df_gt['tof2']< maxv))
    # loco = abs(df_gt['tof0']+df_gt['tof3']-df_gt['tof2']-df_gt['tof1'])<2
    return(loco)

def check_val(df_gt):
    return(df_gt['tof0']+df_gt['tof3']-df_gt['tof2']-df_gt['tof1'])

def delay_stuff(w = .5):
    pulse = np.array([.7,4,7.5,12])
    # w = .4
    delay_bins = np.stack([pulse-w,pulse+w]).T
    return(pulse,w,delay_bins)

def tof3_peaks(df_gt,w = .4):
    pulse,w,delay_bins = delay_stuff(w)
    loco = np.logical_or.reduce([np.logical_and(df_gt['tof3']>b[0],df_gt['tof3']<b[1]) for b in delay_bins])
    return(df_gt.loc[loco])

def tof3_offset(df_gt):

    pulse,w,delay_bins = delay_stuff()
    tof3_off = np.zeros(len(df_gt['tof3']))
    for p in pulse:
        tof3_off[np.logical_and(df_gt['tof3']>p-w,df_gt['tof3']<p+w)]=p
    return(tof3_off)

def tof3_quadrant(df_gt):

    
    pulse,w,delay_bins = delay_stuff()
    tof3_quad = np.zeros(len(df_gt['tof3']))*np.nan
    quad = 0
    for p in pulse:
        tof3_quad[np.logical_and(df_gt['tof3']>p-w,df_gt['tof3']<p+w)]=quad
        quad+=1

    # gb = df.groupby(pd.cut(df_gt['tof3'], delay_bins))
    return(tof3_quad)

def quad_group(df):
    gb = df.groupby(tof3_quadrant(df))
    return([gb.get_group(x) for x in gb.groups])