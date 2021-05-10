import numpy as np
import pandas as pd
import periodictable as perd

# Tof3 quadrant values, determined through calibration
tof_3_peaks = [0.6900568049562702, 4.018218774092078, 7.277728085556459, 11.991842880356202]
# electron flight time determined through simulation
t_elec = {1:4.375828,2:4.097131}

def tof_expected_spec(species = ['H'],ke=17500,tof_dims = {'tof0':50,'tof1':22.5,'tof2':27.7},
                     quadrant = 0,include_delay = True,q = 1,e_loss = 0):
    tof3_expected = quadrant*4
    d_out = {thing:[] for thing in ['species','m','ke','v0','delay']+list(tof_dims.keys())+['tof3']}
    units = ['','[amu]','[eV]','[cm/ns]','bool','[ns]','[ns]','[ns]','[ns]']

    for spec in species:    
        m = perd.elements.symbol(spec).mass
        v0 = v_00(m,ke,q)*(1-e_loss)**2
        v1 = v0*(1-e_loss)**2
        v_t = {'tof0':(v0+v1)/2,'tof1':v1,'tof2':v0}
        d_out['species'].append(spec)
        d_out['m'].append(m)
        d_out['ke'].append(ke)
        d_out['v0'].append(v0)
        d_out['delay'].append(include_delay)
        for lab,val in tof_dims.items():
            tof_offset = 0
            if include_delay == True:
                if lab == 'tof0':
                    toff_offset = -tof3_expected/2
                elif lab == 'tof1':
                    toff_offset = tof3_expected/2
            d_out[lab].append(val/v_t[lab]+toff_offset)
        d_out['tof3'].append(tof3_expected)
    df = pd.DataFrame(d_out)
    return(pd.DataFrame(df.values,
        columns = pd.MultiIndex.from_arrays([list(df.keys().values),units],names  =['','Units'])))
    
def tof_expected(ke_in=np.array(17500),
                 species = 'H',
                 mass = None,
                tof_dims = {'tof0':50,'tof1':22.5,'tof2':27.7},
                quadrant = 0,include_delay = True,q = 1,e_loss = 0):
    if type(ke_in)!=np.array:
        ke = np.array(ke_in).reshape(-1)
    else:
        ke = ke_in


    units = ['','[amu]','[eV]','[cm/ns]','bool','[ns]','[ns]','[ns]','[ns]']
        
    dfs = []
    for spec in species:
        tof3_expected = quadrant*4
        d_out = {thing:[] for thing in ['species','m','ke','v0','delay']+list(tof_dims.keys())+['tof3']}
    
        if mass == None:
            m = perd.elements.symbol(spec).mass
        else: 
            m = mass

        v0 = v_00(m,ke,q)*(1-e_loss)**2
        v1 = v0*(1-e_loss)**2
        v_t = {'tof0':(v0+v1)/2,'tof1':v1,'tof2':v0}
        d_out['species'] = np.array([spec]*len(ke))
        d_out['m']=np.array([m]*len(ke))
        d_out['ke']=ke
        d_out['v0']=v0
        d_out['delay']=np.array([include_delay]*len(ke))
        for lab,val in tof_dims.items():
            tof_offset = 0
            if include_delay == True:
                if lab == 'tof0':
                    toff_offset = -tof3_expected/2
                elif lab == 'tof1':
                    toff_offset = tof3_expected/2
            d_out[lab]=val/v_t[lab]+toff_offset
        d_out['tof3']=np.array([tof3_expected]*len(ke))
        dfs.append(pd.DataFrame(d_out))
    dfs = pd.concat(dfs, ignore_index = True)
    dfs.columns = pd.MultiIndex.from_arrays([list(dfs.keys().values),units],names  =['','Units']) 
    return(dfs)
    
def v_00(m,Vinc=7000,q = 1):
    amu_c = 1.66*10**-27
    cm_c = 10**6
    qVinc = Vinc*1.6*10**-19*q
    return(np.sqrt(qVinc/m*2/amu_c)/cm_c)


def delay_line_offset(tof3=tof_3_peaks):

    A = np.array([ [ 1, 1, 1, 1],
                   [-1, 1, 1, 1],
                   [-1,-1, 1, 1],
                   [-1,-1,-1, 1]])

    B = np.flip(np.array(tof3))
    d0,d1,d2,d3 =(np.linalg.inv(A).dot(B))
    b0 = np.array([0,d0,d1+d0,d2+d1+d0])
    b3 = np.array([d0+d1+d2+d3,d1+d2+d3,d2+d3,d3])

    ft3 = (lambda x,y: abs(x-y))
    return(pd.DataFrame(np.stack([np.arange(4),ft3(b0,b3),b0,b3]).T,columns = ['Q','tof3','b0','b3']))


def remove_delay_line(tof0,tof1,tof2,tof3):

    from scipy.interpolate import interp1d
    def delay_interp(tof3,d_effects):
        f_rn = interp1d(d_effects['tof3'],d_effects['Q'],kind = 'nearest',
                            bounds_error = False,fill_value="extrapolate")
        return(d_effects[['b0','b3']].loc[f_rn(tof3).astype(int)]) 

    # define the newly calculated tof values
    delay = delay_interp(tof3,delay_line_offset())

    dat = {}
    dat['tof0_ac'] = tof0-delay['b0'].values +t_elec[1]
    dat['tof1_ac'] = tof1-delay['b3'].values + t_elec[1]
    dat['tof2_ac'] = tof2+t_elec[2] - t_elec[2]

    return(dat)
