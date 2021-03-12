import numpy as np
import pandas as pd
import periodictable as perd

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
    
def tof_expected(ke_in=np.array(17500),species = 'H',
                tof_dims = {'tof0':50,'tof1':22.5,'tof2':27.7},
                quadrant = 0,include_delay = True,q = 1,e_loss = 0):
    if type(ke_in)!=np.array:
        ke = np.array(ke_in).reshape(-1)
    else:
        ke = ke_in

    if type(species) == str:
        spec_list = [species]
    else:
        spec_list = species

    units = ['','[amu]','[eV]','[cm/ns]','bool','[ns]','[ns]','[ns]','[ns]']
        
    dfs = []
    for spec in species:
        tof3_expected = quadrant*4
        d_out = {thing:[] for thing in ['species','m','ke','v0','delay']+list(tof_dims.keys())+['tof3']}
        
        m = perd.elements.symbol(spec).mass
        
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
