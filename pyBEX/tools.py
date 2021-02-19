import numpy as np
import pandas as pd
import periodictable as perd

def import_good_times(fil):
    lines = []
    if fil =='LoGoodTimes.txt':
        header = ['orbit','Start Time','Stop Time',
                    'NEP Start','NEP End','Hi/Lo']+list('E%d'%n for n in range(1,9))
    else:
        header = open(fil,'r').read().split('#')[-1].split('\n')[0].split('\t')
    arr = pd.read_csv(fil,delim_whitespace = True,
                      comment = '#',header = None)

    head_out = []
    # start_nams = ['start','begin']
    # stop_nams = ['stop','end']
    # time_nams = ['time','gps']
    for h in header:
        # if start_nams 
        head_out.append(h.replace('begin_GPS','Start Time').replace('end_GPS','Stop Time').strip().strip('/arc'))
    arr.columns = head_out
    if type(arr['orbit'].values[0])==str:
        int_orb = arr['orbit'].str.strip('a').str.strip('b').astype(int)
        arr.drop('orbit',axis = 1,inplace = True)
        arr['orbit'] = int_orb

    if 'NEP Start' not in arr: 
        arr['NEP Start'] = np.zeros(len(arr.values))
        print("Auto Generated NEP Start Range")
    if 'NEP End' not in arr: 
        arr['NEP End'] = np.ones(len(arr.values))*60
        print("Auto Generated NEP Stop Range")
    return(arr)

def get_headder(fil):
    hct = 0
    # lines = []
    head = []
    with open(fil) as l:
        for ll in l.readlines():

            if ll.strip():
                if '#' in ll:    
                    # print(ll) 
                    # print(hct)
                    hs = ll.strip().strip('#').strip(':')
                    if hs:
                        head = []
                        if 'met' in hs.lower():
                            for h in hs.split(' '):
                                if h:head.append(h)
                else:
                    break
                
                hct+=1
    #                 lines.append(np.fromstring(ll.split('#')[0].strip().replace('  ',' '),sep = ' '))
    if head and 'met' in head[0].lower():
        head[0]= 'time'
    return(head)

def load_df(path,estep=None,dtype = '.txt',head = None,
                usecols = None,calc_nep = False):

    import glob

    if type(estep) == list:
        fils = []
        for e in estep:
            ftype = "*e%d%s"%(e,dtype)
            fils+=glob.glob(path +ftype)
    elif type(estep) == int:
        ftype = "*e%d%s"%(estep,dtype)
        fils = glob.glob(path + ftype)
    else:
        ftype = "*%s%s"%(str(estep),dtype)
        fils = glob.glob(path + ftype)
    # if head == 'auto':
    #     for head_loc in glob.glob(path + ftype):
    #         try:
    #             head = get_headder(head_loc)
    #             print(head)
    #             break
    #         except(UnboundLocalError):
    #             pass

    from ipywidgets import IntProgress,Output
    from IPython.display import display

    out = Output()
    lb = IntProgress(min=0, max=len(fils)) # instantiate the bar
    display(lb) # display the bar
    display(out)
    lb.value = 0
        
    li = []
    orb = []
    orbit_fails = ''
    fail = 0
    for f in fils:
        if 'orig' not in f and 'new' not in f:
            orbit_arc = f.split('\\')[-1].split('o')[1].split('_')[0]
            try:
                if head == 'auto' and not usecols:
                    cols = get_headder(f)
                elif head=='auto' and usecols:
                    cols = list(get_headder(f)[c] for c in usecols)
                elif type(head)==list and usecols:
                    cols = list(head[c] for c in usecols)
                elif head:
                    cols = head
                else: cols = None

                l = pd.read_csv(f,delim_whitespace = True,header = None,
                                comment = '#',names = (cols if cols else None),
                                    usecols = usecols)

                # if head == 'auto' and not usecols:
                #     l.columns = get_headder(f)
                # elif head=='auto' and usecols:
                #     l.columns = list(get_headder(f)[c] for c in usecols)
                # elif type(head)==list and usecols:
                #     l.columns = list(head[c] for c in usecols)
                # elif head:
                #     l.columns = head

                # Define exception to convert 'en' column to 'ch'
                if 'en' in l:
                    # ch = np.zeros(len(l['en'])).astype(int)

                    l['en']  = l['en'].astype(str)

                orb=[int(orbit_arc[:-1])]*l.shape[0]
                arc=[orbit_arc[-1]]*l.shape[0]

                for nam,val in zip(['arc','orbit'],[arc,orb]):
                    l.insert(1,nam,val)
                if not l.empty:
                    if calc_nep == True:
                        # nep = 360*(l['phase'].values+.5)-3
                        # nep[nep>360] = nep[nep>360]-360
                        l.insert(4,'nep',phase_to_nep(l['phase'].values))
                    li.append(l)
                else:
                    print(cols)
                    print(head)
            except(pd.errors.EmptyDataError):
                # print(f)
                # break
                if fail == 0:
                    out.append_stdout('E%s Import Failed [EmptyDataError] on Orbits: \n'%f.split('.')[0][-1])
                out.append_stdout('%s,'%orbit_arc)
                fail+=1

                orbit_fails+='%s, '%orbit_arc
        lb.value +=1
    # if orbit_fails:
    #     print('E%d Import Failed [EmptyDataError] on Orbits:'%int(f.split('.')[0][-1]))
    #     print(orbit_fails)

    lb.close()
    return((pd.concat(li) if len(li)>= 1 else pd.DataFrame()))


def mask_good_times(ds,good_times,apply_nep =False,include_no_gt=False,return_mask = False,nep_start_max = np.inf,nep_stop_min = -np.inf):
    gt =good_times.loc[((good_times['Start Time']>min(ds['time'])) &\
                             (good_times['Stop Time']<max(ds['time'])))]#.values
    if 'NEP Start' not in gt: 
        gt['NEP Start'] = np.zeros(len(gt.values))
        print(' No Start')
    if 'NEP End' not in gt: 
        gt['NEP End'] = np.ones(len(gt.values))*60
        print('No end')
    
    if apply_nep == False:
        use_cols = ['Start Time','Stop Time','NEP Start','NEP End']
        if include_no_gt == False:
            logo = list(np.logical_and(ds['time']>goodt[0],
                                        ds['time']<goodt[1]) for goodt in gt[use_cols].values if (goodt[2]<=nep_start_max and goodt[3]>=nep_stop_min))
        else:
            logo = []
            for orb in np.arange(min(gt['orbit']),max(gt['orbit'])+1):
                if orb in gt['orbit'].values:
                    for goodt in gt.loc[gt['orbit']==orb][use_cols].values:
                        if goodt[2]<=nep_start_max and goodt[3]>=nep_stop_min:
                            logo.append(np.logical_and(ds['time']>goodt[0],ds['time']<goodt[1]))
                else:
                    logo.append(ds['orbit']==orb)
            return((ds.loc[np.logical_or.reduce(logo)]if return_mask == False else np.logical_or.reduce(logo)))
    else:
        logo = list(np.logical_and.reduce(
                                        [ds['time']>goodt[0],
                                         ds['time']<goodt[1],
                                         ds['nep']>goodt[2]*6,
                                         ds['nep']<goodt[3]*6]) for goodt in gt[use_cols].values if (goodt[2]<=nep_start_max and goodt[3]>=nep_stop_min))
    return((ds.loc[np.logical_or.reduce(logo)]if return_mask == False else np.logical_or.reduce(logo)))



def phase_to_nep(phase,instrument = 'lo'):
    nep = 360*(phase+(.5 if instrument == 'lo' else 0))-3
    nep[nep>=357] = nep[nep>=357]-360
    return(nep.astype(float).round())

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
        # for nam,val in d_out.items():
        #     print(nam)
        #     print(len(val))
        dfs.append(pd.DataFrame(d_out))
    dfs = pd.concat(dfs, ignore_index = True)
    dfs.columns = pd.MultiIndex.from_arrays([list(dfs.keys().values),units],names  =['','Units']) 
    return(dfs)
    
def v_00(m,Vinc=7000,q = 1):
    amu_c = 1.66*10**-27
    cm_c = 10**6
    qVinc = Vinc*1.6*10**-19*q
    return(np.sqrt(qVinc/m*2/amu_c)/cm_c)



def gauss_filt_nan(arr, sigma,mode = 'constant'):
    from scipy import ndimage
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndimage.gaussian_filter(
            loss, sigma=sigma, mode=mode, cval=1)

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = ndimage.gaussian_filter(
            gauss, sigma=sigma, mode=mode, cval=0)
    gauss[nan_msk] = np.nan

    gauss += loss * arr

    return gauss