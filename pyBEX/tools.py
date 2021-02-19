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
    for h in header:
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
    head = []
    with open(fil) as l:
        for ll in l.readlines():

            if ll.strip():
                if '#' in ll:    
                    hs = ll.strip().strip('#').strip(':')
                    if hs:
                        head = []
                        if 'met' in hs.lower():
                            for h in hs.split(' '):
                                if h:head.append(h)
                else:
                    break
                
                hct+=1
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

                # Define exception to convert 'en' column to 'ch'
                if 'en' in l:
                    l['en']  = l['en'].astype(str)

                orb=[int(orbit_arc[:-1])]*l.shape[0]
                arc=[orbit_arc[-1]]*l.shape[0]

                for nam,val in zip(['arc','orbit'],[arc,orb]):
                    l.insert(1,nam,val)
                if not l.empty:
                    if calc_nep == True:
                        l.insert(4,'nep',phase_to_nep(l['phase'].values))
                    li.append(l)
                else:
                    print(cols)
                    print(head)
            except(pd.errors.EmptyDataError):
                if fail == 0:
                    out.append_stdout('E%s Import Failed [EmptyDataError] on Orbits: \n'%f.split('.')[0][-1])
                out.append_stdout('%s,'%orbit_arc)
                fail+=1

                orbit_fails+='%s, '%orbit_arc
        lb.value +=1

    lb.close()
    return((pd.concat(li) if len(li)>= 1 else pd.DataFrame()))


def mask_good_times(ds,good_times,apply_nep =False,
                        include_no_gt=False,return_mask = False,
                            nep_start_max = np.inf,nep_stop_min = -np.inf):

    gt =good_times.loc[((good_times['Start Time']>min(ds['time'])) &\
                             (good_times['Stop Time']<max(ds['time'])))]
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