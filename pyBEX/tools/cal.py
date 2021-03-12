import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def checksum(df_gt,check_max = 1):
    return(np.logical_and.reduce([abs(df_gt['TOF3'])<15,
            abs(df_gt['TOF0']+df_gt['TOF3']-df_gt['TOF2']-df_gt['TOF1'])<check_max]))

# Load the De files to pd data frame
def load_dt(fil,
            use_filt = ['TOF0','TOF1','TOF2','TOF3'],
            filt_triples = False,
            apply_checksum = False,
            tof3_picker = 'Auto',
               min_tof = None):
    stuff = ''
    for t in open(fil).readlines():
        if '#' in t:
            stuff+=t
    # stuff = str(''+t for t in open(fil).readlines() if '#' in t)
    head = []
    # print(stuff)
    for s in stuff.split('Group')[1].split('\n'):
        sml = s.strip().strip('#').strip()
        if sml:
            try:
                head.append(sml.split('.')[1].strip(')').strip('"'))
            except:
                print('dat import failed: %s'%fil)
                return
    # print(head)
    athing = pd.read_csv(fil,comment = '#',delim_whitespace= True,header = None,names = head)
    athing['tof0_sh'] = athing['TOF0']+athing['TOF3']/2
    athing['tof1_sh'] = athing['TOF1']-athing['TOF3']/2
    log_good = [np.ones(len(athing)).astype(bool)]

    if apply_checksum:
        log_good.append(checksum(athing,check_max = 2))
    if filt_triples:
        for stuff in athing:
            if 'validtof' in stuff.lower():

                # print(stuff)
                log_good.append(athing[stuff].values.astype(bool))
    if min_tof:
        for stuff in use_filt:
            log_good.append(athing[stuff]>min_tof)
    
    
    if type(tof3_picker)== str and tof3_picker.lower() == 'auto':
        bb = int(np.sum(np.logical_and.reduce(log_good))/5)
        h,bins = np.histogram(athing.loc[np.logical_and.reduce(log_good)]['TOF3'],
                              (bb if bb > 2 else 5))
        bm = (bins[1:]+bins[:-1])/2
        p = bm[np.argmax(h)]
#         print(p)
        log_good.append(athing['TOF3']>p-1.5)
        log_good.append(athing['TOF3']<p+1.5)
    elif type(tof3_picker) == int:
        p = tof3_picker*4
        log_good.append(athing['TOF3']>p-1.5)
        log_good.append(athing['TOF3']<p+1.5)
    return(athing.loc[np.logical_and.reduce(log_good)])

#plot the time of flights of the files
from scipy.ndimage import gaussian_filter as gf

def plot_tofs(dats,hist_plt = ['tof0_sh','tof1_sh','TOF2','TOF3'],
                        bins = {
                           'tof0_sh': np.linspace(10,350,150),
                           'tof1_sh':np.linspace(10,150,75),
                           'TOF2':np.linspace(10,150,75),
                           'TOF3':np.linspace(0,15,50),
                            'TOF0': np.linspace(10,350,150),
                           'TOF1':np.linspace(10,150,75),  
                                },
                        norm = None,leg_lab = '',info = False):
    
    
    fig,axs = plt.subplots(np.ceil(len(hist_plt)/2).astype(int),2,sharey = False)
    fig.set_size_inches(9,4*len(axs.flatten())/2)
    
    for lab,thing in dats.items():
        if info:
            # slabel = '%4s: \n     (%6s,%6s)'%(str(lab),'Mean','Peak')
            slabel = '%4s: \n%12s(%6s)'%(str(lab),' ','Peak')
        else: 
            slabel = str(lab)

        if 'TOF0' in thing:
            labs = {}
            if info:
                for nam in hist_plt:
                    cent = np.nanmean(thing[nam])
                    h,bino = np.histogram(thing[nam],bins = bins[nam])[:2]
                    mid = (bino[:-1]+bino[1:])/2
                    # print('%.5s,%.5s'%(lab,nam))
                    # print(np.max(h)/np.mean(np.diff(bino)))
    #                 print(np.max(h))
                    
                    peak = mid[gf(np.argmax(h),2)]
    #                 ax.axvline(peak)
                    # slabel=slabel+'\n%8s:(%4.2f,%4.2f)'%(str(nam),cent,peak)
                    slabel=slabel+'\n%8s:(%4.2f)'%(str(nam),peak)
                
            for nam,ax in zip(hist_plt,axs.reshape(-1,1).flatten()):
                cent = np.nanmean(thing[nam])
                h,bino = np.histogram(thing[nam],bins = bins[nam])[:2]
                mid = (bino[:-1]+bino[1:])/2
                peak = mid[gf(np.argmax(h),2)]
#                 ax.axvline(peak)
#                 slabel=slabel+'\n%8s:(%4.2f,%4.2f)'%(str(nam),cent,peak)
                
                if norm == 'max':
                    ax.plot(mid,h/np.nanmax(h),alpha = .4,
                        label = slabel,)
                else:
                    ax.hist(thing[nam],bins = bins[nam],density = True,alpha = .2,
                        label = slabel,histtype = 'stepfilled')
                    ax.hist(thing[nam],bins = bins[nam],density = True,alpha = .8,
                            histtype = 'step',color = 'k')
                ax.set_xlabel('%s [nS]'%nam)
                # ax.semilogy()
#                 ax.semilogx()

    axs.flatten()[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left',title = leg_lab)
    fig.tight_layout()

    fig.subplots_adjust(hspace = .2,top = .925,left = .12)  
    return(fig,axs)

def plot_tofs_2d(thing,pltx,plty,binnum = 75):
    bins = {
       'tof0_sh': np.linspace(10,350,binnum),
       'tof1_sh':np.linspace(10,250,binnum),
       'TOF2':np.linspace(10,150,binnum),
       'TOF3':np.linspace(0,15,binnum),
        'TOF0': np.linspace(10,350,binnum),
       'TOF1':np.linspace(10,250,binnum),  
    }
    fig,ax = plt.subplots()

    x = (bins[pltx][1:]+bins[pltx][:-1])/2
    y = (bins[plty][1:]+bins[plty][:-1])/2
    cnts = np.histogram2d(thing[pltx],thing[plty],
                          bins = [bins[pltx],bins[plty]],density = True)[0].T
    ax.pcolormesh(x,y,np.log(cnts))
    ax.set_xlabel(pltx)
    ax.set_ylabel(plty)
    return(fig,ax)
    
# Find the tof file if it exists
def getListOfFiles(dirName):
    import os
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

def dat_loc(fil,home):
    import os
    f_indicator = fil.strip('.rec').split('_')[-2:]
    for f in getListOfFiles(home):
        # if fil in f and '.rec' not in f:
        #     return(f)
        if all(find in f for find in f_indicator) and '.rec' not in f:
            return(f)
# def dat_locations(fils,home = './'):
#     filz = []
#     for fil in fils:
#         floc = dat_loc(str(fil).strip('.rec'),home = home)
#         if floc:
#             filz.append(floc)
#     return(filz)

def s_run_get_dat(s_run_loc,combine = True,
                  ref_nam = 'file_name',load_params = {},home = './'):

#     dats = {}
#     for fil,ref_nam in zip(s_run_loc[fil_col].values,s_run_loc[ref_nam].values):
#         floc = dat_loc(str(fil).strip('.rec'),home = directory)
#         if floc:
#             dats[str(ref_nam)] = load_dt(floc,**load_params)
#     return(dats)
    dats = {}
    for rn in s_run_loc[ref_nam].values:
        dats[str(rn)] = []
        
    for fil,rn in zip(s_run_loc['file_name'].values,s_run_loc[ref_nam].values):
        floc = dat_loc(str(fil).strip('.rec'),home = home)
        if floc:
            dats[str(rn)].append(load_dt(floc,**load_params))
    if combine:
        for lab,vals in dats.items():
            if vals:
                dats[lab] = pd.concat(vals,ignore_index = True)
            else:pass
    return(dats)
    
def s_run_plot(s_run_loc,overplot = True,ref_nam = 'file_name',
               hist_bins = 'auto',
               auto_params = {'binw':1,'buffer':.3},
               load_params = {},
               plot_params = {'hist_plt':['TOF2','TOF0','TOF1','TOF3']},
               home = './'):
    dats = s_run_get_dat(s_run_loc,ref_nam = ref_nam,
                         load_params = load_params,home = home)
#     dats = {}
#     for fil,ref_nam in zip(s_run_loc[fil_col].values,s_run_loc[ref_nam].values):
#         floc = dat_loc(str(fil).strip('.rec'),home = directory)
#         if floc:
#             dats[str(ref_nam)] = load_dt(floc,**load_params)
    
    if dats:
        if hist_bins == 'auto':
            from .tof_tools import tof_expected
            tofs_ideal = tof_expected(np.unique(s_run_loc['ke'].values),
                                      np.unique(s_run_loc['species'].str.replace('+','')))
            bins = {}
            for val in plot_params['hist_plt']:
                vt = val.strip('_sh').lower()
                bin_start = np.min(tofs_ideal[vt].values*(1-auto_params['buffer']))
                if bin_start <0:
                    bin_start = 0
                bin_stop = np.max(tofs_ideal[vt].values*(1+auto_params['buffer']))
                if bin_start == bin_stop:
                    bin_start = 0
                    bin_stop = 100
                bins[val] = np.linspace(bin_start,bin_stop,
                                        int((bin_stop-bin_start)/auto_params['binw']) ).flatten()
                # print(bins[val].shape)
                bins['TOF3'] = np.linspace(0,15,40)
            plot_params['bins'] = bins

            fig,ax = plot_tofs(dats,**plot_params)
        return(fig,ax)
        
def s_run_rates(s_run_loc,overplot = True,ref_nam = 'file_name',load_params = {},rate = 'RateGold'):
    dats = s_run_get_dat(s_run_loc,ref_nam = ref_nam)
    if dats:
        plt.subplots()
        for lab,d in dats.items():
#             plt.plot(d['RateSilver'],label = lab+': Rsilver')
            plt.plot(d['Time'],d[rate],'.',label = lab+': Rgold')
        plt.legend()
#     return(dats)
#         fig.suptitle(fil_col)
#         for a in ax.flatten():


def import_srun(srun_loc):
    s_run = pd.read_excel(srun_loc,
                        header = 5,usecols = range(1,50))
    s_run.dropna(subset = ['file_name'],inplace = True)
    return(s_run)