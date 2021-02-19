import numpy as np
import pandas as pd
from .tools import *
from . import hist as hi
from . import TOF as tf

def lazy_histograms(hist_location,tof_location,estep,tof2_dat,
                            fil_goodtimes = [],nogt = [],apply_nep = False,
                            include_partials = True):
    # print(type(estep))

    good_times = []
    for gtf in fil_goodtimes:
        good_times.append(import_good_times(gtf))

    #import tof_data
    ftype = "*e%d.txt"%(estep if type(estep)==int else estep[0])
    import glob
    for head_loc in glob.glob(tof_location + ftype):
        try:
            tof_head = get_headder(head_loc)
            break
        except(UnboundLocalError):
            pass

    # i = 0
    # h_indexes = {}
    # for thing in tof_head:
    #     h_indexes[thing]=i
    #     i+=1
    # print(tof_head)
    # use_cols = [h_indexes[thing] for thing in ['time','count','phase','type','tof0', 'tof1', 'tof2', 'tof3']]
    df_tof = load_df(tof_location,estep,head = 'auto', calc_nep = True)
    #import hist data
    # hist_head = get_headder(glob.glob(hist_location + ftype)[0])
    # print(hist_head)
    # use_cols = [0,3,5,7]
    df_hist = load_df(hist_location,estep,head='auto',calc_nep = True)

    gt_hist = []
    gt_tof = []
    for goodt,no_gt in zip(good_times,nogt):
        gt_tof.append(mask_good_times(df_tof,goodt,apply_nep = apply_nep,include_no_gt = no_gt,
                                      return_mask=True,nep_start_max = 10,nep_stop_min = 20))
        gt_hist.append(mask_good_times(df_hist,goodt,apply_nep = apply_nep,include_no_gt = no_gt,
                                       return_mask=True,nep_start_max = 10,nep_stop_min = 20))

   
    #square the histogram bins and apply goodtime filter
    # mat_df = hi.group_up(df_hist.loc[np.logical_and.reduce(gt_hist)])
    mat_df = hi.square_split(df_hist.loc[np.logical_and.reduce(gt_hist)])
    # mat_df.df[]

    #define the species in the tof files
    df_gt = tf.lut_species(df_tof.loc[np.logical_and.reduce(gt_tof)])

    # bin the tof species using the hist bins
    t_bins,p_bins = mat_df.get_bins()['count']
    p_vals = mat_df.get_axes()['count'][1]
    for spec in np.unique(df_gt['species']):
        loc = df_gt['species']==spec
        nam = 'tof%s'%spec
        
        hist = np.histogram2d(df_gt['time'].loc[loc],df_gt['phase'].loc[loc],
                                   bins = [t_bins,p_bins],weights = df_gt['count'].loc[loc])[0]
        for p,h in zip(p_vals,hist.T):
            mat_df.df[(nam,p)] = h

    # compile df of matrices
    if include_partials:
        return(mat_df.accum_bins(8,reduce_f = ({
                                                'time':np.nanmean,
                                                 'start_time':np.nanmin,
                                                 'end_time':np.nanmax,
                                                 'dt':np.nansum
                                                })))
    else:
        mat_out = mat_df.accum_bins(8,reduce_f = ({'time':np.mean,
                                                'start_time':np.min,
                                                'end_time':np.max,
                                                'dt':np.sum
                                                }))
        mat_out.df.dropna(inplace = True)
        return(mat_out)


def lazy_histograms2(hist_location,tof_location,estep,tof2_dat,
                        fil_goodtimes = [],nogt = [],apply_nep = False,
                        include_partials = True,include_DE = True):
    good_times = []
    for gtf in fil_goodtimes:
        good_times.append(import_good_times(gtf))

    #import tof_data
    ftype = "*e%d.txt"%(estep if type(estep)==int else estep[0])
    import glob
    for head_loc in glob.glob(tof_location + ftype):
        try:
            tof_head = get_headder(head_loc)
            break
        except(UnboundLocalError):
            pass

    df_hist = load_df(hist_location,estep,head='auto',calc_nep = True)

    mat_df = hi.square_up3(df_hist)


    if include_DE:
        df_tof = load_df(tof_location,estep,head = 'auto', calc_nep = True)
        #define the species in the tof files
        df_gt = tf.lut_species(df_tof)

        # bin the tof species using the hist bins
        t_bins,p_bins = mat_df.get_bins()['count']
        p_vals = mat_df.get_axes()['count'][1]
        for spec in np.unique(df_gt['species']):
            loc = df_gt['species']==spec
            nam = 'tof%s'%spec
            
            hist = np.histogram2d(df_gt['time'].loc[loc],df_gt['phase'].loc[loc],
                                       bins = [t_bins,p_bins],weights = df_gt['count'].loc[loc])[0]
            for p,h in zip(p_vals,hist.T):
                mat_df.df[(nam,p)] = h


    def accum_filter(df,gt_dat,include_no_gt,e2_dat):
        filters = []
        for gt,no_gt in zip(gt_dat,include_no_gt):
            filters.append(mask_good_times(df,gt,
                                          apply_nep = False,include_no_gt = no_gt,
                                          return_mask=True,nep_start_max = 10,nep_stop_min = 20))
        filters.append(tof_2_filt(df,e2_dat))
        return(np.logical_and.reduce(filters))

    def median_filt(vals = [], buffer = .05):
        filters = []
        for v in vals:
            med = np.nanmedian(v)
            filt = np.logical_and(v>med*(1-buffer),v<med*(1+buffer))
            filters.append(filt)
            print(np.sum(filt)/len(filt))
        return(np.logical_and.reduce(filters))
    


    t_dat = mat_df.mask(~accum_filter(mat_df,good_times,[True,False],tof2_dat),
                                        inplace = False).accum_bins(8,
                                        reduce_f ={
                                                    't_mean':np.mean,
                                                    'time':np.mean,
                                                     'start_time':np.min,
                                                     'end_time':np.max,
                                                     'dt':np.sum
                                                      })
    return(t_dat.mask(~median_filt(vals = [t_dat['end_time']-t_dat['start_time'],
                            abs(1376-t_dat['dt'])],buffer = .05),inplace = False))

def tof_mat_link(mat_df,df_gt,binby = ''):
    # bin the tof species using the hist bins
    t_bins,p_bins = mat_df.get_bins()[binby]
    p_vals = mat_df.get_axes()[binby][1]
    for spec in np.unique(df_gt['species']):
        loc = df_gt['species']==spec
        nam = 'tof%s'%spec
        
        hist = np.histogram2d(df_gt['time'].loc[loc],df_gt['phase'].loc[loc],
                                   bins = [t_bins,p_bins],weights = df_gt['count'].loc[loc])[0]
        for p,h in zip(p_vals,hist.T):
            mat_df.df[(nam,p)] = h
    return(mat_df)

def extra_lazy_histograms(hist_pickel,tof_pickel,estep,fil_goodtimes):
    # print(type(estep))
    if apply_goodtimes==True:
        good_times = list(import_good_times(f_gt) for fgt in fil_goodtimes)

    df_tof = pd.read_pickle(tof_pickel)
    mad_df = pd.read_pickle(hist_pickel)

    for gt in good_times:
        df_tof = df_tof.loc()
    import hist_processing as hi
    #square the histogram bins and apply goodtime filter
    mat_df = hi.group_up((mask_good_times(df_hist,good_times,apply_nep = True) if apply_goodtimes else df_hist))

    #define the species in the tof files
    import tof_processing as tf
    df_gt = tf.lut_species((mask_good_times(df_tof,good_times,apply_nep = True) if apply_goodtimes else df_tof))

    # bin the tof species using the hist bins
    t_bins,p_bins = mat_df.get_bins()['hist']
    p_vals = mat_df.get_axes()['hist'][1]
    for spec in np.unique(df_gt['species']):
        loc = df_gt['species']==spec
        nam = 'tof%s'%spec
        
        hist = np.histogram2d(df_gt['time'].loc[loc],df_gt['phase'].loc[loc],
                                   bins = [t_bins,p_bins],weights = df_gt['count'].loc[loc])[0]
        for p,h in zip(p_vals,hist.T):
            mat_df.df[(nam,p)] = h
    mat_df.mask(~tof2_filt(mat_df,tof2_dat),inplace = True)
    # compile df of matrices
    return(mat_df.accum_bins(8,reduce_f = {'time':np.nanmean,
                                                             'start_time':np.nanmin,
                                                             'end_time':np.nanmax,
                                                             'dt':np.nansum}))


def hist_to_csv(mat_df,estep,output_location = ''):
    from datetime import datetime,timedelta
    import os

    # gb = mat_df.df.groupby(('eph','orbit'))
    i = 0
    big_head = ['bin_center(NEP)', 'center_time(GPS)', 
                'start_time(YYYY-MM-DD DD:HH:MM:SS)', 
                'HB_counts',   'bin_center(NEP)', 'DE_counts','']
    nep = phase_to_nep(mat_df.get_axes()['count'][1]).astype(int)
    t_split0 = '-------------------------------------------------------------------------'

    mat_df[('eph','year')] =  [(datetime(1980, 1, 6)+timedelta(seconds = s)).year for s in mat_df['time'].values]
    for year in [x for _, x in mat_df.df.groupby(('eph','year'))]:
        
        yy = year['eph']['year'].values[0]
        
        year_direct = output_location+'%d\\'%int(yy)
        if not os.path.exists(year_direct):
            os.makedirs(year_direct)

        for orb in [x for _, x in year.groupby(('eph','orbit'))]:
            
            orb_num = orb['eph']['orbit'].values[0]
            out_direct =year_direct+'%d\\'%int(orb_num)
            if not os.path.exists(out_direct):
                os.makedirs(out_direct)

            nam = 'IBEX_lo_o%04d%s_E%d_HB_and_DE_report.csv'%(int(orb_num),('a' if orb_num-int(orb_num)==0 else 'b'),estep)
            # print(nam)
            cols = [['bin_center(NEP)'],['center_time(GPS)'],['center_time(YYYY-MM-DD DD:HH:MM:SS)'],
                                ['HB_counts'],['bin_center(NEP)'],['DE_counts'],['Orbit:%.1f'%orb_num]]
            for start_t,end_t,dt_t,t_t,hist,tof in zip(orb['eph']['start_time'].values,
                                                      orb['eph']['end_time'].values,
                                                      orb['eph']['dt'].values,
                                                      orb['eph']['time'].values,
                                                      orb['count'].values,
                                                      orb['tofH'].values):
                sub_cols = [['']]*(len(big_head)-3)
                sub_cols[0] = [t_split0]
                sub_cols+=[['start_time:      %f'%start_t],
                                ['end_time:      %f'%end_t],
                                ['total_time(secs):            %f'%dt_t]]
                ep = ['']*len(hist)
                spin_time = str(datetime(1980, 1, 6)+timedelta(seconds = t_t))
                mtime = ['']*len(hist)
                mtime[0] = t_t
                omat = []
                for col,lo,lin in zip(cols,sub_cols,[list(nep),mtime,[spin_time]+ep[1:],list(hist),list(nep),list(tof),ep]):
                    col.append(lo[0])
                    col+=lin
            pd.DataFrame(cols).T.to_csv(out_direct+nam,index = False,header = False)


def generate_isn_data(hist_location,tof_location,e2_hist_location,lazy_params = {},
                        output_location = '',estep = range(1,9),
                        goodtime_steps = [1,2,3]):
    e2_dat = load_df(e2_hist_location,2,head = 'auto',calc_nep = False)
    for step in estep:
        hist_to_csv(lazy_histograms2(hist_location,tof_location,
                                    step,e2_dat,**lazy_params),
                    step,output_location)


def tof_to_hist(mat_df,df_gt,species = None):
    t_bins,p_bins = mat_df.get_bins()['hist']
    p_vals = mat_df.get_axes()['hist'][1]
    if 'species' in df_gt:
        for spec in (np.unique(df_gt['species']) if species == None else species):
            loc = df_gt['species']==spec
            nam = 'tof%s'%spec
            hist = np.histogram2d(df_gt['time'].loc[loc],df_gt['phase'].loc[loc],
                                       bins = [t_bins,p_bins],weights = df_gt['count'].loc[loc])[0]
            for p,h in zip(p_vals,hist.T):
                mat_df.df[(nam,p)] = h
    else:
        nam = 'tof'
        hist = np.histogram2d(df_gt['time'],df_gt['phase'],
                                       bins = [t_bins,p_bins],weights = df_gt['count'])[0]
        for p,h in zip(p_vals,hist.T):
            mat_df.df[(nam,p)] = h
    return(mat_df)


def txt_to_df(loc, output_location = '',fname = '',ftype = '.txt',to_hist = False,replace = False):
    # loc = r'C:\Users\Jonny Woof\Box Sync\IBEX Data\Data\trial\SOC_txt\hist_dat'
    import os
    step = []
    for root,lcc,fil in os.walk(loc):
        for stuff in fil:
            step.append(stuff.split('_')[-1].strip(ftype))
            # ftype.append(stuff.split('_')[-1].split('.')[-1])
    # print(step)
    for en in np.unique(np.array(step)):
        new_fil = output_location+fname+en+'.pkl'
        if not os.path.exists(new_fil) or replace == True:
            # temp_df = []
            if to_hist == False:
                temp_df = load_df(loc,en,ftype,calc_nep = False,head = 'auto')
            else:
                from hist_processing import group_up
                thing = load_df(loc,en,ftype,calc_nep = False,head = 'auto')
                if thing.empty == False:
                    temp_df = group_up(thing).df
                else:temp_df = thing
            # if temp_df:
            temp_df.to_pickle(new_fil)
        else:
            print('%s already exists, choose new filename or set replace = True'%new_fil)


def tof_2_filt(df,e2_dat,percentile = .15,inplace = True):
    if 'TOF2_E2' not in df['eph']:
        #load tof2 dat
        # e2_dat = load_df(tof2_loc,2,head = 'auto',calc_nep = False)
        # Use only counts from sector 1
        sect_1 = .17
        sects = np.logical_and.reduce([e2_dat['phase']<.5,abs(e2_dat['phase']-sect_1)>sect_1/2])
        # incorperate tof2 rate data to the data set
        df.add(e2_dat.loc[sects]['time'].values,e2_dat.loc[sects]['count'].values,
                            statistic = 'max',label = 'TOF2_E2')

    # from scipy.ndimage import gaussian_filter as gf
    vals = df['TOF2_E2'].values
    #smooth tof2 data to use for the filter
    sig = np.nanmin(np.stack([gauss_filt_nan(vals,sigma= 1),
                           gauss_filt_nan(vals,sigma= 2)]),
                    axis = 0)

    # Use not outlying data to calculate orbit baseed threshold 
    sig_filt = np.logical_and.reduce([sig<300,sig>10])
    def orb_thresh(dat):
        l = len(dat)
        th1 = np.sort(dat['TOF2_E2'].values)[int(l*percentile)]*1.6
        return(th1 if th1<84 else 84)

    th_o = df['eph'].loc[sig_filt].groupby('orbit').agg(orb_thresh)['TOF2_E2'][df['orbit']].values
    th_o[np.isnan(th_o)] = 84
    t_filt = np.logical_and.reduce([vals>8,sig<th_o])

    return(t_filt)



def mat_df_test(mat_df,labels = None):
    spin_time_av = 14.5

    ref_Vals = ['avg_spin: 14.5 sec']

    # things = {
    # 'stop_start':mat_df['end_time']-mat_df['start_time'],
    # 'mid_start':mat_df['time']-mat_df['start_time'],
    # 'stop_mid':mat_df['end_time']-mat_df['time'],
    # }
    # print('different dt:')
    # print((np.unique(mat_df['dt'][~np.isnan(mat_df['dt'])])/14.4/8))
    # for lab,val in things.items():
    #     print(lab)
    #     print('  min:%d'%np.nanmin(val))
    #     print('  max:%d'%np.nanmax(val))

    # print(np.nansum(mat_df['count']))
    mat_hist = mat_df
    from matplotlib import pyplot as plt
    fig,axs = plt.subplots(10)
    fig.set_size_inches(10,5*len(axs))
    txt1 = {}
    txt2 = {}
    if labels ==None:
        labels = ['']*(len(mat_df)if type(mat_df)==list else 1 )

    for mat_hist,labs in zip((mat_df if type(mat_df)==list else [mat_df]),labels):

        test_params = {
                        'end_time - start_time':mat_hist['end_time']-mat_hist['start_time'],
                        'diff(t_mean)': np.diff(mat_hist['t_mean']),
                        'diff(t)': np.diff(mat_hist['time']),
                        'time-start_time':mat_hist['time'].values-mat_hist['start_time'],
                        'time-end_time':mat_hist['time'].values-mat_hist['end_time'],
                        't_mean-start_time':mat_hist['t_mean'].values-mat_hist['start_time'],
                        't_mean-end_time':mat_hist['t_mean'].values-mat_hist['end_time'],
                        'dt':mat_hist['dt'][~np.isnan(mat_hist['dt'])],
                        'time-t_mean':mat_hist['time']-mat_hist['t_mean'],
                        'dt_mid':abs(1376-mat_hist['dt'])
                      }

        loc_ref = 0
        for lab,val,ax in zip(test_params.keys(),test_params.values(),axs):
            
            if lab not in txt1:
                txt1[lab] = ['discreet vals:']
                txt2[lab] = []
                
            # txt[lab].append(str(lab))
            # txt1[lab].append('different vals:')
            unq = np.unique(val[~np.isnan(val)])
            txt1[lab].append(str(len(unq)))
            txt1[lab].append(np.array2string(unq,max_line_width = 30,
                                       precision= 2,separator = ',',threshold = 10))
            
            # tx[lab].append('Fraction out of med:')

            txt2[lab].append('[%s] min:%f'%(labs,np.nanmin(val)))
            txt2[lab].append('max:%f'%np.nanmax(val))
            med = np.nanmedian(val[abs(val)>0])
            ax.axvline(med)
            txt2[lab].append('med:%f'%med)
            txt2[lab].append('med_frac:%f'%(np.sum(np.logical_and(val<med*1.01,val>med*.99))/len(val[~np.isnan(val)])))
            
            # print(txt)

            ax.hist(val,50,alpha = .2,density = False,label = labs)
            ax.set_title(lab)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            for txt,loc,off,align in zip(
                                         [txt1,txt2],
                                         [(0, 1),(1, 1)],
                                         [(12, -2),(-12, -2)],
                                         ['left','right']
                                         ):
                annot = ''
                for s in txt[lab]: 
                    annot+='\n %s'%s
                ax.annotate(annot, xy=loc, 
                            xycoords='axes fraction',
                            textcoords='offset points',
                            xytext=off, va='top',
                            ha = align 
                        )
            ax.semilogy()
    fig.tight_layout()
    # ax[1].hist(np.diff(mat_hist['t_mean']),50,alpha = .2)
    # ax[1].set_title('diff(t_mean)')

    # for thing in ['start_time','end_time']:
    #     ax[2].hist((mat_hist['time'].values-mat_hist[thing]),50,alpha = .2)
    # # ax[2].set_title()

    # ax[3].hist(mat_hist['dt'][~np.isnan(mat_hist['dt'])],20)

    # ax[4].hist(mat_hist['time']-mat_hist['t_mean'],50)
    # dt = np.nanmedian(np.diff(mat_hist['time']))
    # for l in [dt/2,-dt/2]:
    #     ax[4].axvline(l)
    fig.tight_layout()
    # for a in ax:
    #     a.semilogy()
