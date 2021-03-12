import numpy as np
import pandas as pd
from .tools import *
from . import hist as hi
from . import TOF as tf


def generate_isn_data(hist_location,tof_location,e2_hist_location,lazy_params = {},
                        output_location = '',estep = range(1,9),
                        goodtime_steps = [1,2,3]):

    e2_dat = load_df(e2_hist_location,2,head = 'auto',calc_nep = False)
    for step in estep:
        hist_to_csv(lazy_histograms(hist_location,tof_location,
                                    step,e2_dat,**lazy_params),
                    step,output_location)


def unified_histograms(hist_location,
                       # tof_location,estep,tof2_dat,
                        apply_nep = False,
                        orbit_range = [0,np.inf],
                        steps = [2],
                        bin_args = {}):
    
    all_times = load_df(hist_location,
                            [1,2,3,4,5,6,7,8,9,'A','B','C'],
                            orbit_range = orbit_range,
                            head = 'auto',
                            calc_nep = False,
                            True_square = False,
                            # usecols=[0,7],
                            usecols=[0,1,2,7],
                            # usecols=[0,1,2,7,8,9,10],
                            use_steps = 'unique_times',
                            accum_duplicates = 'drop')
    # return(all_times)
    mat_df = hi.square_spinterp2(all_times.sort_values('time'),
                                true_square = False,
                                nan_edge = 'interp',
                                time_binner = 'start',
                                    extra_bin = ['R.A.','Decl'],
                                    extra_labels = ['R.A.','Decl'],
                                    extra_actions=[None,None])

    # bin_edges = np.linspace(0,1,int(360/bin_width)+1)#-3/360
    # binm =(bin_edges[1:]+bin_edges[:-1])/2
    # bin_edges = mat_df.get_bins()['phase'][1]
    # binm = mat_df.get_axes()['phase'][1]


    for e in steps:
        h_raw = load_df(hist_location,e,
                                    orbit_range = orbit_range,
                                    head = 'auto',
                                    calc_nep = True,
                                    True_square =False,
                                    usecols=[0,5,7,8,9,10],
                                    # w = 60,
                                    use_steps = 'all',
                                    accum_duplicates = 'integrate')
        # will only work for true square bins
        if not h_raw.empty:
            binm = np.sort(np.unique(h_raw['phase']))
            bin_edges = np.append(binm-np.nanmedian(np.diff(binm))/2,
                                  binm[-1]+np.nanmedian(np.diff(binm))/2)
            
            square_estep = hi.multi_bin(
                                h_raw,
                                bin_edges = bin_edges,
                                labels = ['time','dt','loc-X-RE','loc-Y-RE','loc-Z-RE','count'],
                                actions= [np.nanmin,np.nanmedian,np.nanmean,np.nanmean,np.nanmean,None],
                                accum_labels = ['t_mean','dt','loc-X-RE','loc-Y-RE','loc-Z-RE','count'],
                                true_square = False,
                                )

            for lab2 in ['loc-X-RE','loc-Y-RE','loc-Z-RE']: 
                mat_df.add(square_estep['t_mean'].flatten(),
                            square_estep[lab2].flatten(),label = lab2,statistic = 'mean',combine = True)

            e_lab = 'E%s'%str(e).lower()
            mat_df.add(square_estep['t_mean'].flatten(),
                        square_estep['dt'],label = 'dt',
                                    ax = [e_lab],dat_type = 'supp',statistic = 'mean')

            mat_df.add(square_estep['t_mean'].flatten(),
                square_estep['t_mean'],label = 't_groups',
                            ax = [e_lab],dat_type = 'supp')
            
            mat_df.add(square_estep['t_mean'].flatten(),
                            square_estep['count'],
                            label = e_lab,ax = binm,statistic = 'mean')

            nan_rep = np.logical_and(np.isnan(mat_df.df[('eph','t_mean')].values),
                                     ~np.isnan(mat_df.df[('t_groups',e_lab)].values))

            mat_df.df[('eph','t_mean')][nan_rep] = mat_df[('t_groups',e_lab)][nan_rep]
            mat_df.df[('eph','time')][nan_rep] = mat_df[('t_groups',e_lab)][nan_rep]
            
            # mat_df[('eph','t_mean')].loc[]
            print(np.sum(h_raw['count']))
            print(np.nansum(mat_df[e_lab])/np.sum(h_raw['count']))
            t_counts = np.nansum(mat_df[e_lab].dropna(),axis = 1)
            print(np.sum(~np.isnan(t_counts)))
            print(abs(np.nansum(mat_df[e_lab])-np.sum(h_raw['count']))/np.nanmedian(t_counts))
    
    mat_df.df.insert(1,('eph','start_time'), mat_df.df[('eph','t_mean')])

    end = np.zeros(mat_df['t_mean'].shape)*np.nan
    end[:-1] = mat_df['start_time'][1:]
    end[np.isnan(end)] = mat_df['start_time'].loc[np.isnan(end)]+\
                                    np.append(np.diff(mat_df['time']),
                                               np.median(np.diff(mat_df['time'])))[np.isnan(end)]
    mat_df.df.insert(2,('eph','end_time'),end)
    mat_df.drop_mat('phase')
    mat_df.drop_mat('t_groups')
    # mat_df.df.drop(('eph','t_mean'),inplace = True)
    # eph_dat['end_time'] = np.zeros(len(eph_dat['t_mean'])).reshape(-1,1)*np.nan
    # eph_dat['end_time'][:-1] = eph_dat['start_time'][1:]
    # end_na = np.isnan(eph_dat['end_time'])
    # eph_dat['end_time'][end_na] = eph_dat['start_time'][end_na]+dif_interp[end_na]
        
    return(mat_df)


def add_filter_de(mat_df,
                        good_times= [],
                        tof_location = '',
                        nogt_orbs = [],
                        apply_nep = False,
                        include_partials = True,
                        include_DE = False,):
    # apply different goodtime filters inputting nan values in place
    if include_DE:
        # load DE data and implement into historam df
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
    # else: 
    #     p_vals = mat_df.get_axes()['count'][1]
    #     hist = -np.ones(mat_df['count'].shape)
    #     for p,h in zip(p_vals,hist.T):
    #         mat_df.df[('tofH',p)] = h

    g_times = {fil.strip('.txt'): import_good_times(fil) for fil in good_times}

    # mat_df.add(~accum_filter(mat_df,good_times,[True,False],tof2_dat),
    #                                         inplace = True)
    for lab,filt,nogt in zip(g_times.keys(),
                                g_times.values(),
                                            nogt_orbs):
        mat_df.df[('eph',lab)] = mask_good_times(mat_df['eph'],filt,
                                                apply_nep = False,
                                                include_no_gt = nogt,
                                                return_mask=True,
                                                nep_start_max = 10,
                                                nep_stop_min = 20).astype(int)

    return(mat_df)

def gen_unified_hist_dat(
                         data_location,
                            output_location,
                            goodtime_files = [],
                            steps = [1,2,3,4,5,6,7,'A','B','C'],
                            orbit_range = [0,np.inf],
                            ):

    # turn the textfiles into a data base
    mat_df = unified_histograms(dat_location,
                                orbit_range = orbit_range,
                                bin_args = {
                                    'true_square':False,
                                 'nan_edge':'interp',
                                 'time_binner':'start'},
                                steps = steps)
    # Add the boolean filters to the data frame
    mat_df = pb.ISN_data.add_filter_de(mat_df,
                                       good_times = goodtime_files,
                                            nogt_orbs = [True,False])
    # Add a fractional year value to the data frame
    mat_df.df.insert(0,('eph','Year'),pb.tools.gps_to_datetime(mat_df['time']))
    # Print the the data frame to csv
    mat_df.mat_print(output_location,
             splitby = 'Year',reduce_func = np.floor,print_type = int,split_folders = True)

def time_filters(df,gt_dat,include_no_gt):
    filters = []
    for gt,no_gt in zip(gt_dat,include_no_gt):
        filters.append(mask_good_times(df,gt,
                                    apply_nep = False,
                                    include_no_gt = no_gt,
                                    return_mask=True,
                                    nep_start_max = 10,
                                    nep_stop_min = 20))
    # filters.append(tof_2_filt(df,e2_dat))
    return(np.logical_and.reduce(filters))



def lazy_histograms(hist_location,tof_location,estep,tof2_dat,
                        fil_goodtimes = [],nogt = [],apply_nep = False,
                        bin_function = 'square_spinterp',
                        include_partials = True,
                        include_DE = True,bin_args = {},load_args = {}):
    
    #============================================================
    # init funcs/params

    def accum_filter(df,gt_dat,include_no_gt,e2_dat):
        filters = []
        for gt,no_gt in zip(gt_dat,include_no_gt):
            filters.append(mask_good_times(df,gt,
                                          apply_nep = False,include_no_gt = no_gt,
                                          return_mask=True,nep_start_max = 10,nep_stop_min = 20))
        filters.append(tof_2_filt(df,e2_dat))
        return(np.logical_and.reduce(filters))

    def median_filt(vals = [], buffer = .05):
        #gets rid of outliers
        filters = []
        for v in vals:
            med = np.nanmedian(v)
            filt = np.logical_and(v>med*(1-buffer),v<med*(1+buffer))
            filters.append(filt)
            # print(np.sum(filt)/len(filt))
        return(np.logical_and.reduce(filters))

    def running_median_filt(vals = [], buffer = .05,window = 21):
        #gets rid of outliers
        filters = []
        for v in vals:
            med = pd.Series(v).rolling(window = window,center = True).median()
            filt = np.logical_and(v>med*(1-buffer),v<med*(1+buffer))
            filters.append(filt)
            # print(np.sum(filt)/len(filt))
        return(np.logical_and.reduce(filters))


    bin_funcs = {
                    'square_up': hi.square_up,
                    'square_manybin': hi.square_manybin,
                    'square_spinterp': hi.square_spinterp,
                    'square_groups': hi.square_groups,
                    'square_split': hi.square_split
                }
    #============================================================
    

    # load goodtimes
    good_times = []
    for gtf in fil_goodtimes:
        good_times.append(import_good_times(gtf))


    # load histogram data 
    df_hist = load_df(hist_location,estep,head='auto',calc_nep = True,**load_args)
    total_counts = np.sum(df_hist['count'])

    # square histogram data 
    mat_df = bin_funcs[bin_function](df_hist,**bin_args)


    if include_DE:
        # load DE data and implement into historam df
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
    else: 
        p_vals = mat_df.get_axes()['count'][1]
        hist = -np.ones(mat_df['count'].shape)
        for p,h in zip(p_vals,hist.T):
            mat_df.df[('tofH',p)] = h
    # apply different goodtime filters inputting nan values in place

    print('Data after binning:%f'%(np.nansum(mat_df['count'])/total_counts))
    mat_df.mask(~accum_filter(mat_df,good_times,[True,False],tof2_dat),
                                            inplace = True)

    print('Data after goodtimes:%f'%(np.nansum(mat_df['count'])/total_counts))
    if include_partials:
        # apply reduce function accounting for nan values
        return(mat_df.accum_bins(8,reduce_f = ({
                                                'time':np.nanmean,
                                                 'start_time':np.nanmin,
                                                 'end_time':np.nanmax,
                                                 'dt':np.nansum
                                                })))

    else:
        mat_df.mask(~median_filt(vals = [mat_df['end_time']-mat_df['start_time'],
                                np.diff(mat_df['t_mean'],prepend = np.nan)],buffer = .02),
                                inplace = True)
        
        print('Data after global median filter:%f'%(np.nansum(mat_df['count'])/total_counts))
        mat_df.mask(~running_median_filt(vals = [mat_df['end_time']-mat_df['start_time']],
                                         buffer = .0007))

        
        # apply reduce function make all partial bins nan
        t_dat = mat_df.accum_bins(8,reduce_f ={
                                                't_mean':np.mean,
                                                'time':np.mean,
                                                 'start_time':np.min,
                                                 'end_time':np.max,
                                                 'dt':np.sum
                                              },mat_reduce = np.sum)
        # drop nan values, as well as outliers associated with median filt
        t_dat.mask(~median_filt(vals = [t_dat['end_time']-t_dat['start_time'],
                                abs(1376-t_dat['dt'])],buffer = .05),inplace = True)
        t_dat.df.dropna(inplace = True) 
        print('Data after dropping empty bins:%f'%(np.nansum(t_dat['count'])/total_counts))
        return(t_dat)

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


def hist_to_csv(mat_df,estep,output_location = ''):
    # takes mat_df data struct and outputs csv file in ISN_data format
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
            cols = [['bin_center(NEP)'],['center_time(GPS)'],['center_time(YYYY-MM-DD DD:HH:MM:SS)'],
                        ['HB_counts'],['bin_center(NEP)'],['DE_counts'],['Orbit:%.1f'%orb_num]]

            for start_t,end_t,dt_t,t_t,hist,tof in zip(
                                                      orb['eph']['start_time'].values,
                                                      orb['eph']['end_time'].values,
                                                      orb['eph']['dt'].values,
                                                      orb['eph']['time'].values,
                                                      orb['count'].values,
                                                      orb['tofH'].values
                                                      ):
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
                for col,lo,lin in zip(cols,sub_cols,[list(nep),mtime,
                                    [spin_time]+ep[1:],list(hist),
                                            list(nep),
                                            list(tof),
                                            ep]):
                    col.append(lo[0])
                    col+=lin

            pd.DataFrame(cols).T.to_csv(out_direct+nam,index = False,header = False)


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
    import os
    step = []
    for root,lcc,fil in os.walk(loc):
        for stuff in fil:
            step.append(stuff.split('_')[-1].strip(ftype))
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
            temp_df.to_pickle(new_fil)
        else:
            print('%s already exists, choose new filename or set replace = True'%new_fil)


def tof_2_filt(df,e2_dat,percentile = .15,inplace = True):
    if 'TOF2_E2' not in df['eph']:
        # Use only counts from sector 1 and 3
        sect_1 = .67
        dsect = .17
        sects = np.logical_and.reduce([e2_dat['phase']>.45,abs(e2_dat['phase']-sect_1)>dsect/2])
        # incorperate tof2 rate data to the data set
        df.add(e2_dat.loc[sects]['time'].values,e2_dat.loc[sects]['count'].values,
                            # statistic = 'max',
                            label = 'TOF2_E2')

    vals = df['TOF2_E2'].values.flatten()
    #smooth tof2 data to use for the filter
    sig = np.nanmin(np.stack([gauss_filt_nan(vals,sigma= 1),
                           gauss_filt_nan(vals,sigma= 2)]),
                    axis = 0)

    # Use not outlying data to calculate orbit baseed threshold 
    sig_filt = np.logical_and.reduce([sig<300,sig>10])
    def orb_thresh(dat):
        dd = dat.values#[~np.isnan(dat.values)]
        l = len(dd)
        th1 = np.sort(dd)[int(l*percentile)]*1.6
        return(th1 if th1<84 else 84)
    #mask outliers
    df[('eph','TOF2_E2')].loc[~sig_filt] = np.nan
    th_o = df['eph'].groupby('orbit').agg({'TOF2_E2':orb_thresh}).loc[df['orbit'].values].values.flatten()
    th_o[np.isnan(th_o)] = 84
    t_filt = np.logical_and.reduce([vals>8,sig<th_o])

    return(t_filt)



def mat_df_test(mat_df,labels = None):
    # Plot and test the mat_df generated data

    mat_hist = mat_df
    from matplotlib import pyplot as plt
    fig,axs = plt.subplots(7)
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
                        # 'time-start_time':mat_hist['time'].values-mat_hist['start_time'],
                        # 'time-end_time':mat_hist['time'].values-mat_hist['end_time'],
                        't_mean-start_time':mat_hist['t_mean'].values-mat_hist['start_time'],
                        't_mean-end_time':mat_hist['t_mean'].values-mat_hist['end_time'],
                        # 'dt':mat_hist['dt'][~np.isnan(mat_hist['dt'])],
                        'time-t_mean':mat_hist['time']-mat_hist['t_mean'],
                        'start_end':mat_hist['start_time'].values[1:]-mat_hist['end_time'].values[:-1],
                        # 'dt_mid':abs(1376-mat_hist['dt'])
                        # 'spino (end_t-start_t)/dt':(mat_hist['end_time']-mat_hist['start_time'])/mat_hist['dt']
                      }

        loc_ref = 0
        for lab,val,ax in zip(test_params.keys(),test_params.values(),axs):
            
            if lab not in txt1:
                txt1[lab] = ['discreet vals:']
                txt2[lab] = []
                
            unq = np.unique(val[~np.isnan(val)])
            txt1[lab].append(str(len(unq)))
            txt1[lab].append(np.array2string(unq,max_line_width = 30,
                                       precision= 2,separator = ',',threshold = 10))

            txt2[lab].append('[%s] min:%f'%(labs,np.nanmin(val)))
            txt2[lab].append('max:%f'%np.nanmax(val))
            med = np.nanmedian(val[abs(val)>0])
            ax.axvline(med)
            txt2[lab].append('med:%f'%med)
            txt2[lab].append('med_frac:%f'%(np.sum(np.logical_and(val<med*1.01,val>med*.99))/len(val[~np.isnan(val)])))

            # bins = np.linspace(med-abs(.5*med),med+abs(.5*med),50)
            bins = 50
            ax.hist(val,bins,alpha = .2,density = False,label = labs)
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
