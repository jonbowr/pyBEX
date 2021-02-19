import numpy as np
import pandas as pd
from .mat_df import mat_df


def hdf(dat,dat_cols,mat,mat_cols,
                 major_lables = ['eph','hist']):
    h = pd.MultiIndex.from_arrays([major_lables[:-1]*len(dat_cols)+major_lables[1:]*len(mat_cols),
                                  dat_cols+list(mat_cols)],names = ['type','lab'])
    for d in [dat,mat]:print(d.shape)
    return(pd.DataFrame(np.concatenate([dat,mat],axis = 1),columns = h))

def accum_duplicate_times(df_hist,method = 2,dt_multiplier = 8):
    import time
    if method ==1:
        ts = time.time()
        steps = np.unique(df_hist['ch'])
        u_time,ind1,ind2 = np.intersect1d(df_hist['time'].loc[df_hist['ch']==steps[0]],
                                          df_hist['time'].loc[df_hist['ch']==steps[1]],
                                          return_indices = True)

        df_acc = df_hist.copy()
        counts = df_acc['count'].values
        counts[df_acc['ch'].values==steps[0]][ind1] += df_hist['count'].loc[df_hist['ch']==steps[1]].values[ind2]
        counts[df_acc['ch'].values==steps[1]][ind2] += df_hist['count'].loc[df_hist['ch']==steps[0]].values[ind1]
        
        df_acc['count'] = counts
        # df_acc['count'].replace(df_acc['ch'].values==steps[0],
        #                         df_hist['count'].loc[df_hist['ch']==steps[1]].values[ind2]*np.nan)
        # df_acc['count'].replace(df_acc['ch'].values==steps[1],
        #                         df_hist['count'].loc[df_hist['ch']==steps[0]].values[ind1]*np.nan)
        # t_accu = np.ones(len(counts))
        # t_accu[df_acc['ch'].values==steps[0]][ind1] += 1
        # t_accu[df_acc['ch'].values==steps[1]][ind2] += 1
        # df_acc['count'].replace(counts,inplace = True)
        df_acc.drop_duplicates('time',inplace = True)
        print(time.time()-ts)
        return(df_acc)
    # return(df_acc.drop(['count'],axis = 1))
    elif method == 2:
        df_acc = df_hist.copy()
        df_acc['dt'] = np.ones(df_acc.shape[0])
        df_acc[['count','dt']] = df_acc.groupby(['time'])['count','dt'].transform('sum')
        
        df_acc.drop_duplicates('time',inplace = True)

        dt_av = np.median(np.diff(df_acc['time']))
        # dt =np.min(abs(
        #         np.stack([
        #             np.diff(df_acc['time'],
        #                 prepend = df_acc['time'].values[0]-dt_av),
        #             np.diff(np.flip(df_acc['time']),
        #                 prepend = df_acc['time'].values[-1]+dt_av)])),
        #         axis = 0).flatten()
        df_acc['dt'] = (df_acc['dt']*dt_av*dt_multiplier)#.round(8)
        orb = df_acc['orbit'].values.astype(float)
        orb[df_acc['arc']=='b'] += .5
        df_acc['orbit'] = orb
        return(df_acc)


def square_time_info(tot_time,tot_dt,dt_scaler = 8):
    inf_out = {}
    for f,nam in zip([np.nanmean,np.nanmin,np.nanmax],['t_mean','start time','end time','dt']):
        inf_out.append(f(tot_time,axis = 1))
    inf_out.append(np.nansum(tot_dt,axis = 1)*dt_scaler)
    return(inf_info,inf_out)


def phase_round(phase_angle,bin_edges,binm):
    t_loc = np.digitize(phase_angle,bin_edges[:-1].flatten()).flatten()-1
    return(binm[t_loc])


def square_param_accum(df_hist,param_to_bin,bins,square_w,accum_params = [],accum_actions=[],accum_labels = []):
    otime,ind1,ind2 = np.intersect1d(param_to_bin,bins,return_indices = True)
    tot = {}
    for nam,act,label in zip(accum_params,accum_actions,accum_labels):
        thing = np.array([np.nan]*len(bins))
        thing[ind2] = df_hist[nam].values[ind1]
        sqr_thing = thing.reshape(-1,square_w)
        tot[label] = (act(sqr_thing,axis =1).reshape(-1,1) if act else sqr_thing.copy()) 
    return(tot)

def square_up(df_in,bin_width = 6,spin_sum = 1):
    df_hist = accum_duplicate_times(df_in)

    bin_edges = np.linspace(0,1,int(360/bin_width)+1)#-3/360
    binm =(bin_edges[1:]+bin_edges[:-1])/2

    # p1 = df_hist['phase'].values.copy()
    p1 = phase_round(df_hist['phase'].values,bin_edges,binm)
    # return(phase_round(p1,bin_edges))
    # p1_count = df_hist['count'].values.copy()

    # Round the phase angle to the local bin center
    # p1 = (np.floor((p1/bin_edges[1])))*bin_edges[1]
    # p1[p1==1]= 0 
    # p1+=binm[0]


    deltt = np.diff(df_hist['time'])
    flipav = np.diff(p1,prepend = 1)<0
    spin_time = np.median(np.diff(df_hist['time'].values[flipav]))

    # print(df_hist['time'])
    #interpolate the one dimensional phase angle array and make square
    flip = np.insert(np.logical_or.reduce([np.diff(p1)<0,
                                   np.diff(df_hist['time'])>spin_time,
                                   np.diff(df_hist['orbit'])!=0
                                   ]),0,True)
    cum_phase = np.cumsum(flip)-1+p1
    tot_spinum = np.sum(flip)
    tot_phase = (np.stack([np.unique(p1)]*tot_spinum)+np.arange(tot_spinum).reshape(-1,1)).reshape(-1,1)
    

    t_dict = square_param_accum(df_hist,cum_phase,tot_phase,len(binm),
                                accum_params = ['time','orbit','time','time','dt','count','phase','time'],
                                accum_actions= [np.nanmean,np.nanmax,np.nanmin,np.nanmax,np.nansum,None,None,None],
                                accum_labels = ['t_mean','orbit','start_time','end_time','dt','count','phase','sqr_time'])


    # interpolate time bins to make continuous
    tbin = np.median(np.diff(t_dict['t_mean'][~np.isnan(t_dict['t_mean'])]))
    dt = t_dict['t_mean'][-1]-t_dict['t_mean'][0]
    time_bins = np.linspace(t_dict['t_mean'][0]-tbin/2,t_dict['t_mean'][-1]+tbin/2,int(dt/tbin)+1)
    t_loc = np.digitize(t_dict['t_mean'],time_bins[:-1].flatten()).flatten()-1


    # major_dat = {}
    # major_dat['time'] = (time_bins[1:]+time_bins[:-1])/2
    # minor_dat = {}
    dat = {'eph':[(time_bins[1:]+time_bins[:-1]).reshape(-1,1)/2]}
    min_labs = {'eph':[np.array(['time'])]}
    for nam,val in t_dict.items():
        thing = np.zeros((len(time_bins)-1,val.shape[1]))*np.nan
        thing[t_loc] = val
        if val.shape[1]>1:    
            dat[nam] = thing
            min_labs[nam]=binm.round(6)
        else:
            dat['eph'].append(thing)
            min_labs['eph'].append(np.array([nam]))
    dat['eph'] = np.concatenate(dat['eph'],axis = 1)
    min_labs['eph'] = np.concatenate(min_labs['eph'])
    # print(np.concatenate(min_labs))
    # f_counts = np.zeros((len(time_bins)-1,tot['count'].shape[1]))*np.nan
    # f_counts[t_loc] = tot['count'] 
    # f_counts = t_dict['count']
    # del t_dict['count']
    # out_dat['time'][np.isnan(out_dat['time'])] = (time_bins[1:]+time_bins[:-1])[np.isnan(out_dat['time'])]/2

    # return(mat_df(list(dat.values()),minor_lables = min_labs
    #             [list(major_dat.keys())]+[binm]*len(minor_dat),['eph','hist','phase']))

    return(mat_df(dat = list(dat.values()), 
                    minor_lables = list(min_labs.values()),
                    major_lables = list(dat.keys())))

def square_up2(df_in,bin_width = 6,spin_sum = 1,nan_edge = 'mean',square_params = {}):
    # df_hist = accum_duplicate_times(df_in)

    def square_bin(df,bin_edges,
                    bin_width = 6,
                        spin_num = 1,spin_div = 4):

        stop_flip = np.argwhere(np.diff(df['phase'])<0).flatten().astype(int)
        start_flip = stop_flip+1
        spint = np.nanmedian(df['time'].values[stop_flip][1:]-df['time'].values[start_flip][:-1])
        cyclet = np.nanmedian(np.diff(df['time'].values[stop_flip]))


        # print('spint:%f'%spint)
        # print('cyclet:%f'%cyclet)
        # print('div:%f'%(cyclet/spint))

        spin_time = spint/spin_div
        spin_time = cyclet/spint
        # print(spin_time)
        spin_time = cyclet/spin_div
        # print(spin_time)
        # print(np.median(np.diff(df['time'].values[stop_flip]))/spin_time)
        # bin_edges = np.linspace(0,1,int(360/bin_width)+1)

        binm =(bin_edges[1:]+bin_edges[:-1])/2

        dt = max(df['time'])-min(df['time'])+spin_time
        
        spino = int(np.ceil(dt/spin_time))
        time_bins = np.linspace(min(df['time'])-spin_time/2,
                                min(df['time'])+spino*spin_time+spin_time/2,
                                spino+1)
        from scipy.stats import binned_statistic_2d as bin2d
        
        return(bin2d(df['time'],df['phase'],df['count'],
                               bins = [time_bins,bin_edges],expand_binnumbers=True,statistic = 'sum'))

    def manybin(df,bin_edges,labels,actions,accum_labels):
        counts,xbins,ybins,locs = square_bin(df,bin_edges,**square_params)
        dat = {}
        for lab in np.unique(np.array(labels)):
            things = (np.zeros(counts.shape)*np.nan)
            things[[locs[0,:]-1,locs[1,:]-1]] = df[lab].values
            dat[lab] = things
        
        nanb = np.sum(np.isnan(things),axis = 1)
        full = np.logical_and(nanb>0,nanb<things.shape[1])
        thing = np.sum(full)/len(full)
        split_bins = np.logical_and(full[1:],full[:-1])
        start_split = np.insert(split_bins,0,False)
        stop_split = np.insert(split_bins,-1,False)


        start_log = np.isnan(things[start_split])
        things[start_split][start_log]=things[stop_split][start_log]

        things = things[~stop_split]
        
        new_full = np.sum(np.isnan(things),axis = 1)
        # remove_parts = np.logical_or(new_full==things.shape[1],new_full == 0)
        remove_parts = ~new_full.astype(bool)
        out_dat = {}
        for lab,act,nlab in zip(labels,actions,accum_labels):
            sqr_thing = dat[lab].copy()
            
            sqr_thing[start_split][start_log] = dat[lab][stop_split][start_log]
            sqr_thing = sqr_thing[~stop_split][remove_parts]
            out_dat[nlab]=(act(sqr_thing,axis =1).reshape(-1,1) if act else sqr_thing.copy())
        return(out_dat)


    bin_edges = np.linspace(0,1,int(360/bin_width)+1)#-3/360
    binm =(bin_edges[1:]+bin_edges[:-1])/2
    t_dict = manybin(accum_duplicate_times(df_in),bin_edges,
                                labels = ['time','orbit','dt','count'],
                                actions= [np.nanmean,np.nanmax,np.nansum,None],
                                accum_labels = ['t_mean','orbit','dt','count'])



    # interpolate time bins to make continuous
    tbin = np.nanmedian(np.diff(t_dict['t_mean'][~np.isnan(t_dict['t_mean'])]))
    dt = np.nanmax(t_dict['t_mean'])-np.nanmin(t_dict['t_mean'])
    time_bins = np.linspace(t_dict['t_mean'][0]-tbin/2,t_dict['t_mean'][-1]+tbin/2,int(dt/tbin)+1)
    t_loc = np.digitize(t_dict['t_mean'],time_bins[:-1].flatten()).flatten()-1


    for nam in ['start_time','end_time']:
        t_dict[nam] = t_dict['t_mean'].copy()

    eph_dat = {'time':(time_bins[1:]+time_bins[:-1]).reshape(-1,1)/2}
    mat_dat = {}
    min_labs = {'eph':[np.array(['time'])]}
    for nam,val in t_dict.items():
        thing = np.zeros((len(time_bins)-1,val.shape[1]))*np.nan
        thing[t_loc] = val
        if val.shape[1]>1:    
            mat_dat[nam] = thing
            min_labs[nam]=binm.round(6)
        else:
            eph_dat[nam] = thing
            min_labs['eph'].append(np.array([nam]))

    # print(tbin)
    min_labs['eph'] = np.concatenate(min_labs['eph'])
    tmean = eph_dat['t_mean'].copy()
    tbo = eph_dat['time'].copy()
    eph_dat['start_time'][1:] = (tmean[1:]+tmean[:-1])/2
    eph_dat['start_time'][0] += -tbin/2
    eph_dat['end_time'][:-1] = (tmean[1:]+tmean[:-1])/2
    eph_dat['end_time'][-1] += tbin/2
    if nan_edge == 'mean':
        eph_dat['start_time'][np.isnan(eph_dat['start_time'])] = tmean[np.isnan(eph_dat['start_time'])]-tbin/2
        eph_dat['end_time'][np.isnan(eph_dat['end_time'])] = tmean[np.isnan(eph_dat['end_time'])]+tbin/2
    elif nan_edge == 'bin':
        eph_dat['start_time'][np.isnan(eph_dat['start_time'])] = tbo[np.isnan(eph_dat['start_time'])]-tbin/2
        eph_dat['end_time'][np.isnan(eph_dat['end_time'])] = tbo[np.isnan(eph_dat['end_time'])]+tbin/2

    # return(eph_dat,mat_dat,min_labs)
    return(mat_df(dat = [np.concatenate(list(eph_dat.values()),axis = 1)]+list(mat_dat.values()), 
                                minor_lables = list(min_labs.values()),
                                major_lables = list(min_labs.keys())))



def square_up3(df_in,bin_width = 6,spin_sum = 1,nan_edge = 'mean',square_params = {}):
    # df_hist = accum_duplicate_times(df_in)

    def square_bin(df,bin_edges,
                    bin_width = 6,
                        spin_num = 1,binw = 4):


        stop_flip = np.argwhere(np.diff(df['phase'])<0).flatten().astype(int)
        start_flip = stop_flip+1 

        spint = np.nanmedian(df['time'].values[stop_flip][1:]-df['time'].values[start_flip][:-1])
        cyclet = np.nanmedian(np.diff(df['time'].values[stop_flip]))

        t_flip = np.argwhere(np.logical_or(
                                np.logical_and(np.diff(df['phase'])<0,
                                                np.diff(df['time'])>spint),
                                np.diff(df['time'])>cyclet*2)
                                            ).flatten().astype(int)
        start_flip = np.insert(t_flip+1,0,0)


        t_start = df['time'].values[np.insert(t_flip+1,0,0)]
        # t_start = np.insert(t_start,min(t_start)-cyclet/2,0)

        t_stop = np.append(df['time'].values[t_flip],t_start[-1]+spint)
        # t_stop = np.insert(t_start,min(t_start)-cyclet/2,0)
        time_bins = np.stack([t_start-spint*binw,t_stop+spint*binw]).T.flatten()


        binm =(bin_edges[1:]+bin_edges[:-1])/2

        from scipy.stats import binned_statistic_2d as bin2d
        
        return(bin2d(df['time'],df['phase'],df['count'],
                               bins = [time_bins,bin_edges],expand_binnumbers=True,statistic = 'sum'))

    def manybin(df,bin_edges,labels,actions,accum_labels,max_bint = 300):
        counts,xbins,ybins,locs = square_bin(df,bin_edges,**square_params)
        dat = {}
        for lab in np.unique(np.array(labels)):
            things = (np.zeros(counts.shape)*np.nan)
            things[[locs[0,:]-1,locs[1,:]-1]] = df[lab].values
            dat[lab] = things
        dat['count'] = counts

        out_dat = {}
        # keep = np.diff(xbins)<max_bint
        for lab,act,nlab in zip(labels,actions,accum_labels):
            out_dat[nlab]=(act(dat[lab],axis =1).reshape(-1,1) if act else dat[lab].copy())[::2]
        return(out_dat)

    def interpt(time):
        t_no = time[~np.isnan(time)]
        t_stepin = (t_no-t_no[0])/(t_no[-1]-t_no[0])
        dt = t_no[-1]-t_no[0]
        spino = np.median(np.diff(t_no))
        return(np.interp(np.linspace(0,1,int(dt/spino)),t_stepin,t_no,left = np.nan,right = np.nan))


    bin_edges = np.linspace(0,1,int(360/bin_width)+1)#-3/360
    binm =(bin_edges[1:]+bin_edges[:-1])/2
    t_dict = manybin(accum_duplicate_times(df_in),bin_edges,
                                labels = ['time','orbit','dt','count','phase'],
                                actions= [np.nanmean,np.nanmax,np.nansum,None,None],
                                accum_labels = ['t_mean','orbit','dt','count','phase'])



    # interpolate time bins to make continuous
    # tbin = np.nanmedian(np.diff(t_dict['t_mean'][~np.isnan(t_dict['t_mean'])]))
    # dt = np.nanmax(t_dict['t_mean'])-np.nanmin(t_dict['t_mean'])
    # time_bins = np.linspace(t_dict['t_mean'][0]-tbin/2,t_dict['t_mean'][-1]+tbin/2,int(dt/tbin)+1)
    tt = interpt(t_dict['t_mean'])
    tbin = np.nanmedian(np.diff(tt))
    dif_norm = np.append(np.diff(tt),tbin)

    time_bins = np.insert(tt+dif_norm/2,0,tt[0]-tbin/2)
    # time_bins = interpt(np.insert(t_dict['t_mean']+tbin/2,0,t_dict['t_mean'][0]-tbin/2))
    # t_loc = np.digitize(t_dict['t_mean'],time_bins[:-1].flatten()).flatten()-1
    t_loc = np.digitize(t_dict['t_mean'],time_bins).flatten()-1

    # tt = tt-tbin/2
    for nam in ['start_time','end_time']:
        t_dict[nam] = t_dict['t_mean'].copy()

    eph_dat = {'time':(time_bins[1:]+time_bins[:-1]).reshape(-1,1)/2}
    # eph_dat = {'time':tt.reshape(-1,1)}
    mat_dat = {}
    min_labs = {'eph':[np.array(['time'])]}
    for nam,val in t_dict.items():
        thing = np.zeros((len(time_bins)-1,val.shape[1]))*np.nan
        # thing = np.zeros((len(tt),val.shape[1]))*np.nan
        thing[t_loc] = val
        if val.shape[1]>1:    
            mat_dat[nam] = thing
            min_labs[nam]=binm.round(6)
        else:
            eph_dat[nam] = thing
            min_labs['eph'].append(np.array([nam]))

    # print(tbin)
    print(dif_norm.shape)
    print(eph_dat['start_time'].shape)
    min_labs['eph'] = np.concatenate(min_labs['eph'])
    eph_dat['start_time'] += -dif_norm.reshape(-1,1)/2
    eph_dat['end_time'] += dif_norm.reshape(-1,1)/2


    # tmean = eph_dat['t_mean'].copy()
    # tbo = eph_dat['time'].copy()
    # eph_dat['start_time'][1:] = (tmean[1:]+tmean[:-1])/2
    # eph_dat['start_time'][0] += -tbin/2
    # eph_dat['end_time'][:-1] = (tmean[1:]+tmean[:-1])/2
    # eph_dat['end_time'][-1] += tbin/2
    # if nan_edge == 'mean':
    #     eph_dat['start_time'][np.isnan(eph_dat['start_time'])] = tmean[np.isnan(eph_dat['start_time'])]-tbin/2
    #     eph_dat['end_time'][np.isnan(eph_dat['end_time'])] = tmean[np.isnan(eph_dat['end_time'])]+tbin/2
    # elif nan_edge == 'bin':
    #     eph_dat['start_time'][np.isnan(eph_dat['start_time'])] = tbo[np.isnan(eph_dat['start_time'])]-tbin/2
    #     eph_dat['end_time'][np.isnan(eph_dat['end_time'])] = tbo[np.isnan(eph_dat['end_time'])]+tbin/2

    # return(eph_dat,mat_dat,min_labs)
    return(mat_df(dat = [np.concatenate(list(eph_dat.values()),axis = 1)]+list(mat_dat.values()), 
                                minor_lables = list(min_labs.values()),
                                major_lables = list(min_labs.keys())))

    #========= sep_dat try 1
    # dat = {'eph':[(time_bins[1:]+time_bins[:-1]).reshape(-1,1)/2]}
    # min_labs = {'eph':[np.array(['time'])]}
    # for nam,val in t_dict.items():
    #     thing = np.zeros((len(time_bins)-1,val.shape[1]))*np.nan
    #     thing[t_loc] = val
    #     if val.shape[1]>1:    
    #         dat[nam] = thing
    #         min_labs[nam]=binm.round(6)
    #     else:
    #         dat['eph'].append(thing)
    #         min_labs['eph'].append(np.array([nam]))
    # dat['eph'] = np.concatenate(dat['eph'],axis = 1)
    # min_labs['eph'] = np.concatenate(min_labs['eph'])

    # return(mat_df(dat = list(dat.values()), 
    #                 minor_lables = list(min_labs.values()),
    #                 major_lables = list(min_labs.keys())))



def square_split(hist_raw,splitby = 'orbit',square_up_params = {},split_func = square_up3):

    # dat = pd.concat([square_up2(oh).df for oh in [x for _, x in hist_raw.groupby(splitby)]],
    #                         ignore_index = True)

    # dat = []
    # for oh in [x for _, x in hist_raw.groupby(splitby)]:
    #     dat.append(square_up2(oh))
    dat = [split_func(oh,**square_up_params) for oh in [x for _, x in hist_raw.groupby(splitby)]]

    dat_out = []    
    for ds,dn in zip(dat[:-1],dat[1:]):
        n = int((min(dn['time'])-max(ds['time']))/np.nanmedian(np.diff(ds['time'])))
        d_new = np.ones((n-2,ds.df.shape[1]))*np.nan
        d_new[:,0] = np.linspace(max(ds['time']),min(dn['time']),n)[1:-1]
        
        # dfnan = pd.DataFrame(d_new,columns = ds.df.keys())

        dat_out.append(ds.df.append(pd.DataFrame(d_new,columns = ds.df.keys()),
                                    ignore_index=True))
    dat_out.append(dn.df)
    # return()
    return(mat_df(pd.concat(dat_out,ignore_index = True)))
    # return(mat_df(pd.concat([square_up2(oh).df for oh in [x for _, x in hist_raw.groupby(splitby)]],
    #                         ignore_index = True)))


    # Need to update to match mat_df_heirarchy
    # dat = {}
    # dat['time'] = (time_bins[1:]+time_bins[:-1])/2
    # for nam,val in t_dict.items():
    #     thing = np.zeros((len(time_bins)-1,val.shape[1]))*np.nan
    #     thing[t_loc] = val
    #     dat[nam] = thing
    # min_labs = []
    # for lab,val in dat.items():
    #     if val.shape[1]>1:    
    #         min_labs.append(binm.round(6))
    #     else:
    #         min_labs.append(np.array(['']))    
    # return(mat_df(dat = list(dat.values()), 
    #                 minor_lables = min_labs,
    #                 major_lables = list(dat.keys()),
    #                 major_ax='time',))

def interp_time(df_in):
    df = accum_duplicate_times(df_in)
    # interpolate time bins to make continuous
    tbin = np.median(np.diff(df['time'][~np.isnan(df['time'])]))
    s = np.nanmin(df['time'])
    st = np.nanmax(df['time'])
    dt = st-s
    time_bins = np.linspace(s-tbin/2,st+tbin/2,int(dt/tbin)+1)
    t_loc = np.digitize(df['time'],time_bins[:-1].flatten()).flatten()-1
    print(t_loc.shape)
    print(time_bins.shape)
    out_dat = {}
    for nam,val in df.items():
        print(nam)
        if np.dtype(val) == 'float64' or np.dtype(val) == 'int64':  
            thing = np.zeros(len(time_bins)-1)*np.nan
            thing[t_loc] = val.values
            out_dat[nam] = thing
    return(pd.DataFrame(out_dat))

def group_up(df_in,bin_width = 6,to_mat_df = True,dt_multiplier = 8):
    df = accum_duplicate_times(df_in,dt_multiplier = dt_multiplier)

    flip = np.diff(df['phase'],prepend = 1)<0
    spin_time = np.median(np.diff(df['time'].values[flip]))

    phase_bins = np.linspace(0,1,int(360/bin_width)+1)
    # p1 = phase_round(df['phase'].values,bin_edges,binm)


    dt = max(df['time'])-min(df['time'])
    tbin = np.median(np.diff(df['time'].values))


    spino = int(np.ceil(dt/spin_time))
    time_bins = np.linspace(min(df['time'])-tbin/2,
                            min(df['time'])+spino*spin_time+tbin/2,
                            spino+1)
    # hist =  np.histogram2d(df['time'],df['phase'],[time_bins,phase_bins],weights = df['count'],density = True)[0].T
    from scipy.stats import binned_statistic_2d as bin2d
    hist,garb,garb,bin_locs = bin2d(df['time'],df['phase'],df['count'], statistic = 'sum',
                           bins = [time_bins,phase_bins],expand_binnumbers=True)
    # return(bin_locs)
    phase = (phase_bins[1:]+phase_bins[:-1]).round(6)/2
    t = (time_bins[1:]+time_bins[:-1])/2


    df['rtime'] = phase_round(df['time'].values,time_bins,t)
    # df['rphase'] = phase_round(df['phase'].values,phase_bins,phase)
    # return(df)
    # return(df,t,bin_locs)
    p_actions = {'orbit':['orbit','max'],
                    't_mean':['time','mean'],
                    # 'end_time':['time','max'],
                    'start_time':['time','mean'],
                    'end_time':['time','mean'],
                    'dt':['dt','sum'],
                    # 'loc-X-RE':['loc-X-RE','mean'],
                    # 'loc-Y-RE':['loc-Y-RE','mean'],
                    # 'loc-Z-RE':['loc-Z-RE','mean'],
                    # 'Decl':['Decl','mean'],
                    # 'R.A.':['R.A.','mean'],
                    }
    lab = ['time']
    eph_dat = {'time':t}
    gp = df.groupby('rtime')
    glocs = np.unique(bin_locs.T-1) 
    dt_bins = np.mean(np.diff(time_bins))
    for nam,act in p_actions.items():
        # print(nam)
        thing = np.zeros(t.shape)*np.nan
        lab.append(nam)
        thing[np.unique(bin_locs.T-1)] = gp[act[0]].agg(act[1])
        eph_dat[nam] = thing*(dt_multiplier if nam =='dt' else 1)# +\
                       # (dt_bins/2 if nam == 'end_time' else 0)+\
                       # (-dt_bins/2 if nam == 'start_time' else 0))
    tmean = eph_dat['start_time'].copy()
    eph_dat['start_time'][1:] = (tmean[1:]+tmean[:-1])/2
    eph_dat['start_time'][0] +=-dt_bins/2
    eph_dat['end_time'][:-1] = (tmean[1:]+tmean[:-1])/2
    eph_dat['end_time'][-1] += dt_bins/2
    eph_dat['start_time'][np.isnan(eph_dat['start_time'])] = tmean[np.isnan(eph_dat['start_time'])]-dt_bins/2
    eph_dat['end_time'][np.isnan(eph_dat['end_time'])] = tmean[np.isnan(eph_dat['end_time'])]+dt_bins/2

    return(mat_df([np.stack(list(eph_dat.values())).T,hist],[lab,phase],['eph','count']))

def group_up2(df_in,bin_width = 6,to_mat_df = True,dt_multiplier = 8,binw = 4):
    df = accum_duplicate_times(df_in,dt_multiplier = dt_multiplier)


    stop_flip = np.argwhere(np.diff(df['phase'])<0).flatten().astype(int)
    start_flip = stop_flip+1 

    spint = np.nanmedian(df['time'].values[stop_flip][1:]-df['time'].values[start_flip][:-1])
    cyclet = np.nanmedian(np.diff(df['time'].values[stop_flip]))

    t_flip = np.argwhere(np.logical_or(
                            np.logical_and(np.diff(df['phase'])<0,
                                            np.diff(df['time'])>spint*binw),
                            np.diff(df['time'])>cyclet+spint)
                                        ).flatten().astype(int)
    start_flip = np.insert(t_flip+1,0,0)


    t_start = df['time'].values[np.insert(t_flip+1,0,0)]
    # t_start = np.insert(t_start,min(t_start)-cyclet/2,0)

    t_stop = np.append(df['time'].values[t_flip],t_start[-1]+spint)
    # t_stop = np.insert(t_start,min(t_start)-cyclet/2,0)
    time_bins = np.stack([t_start-spint*binw,t_stop+spint*binw]).T.flatten()
    phase_bins = np.linspace(0,1,int(360/bin_width)+1)
    # dt = max(df['time'])-min(df['time'])
    # tbin = np.median(np.diff(df['time'].values))


    # spino = int(np.ceil(dt/spin_time))
    # time_bins = np.linspace(min(df['time'])-tbin/2,
    #                         min(df['time'])+spino*spin_time+tbin/2,
    #                         spino+1)
    # hist =  np.histogram2d(df['time'],df['phase'],[time_bins,phase_bins],weights = df['count'],density = True)[0].T
    from scipy.stats import binned_statistic_2d as bin2d
    hist,garb,garb,bin_locs = bin2d(df['time'],df['phase'],df['count'], statistic = 'sum',
                           bins = [time_bins,phase_bins],expand_binnumbers=True)
    # return(bin_locs)
    phase = (phase_bins[1:]+phase_bins[:-1]).round(6)/2
    t = (time_bins[1:]+time_bins[:-1])/2


    df['rtime'] = phase_round(df['time'].values,time_bins,t)
    # df['rphase'] = phase_round(df['phase'].values,phase_bins,phase)
    # return(df)
    # return(df,t,bin_locs)
    p_actions = {'orbit':['orbit','max'],
                    't_mean':['time','mean'],
                    # 'end_time':['time','max'],
                    'start_time':['time','mean'],
                    'end_time':['time','mean'],
                    'dt':['dt','sum'],
                    # 'loc-X-RE':['loc-X-RE','mean'],
                    # 'loc-Y-RE':['loc-Y-RE','mean'],
                    # 'loc-Z-RE':['loc-Z-RE','mean'],
                    # 'Decl':['Decl','mean'],
                    # 'R.A.':['R.A.','mean'],
                    }
    lab = ['time']
    eph_dat = {'time':t}
    gp = df.groupby('rtime')
    glocs = np.unique(bin_locs.T-1) 
    dt_bins = np.mean(np.diff(time_bins))
    for nam,act in p_actions.items():
        # print(nam)
        thing = np.zeros(t.shape)*np.nan
        lab.append(nam)
        thing[np.unique(bin_locs.T-1)] = gp[act[0]].agg(act[1])
        eph_dat[nam] = thing*(dt_multiplier if nam =='dt' else 1)# +\
                       # (dt_bins/2 if nam == 'end_time' else 0)+\
                       # (-dt_bins/2 if nam == 'start_time' else 0))
    tmean = eph_dat['start_time'].copy()
    eph_dat['start_time'][1:] = (tmean[1:]+tmean[:-1])/2
    eph_dat['start_time'][0] +=-dt_bins/2
    eph_dat['end_time'][:-1] = (tmean[1:]+tmean[:-1])/2
    eph_dat['end_time'][-1] += dt_bins/2
    eph_dat['start_time'][np.isnan(eph_dat['start_time'])] = tmean[np.isnan(eph_dat['start_time'])]-dt_bins/2
    eph_dat['end_time'][np.isnan(eph_dat['end_time'])] = tmean[np.isnan(eph_dat['end_time'])]+dt_bins/2

    return(mat_df([np.stack(list(eph_dat.values())).T,hist],[lab,phase],['eph','count']))



def square_bin(df_in,bin_width = 6,spin_num = 1):
    df = accum_duplicate_times(df_in)

    flip = np.diff(df['phase'],prepend = 1)<0
    spin_time = np.median(np.diff(df['time'].values[flip]))

    phase_bins = np.linspace(0,1,int(360/bin_width)+1)
    
    dt = max(df['time'])-min(df['time'])
    tbin = np.median(np.diff(df['time'].values))

    spino = int(np.ceil(dt/spin_time))
    time_bins = np.linspace(min(df['time'])-tbin,
                            min(df['time'])+spino*spin_time+tbin,
                            spino+1)
    # time_bins = np.linspace(min(df['time'])-tbin,max(df['time'])+tbin,int(dt/(spin_time*spin_num)))
    from scipy.stats import binned_statistic_2d as bin2d

    bin_locs = bin2d(df['time'],df['phase'],df['count'],
                           bins = [time_bins,phase_bins],expand_binnumbers=True)[-1]
    return(df,bin_locs)
    # p_actions = {'time':np.nanmean,'orbit':np.unique,'dt':np.nansum}
    # s_actions = {'start time':['time',np.nanmin],'end time':['time',np.nanmax]}

    # e_dat = []
    # e_labs = []
    # for nam,f in p_actions.items():
    #     e_labs.append(nam)
    #     dh = np.zeros(counts.shape)*np.nan
    #     dh[binlocs] =


    # counts,binlocs = bin2d(df_in['time'],df_in['phase'],df_in['count'],statistic = 'sum'
    #                        bins = [time_bins,phase_bins])[0]

    # p_actions = {'time':'mean','orbit':'max','dt':'sum','count':'sum'}
    # # s_actions = {'start time':['time',np.nanmin],'end time':['time',np.nanmax]}

    # e_dat = []
    # e_labs = []
    # for nam,f in p_actions.items():
    #     e_labs.append(nam)
    #     dh = np.zeros(counts.shape)*np.nan
    #     dh[binlocs] = 







    # hist =  np.histogram2d(df['time'],df['phase'],[time_bins,phase_bins],weights = df['count'],
    #                        density = False)[0]
    # int_time = np.sum(np.histogram2d(df['time'],df['phase'],[time_bins,phase_bins],weights = df['dt'],
    #                        density = False)[0],axis = 1)

    phase = (phase_bins[1:]+phase_bins[:-1])/2
    t = (time_bins[1:]+time_bins[:-1])/2

    # return(hdf(t.reshape(-1,1),['time'],hist.T,phase))
    # return(mat_df([np.stack([t,int_time]).T,hist],[['time','dt'],phase],['eph','hist']))