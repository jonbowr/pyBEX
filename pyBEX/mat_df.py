import numpy as np
import pandas as pd

class mat_df_heirarchy:

    def __init__(self,dat = [] ,minor_lables = [],major_lables= [],major_ax = None):
        if type(dat)==list:
            maj_tot = []
            # min_tot = []
            for m_labs,maj_lab in zip(minor_lables,major_lables):
                maj_tot += [maj_lab]*len(m_labs) 
                # min_tot += list(m_labs)

            h = pd.MultiIndex.from_arrays([maj_tot,list(np.concatenate(minor_lables))],names = ['type','val'])

            self.df = pd.DataFrame(np.concatenate(dat,axis = 1),columns = h)
            # self.major_df = self.df[major_ax].keys()[0]
            # self.dependent_df = [thing for thing in self.df.keys().values if self.major_df in thing]

            if not major_ax:
                major_ax = [0,0]
            if type(major_ax[0]) == int:
                self.major_ax = (major_lables[major_ax[0]],minor_lables[major_ax[0]][major_ax[1]])        
            elif type(major_ax[0]) == str:
                self.major_ax = major_ax
        elif type(dat)==pd.DataFrame:
            self.df = dat
            self.major_ax = major_ax
            self.dependent_df = dependent_data
            # print(dat.keys().values[0])
            # if not major_ax:
            #     self.major_ax = dat.keys().values[0]
            #     self.dependent_df = dat.keys().values[0][0]
            # elif type(major_ax) == int:
            #     self.major_ax =  dat.keys().values[major_ax]
            #     self.dependent_df = dat.keys().values[major_ax,0]
            # elif type(major_ax) == tuple and type(major_ax[0]) == str:
            #     self.major_ax = major_ax
            #     self.dependent_df = major_ax[0]
        


        # self.label_tree = {thing:[(thing,st) for st in stuff] for thing,stuff in zip(major_lables,minor_lables)}
        # self.label_tree = {thing:np.array(stuff) for thing,stuff in zip(major_lables,minor_lables)}


    def __getitem__(self,item):
        if item != self.major_df:
            return(self.df[item])
        else: 
            return(self.df[[k for k in self.df.keys() if self.major_df in k]])
        # if item in self.df:
    #         return(self.df[item])
    #     elif item in self.df[self.dependent_df]:
    #         return(self.df[self.dependent_df][item])

    def __setitem__(self,item,value):
        self.df[item] = value

    def __str__(self):
        return(str(self.df))

    def __call__(self):
        return(np.unique(np.stack(self.df.keys().values)[:,0]))
        # return(self.df)

    def __iter__(self):
        mkeys = np.stack(self.df.keys())[:,0]
        ind = np.sort(np.unique(mkeys,return_index = True)[1])
        # return(iter({mkeys[i]:self.df[mkeys[i]] for i in ind}))
        return(iter(mkeys[ind]))

    def __repr__(self):
        return(self.df.__repr__())

    def loc(self,locr):
        # return(self.df.loc[locr])
        return(mat_df(self.df.loc[locr],major_ax = self.major_ax))

    def get_axes(self):
        return({lab:[self.df[self.major_ax].values,
                        self.labels()[lab].astype(float)] for lab in self\
                        if lab != self.major_ax[0]})

    def get_bins(self):
        # ax = self.get_axes()
        hbins = {}
        for lab,ax in self.get_axes().items():
            bins = []
            for a in ax:
                d = np.nanmedian(np.diff(a))
                bins.append(np.insert(a+d/2,0,a[0]-d/2))
            hbins[lab] = bins
        return(hbins)

    def get_base_ax(self):
        return(self.df[self.major_ax].values)

    def get_base_bins(self):
        a = self.get_base_ax()
        d = np.nanmedian(np.diff(a))
        return(np.insert(a+d/2,0,a[0]-d/2))


    # def add(self,dat,ax = 0,statistic = 'mean',inplace = True,label = ''):
    #     from scipy.stats import binned_statistic as bs
        # if ax ==0:
        #     self.df[(self.dependent_df,label)] = bs(dat,bins= )


    def labels(self):
        return({lab:self.df[lab].keys().values for lab in self})

    def get_mats(self):
        # mats = []
        # labs = []
        return([self.df[thing].values for thing in self.label_tree])

    def keys(self):
        key = {}
        return({lab:[(lab,k) for k in self.df[lab].keys().values] for lab in self})

    def reduce(self,reduce_type = 'sum'):
        reduce_df = self.df[[nam for nam in self.labels() \
                         if nam != self.dependent_df]].sum(axis = 1,skipna = True,level = 0)

        return(pd.concat([reduce_df],
                  keys =[reduce_type.upper()],
                  names = list(self.df.keys().names),
                  axis = 1,
                  ))

    def _repr_pretty_(self, p, cycle):
        from IPython.display import display
        # display(self.reduce())
        display_cols = []
        for labs,vals in self.keys().items(): 
            if self.major_df in labs:
                display_cols += vals
            else:
                display_cols += [vals[i] for i in [0,-1]]
        # display_cols = [vals for vals  in self.df.keys() if vals[0]== self.dependent_df ]
        display(pd.concat([self.df[display_cols],self.reduce()],axis = 1))

    def accum_bins(self,binnum = 1,reduce_f ={}):
        n_dat = []
        m_labs = []
        maj_labs = []
        rem = self.df[self.dependent_df].shape[0]%binnum
        tf = np.ones(self.df.shape[0]).astype(bool)
        tf[np.random.choice(range(len(tf)), rem, replace=False)] = False
        df = self.df.loc[tf]
        for mtype,vals in self.labels().items():
            if mtype == self.dependent_df:
                dep_dat = []
                if reduce_f:
                    for lab in df[mtype]:
                        print(lab)
                        if lab in reduce_f:
                            dep_dat.append(reduce_f[lab](df[mtype][lab].values.reshape(-1,binnum),
                                                      axis = 1))
                        else:
                            dep_dat.append(np.nanmean(df[mtype][lab].values.reshape(-1,binnum),
                                                      axis = 1))
                    n_dat.append(np.stack(dep_dat).T)
                else:
                    n_dat.append(np.nanmean(df[mtype].values.reshape(-1,binnum,
                                                    df[mtype].shape[1]),axis = 1))
            if mtype != self.dependent_df:
                n_dat.append(np.nansum(df[mtype].values.reshape(-1,binnum,
                                                    df[mtype].shape[1]),axis = 1))
            m_labs.append(list(self.df[mtype].keys().values))
            maj_labs.append(mtype)
        return(mat_df(n_dat,m_labs,maj_labs))


class mat_df:

    def __init__(self,dat = [] ,minor_lables = [],major_lables= [],
                                     dependent_data = 0,major_ax = None):
        if type(dat)==list:
            # maj_tot = []
            # min_tot = []
            # for m_labs,maj_lab in zip(minor_lables,major_lables):
            #     maj_tot += [maj_lab]*len(m_labs) 
            #     min_tot += list(m_labs)
            # h = pd.MultiIndex.from_arrays([maj_tot,min_tot],names = ['type','lab'])
            # self.df = pd.DataFrame(np.concatenate(dat,axis = 1),columns = h)
            maj_tot = []
            for m_labs,maj_lab in zip(minor_lables,major_lables):
                maj_tot += [maj_lab]*len(m_labs) 
            h = pd.MultiIndex.from_arrays([maj_tot,list(np.concatenate(minor_lables))],names = ['type','val'])
            self.df = pd.DataFrame(np.concatenate(dat,axis = 1),columns = h)
            self.dependent_df = major_lables[dependent_data]



            if not major_ax:
                major_ax = [0,0]
            if type(major_ax[0]) == int:
                self.major_ax = (major_lables[major_ax[0]],minor_lables[major_ax[0]][major_ax[1]])        
            elif type(major_ax[0]) == str:
                self.major_ax = major_ax
        elif type(dat)==pd.DataFrame:
            self.df = dat
            if not major_ax:
                self.major_ax = dat.keys().values[0]
                self.dependent_df = dat.keys().values[0][0]
            elif type(major_ax) == int:
                self.major_ax =  dat.keys().values[major_ax]
                self.dependent_df = dat.keys().values[major_ax,0]
            elif type(major_ax) == tuple and type(major_ax[0]) == str:
                self.major_ax = major_ax
                self.dependent_df = major_ax[0]
        


        # self.label_tree = {thing:[(thing,st) for st in stuff] for thing,stuff in zip(major_lables,minor_lables)}
        # self.label_tree = {thing:np.array(stuff) for thing,stuff in zip(major_lables,minor_lables)}


    def __getitem__(self,item):
        if item in self.df:
            return(self.df[item])
        elif item in self.df[self.dependent_df]:
            return(self.df[self.dependent_df][item])

    def __setitem__(self,item,value):
        self.df[item] = value

    def __str__(self):
        return(str(self.df))

    def __call__(self):
        return(self.df)

    # def __iter__(self):
    #     return(iter(np.unique(np.stack(self.df.keys().values)[:,0])))
    def __iter__(self):
        mkeys = np.stack(self.df.keys())[:,0]
        ind = np.sort(np.unique(mkeys,return_index = True)[1])
        # return(iter({mkeys[i]:self.df[mkeys[i]] for i in ind}))
        return(iter(mkeys[ind]))

    def __repr__(self):
        return(self.df.__repr__())

    def loc(self,locr):
        # return(self.df.loc[locr])
        return(mat_df(self.df.loc[locr],major_ax = self.major_ax))

    def get_axes(self):
        return({lab:[self.df[self.major_ax].values,
                        self.labels()[lab].astype(float)] for lab in self\
                        if lab != self.major_ax[0]})

    def get_bins(self):
        # ax = self.get_axes()
        hbins = {}
        for lab,ax in self.get_axes().items():
            bins = []
            for a in ax:
                d = np.nanmedian(np.diff(a))
                bins.append(np.insert(a+d/2,0,a[0]-d/2))
            hbins[lab] = bins
        return(hbins)

    def get_base_ax(self):
        return(self.df[self.major_ax].values)

    def get_base_bins(self):
        a = self.get_base_ax()
        d = np.nanmedian(np.diff(a))
        return(np.insert(a+d/2,0,a[0]-d/2))


    def add(self,x,values,ax = 0,statistic = 'mean',inplace = True,label = ''):
        if ax ==0:
            from scipy.stats import binned_statistic as bs
            self.df[(self.dependent_df,label)] = bs(
                                                    x,
                                                    values,
                                                    bins= self.get_base_bins(),
                                                    statistic = statistic
                                                    )[0]



    def labels(self):
        return({lab:self.df[lab].keys().values for lab in self})

    def get_mats(self):
        # mats = []
        # labs = []
        return([self.df[thing].values for thing in self.label_tree])

    def keys(self):
        return({lab:[(lab,k) for k in self.df[lab].keys().values] for lab in self})

    def reduce(self,reduce_type = 'sum'):
        reduce_df = self.df[[nam for nam in self.labels() \
                         if nam != self.dependent_df]].sum(axis = 1,skipna = True,level = 0)

        return(pd.concat([reduce_df],
                  keys =[reduce_type.upper()],
                  names = list(self.df.keys().names),
                  axis = 1,
                  ))
    
    def mask(self,log,inplace = True,keep_shape = True):
        # if not inplace:
        ndf = self.df.copy()
        for k,keys in self.keys().items():
            if keep_shape:
                ndf.loc[log,keys] = np.nan
        ndf.loc[:,self.major_ax] = self.df[self.major_ax]

        if inplace:
            self.df = ndf
            return(self)
        else:
            return(mat_df(ndf))

    def _repr_pretty_(self, p, cycle):
        from IPython.display import display
        # display(self.reduce())
        display_cols = []
        for labs,vals in self.keys().items(): 
            if labs == self.dependent_df:
                display_cols += vals
            else:
                display_cols += [vals[i] for i in [0,-1]]
        # display_cols = [vals for vals  in self.df.keys() if vals[0]== self.dependent_df ]
        display(pd.concat([self.df[display_cols],self.reduce()],axis = 1))

    def accum_bins(self,binnum = 1,reduce_f ={}):
        n_dat = []
        m_labs = []
        maj_labs = []
        rem = self.df[self.dependent_df].shape[0]%binnum
        tf = np.ones(self.df.shape[0]).astype(bool)
        # tf[np.random.choice(range(len(tf)), rem, replace=False)] = False
        tf[-rem:] = False
        df = self.df.loc[tf]

        # df = self.df.loc[-rem,:]
        for mtype,vals in self.labels().items():
            if mtype == self.dependent_df:
                dep_dat = []
                if reduce_f:
                    for lab in df[mtype]:
                        print(lab)
                        if lab in reduce_f:
                            dep_dat.append(reduce_f[lab](df[mtype][lab].values.reshape(-1,binnum),
                                                      axis = 1))
                        else:
                            dep_dat.append(np.nanmean(df[mtype][lab].values.reshape(-1,binnum),
                                                      axis = 1))
                    n_dat.append(np.stack(dep_dat).T)
                else:
                    n_dat.append(np.nanmean(df[mtype].values.reshape(-1,binnum,
                                                    df[mtype].shape[1]),axis = 1))
            if mtype != self.dependent_df:
                n_dat.append(np.nansum(df[mtype].values.reshape(-1,binnum,
                                                    df[mtype].shape[1]),axis = 1))
            m_labs.append(list(self.df[mtype].keys().values))
            maj_labs.append(mtype)
        return(mat_df(n_dat,m_labs,maj_labs))