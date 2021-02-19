import os
import re

def get_orbit_data(oroot,data_type = 'tof',
                   out_loc = '',orbit_range = [1,2],
                   e_steps= '1234567'):
    if data_type.lower() == 'tof':
        fil_iter = 'lotf-'
    elif 'hist' in data_type.lower():
        fil_iter = 'lohb-'

    # for root, dirs, files in os.walk(oroot):
    for rooti in os.listdir(oroot):
        orbit_nam = rooti.split('/')[-1]
        if orbit_nam:  
            orbit_num = int(re.sub("[^0-9]",'',orbit_nam))
            if orbit_num >=orbit_range[0] and orbit_num < orbit_range[1]:
                print('=====================')
                print(orbit_nam)
                for root,dirs,filenames in os.walk(oroot+rooti):
                    for filename in filenames:
                        lab = filename.split('.')[1]    
                        if fil_iter in lab:
                            if lab[-1] in e_steps:
                                print(filename)
                                if data_type.lower() == 'tof':
                                    cmd = r'lo_de_tof -h -t tofsH %s > %s%s_lo_tofsH_e%s.txt'%(root+'/'+filename,out_loc,orbit_nam,lab[-1])
                                elif data_type.lower()=='hist':
                                    cmd = r'me_show %s -x header -x mtype=lo_triple_H >%s%s_lo_hist_H_e%s.txt'%(root+'/'+filename,out_loc,orbit_nam,lab[-1])
                                elif data_type.lower() == 'hist-double':
                                    cmd = r'me_show %s -x header -x mtype=c6 >%s%s_lo_hist_double_e%s.txt'%(root+'/'+filename,out_loc,orbit_nam,lab[-1])
                                os.system(cmd)

# syntax examples
# get_orbit_data('/usr/local/ibex/id-bas4/orbit/','tof','data/tof_dat/',orbit_range = [433,600])
# get_orbit_data('/usr/local/ibex/id-bas4/orbit/','hist','data/hist_dat/',orbit_range = [433,600])
id_leops = '/usr/local/IBEX/id-leops/orbit/'
id_bas4 = '/usr/local/ibex/id-bas4/orbit/'
id_bas5 = '/usr/local/ibex/id-bas5/orbit/'

get_orbit_data(id_bas4,data_type = 'hist-double',
               out_loc = 'data/e2_singles_dat/',orbit_range = [433,600],e_steps = '2')
