import os
import re

def list_itter(thing):
    if type(thing) is list:
        return(thing)
    elif type(thing) is str:
        return([thing])

def get_orbit_data(oroot,out_loc = '',orbit_range = [1,2],
                   data_type = 'tof',mtype = 'tofsH',
                   e_steps= '1234567',replace = False,
                    subdivide_output = True,
                   ):

    # Function to search database root and process all files
    # of Desired Type
    # oroot: database location
    # Data type: (
    #                tof: Direct event H data, 
    #             hist: H histogram data, 
    #             hist_double:E2, tof2 doubles data: 
    #              ) 
    # out_loc: directory to print the output to
    # orbit range:
    # esteps: string, all energy step tags (1,2,3,4,5,6,7,8,A,B,C) found in string will be processed
    # Replace: Bool, wheather to replace or skip generating data that already exists in the out_loc directory
    
    # Dict to relate simple keys to file itterators
    file_extensions =  {
                        'tof':'lotf-',
                        'hist':'lohb-',
                        'spin':'osbdba',
                        'sc_loc':'bm',
                        'star_sensor':'star',
                        'exposure':'lo_exp'
                        }

    # Function to generate output file names
    output_files = (lambda out_loc,out_fil,data_type,mtype,out_ext:
                        (r'%s_%s_%s_%s.txt'%(os.path.join(out_loc,out_fil),
                                                mtype,data_type,out_ext.replace('-','_'))).replace('__','_'))
    
    # Dict defining data generation commands
    output_commands = {
                'tof':(lambda in_loc,in_fil,mtype,out_fil:
                        r'lo_de_tof -h -t %s %s > %s'%(os.path.join(in_loc,in_fil),
                                                        mtype,out_fil)),
                'hist':(lambda in_loc,in_fil,mtype,out_fil:
                        r'me_show %s -x header -x mtype=%s > %s'%(os.path.join(in_loc,in_fil),
                                                        mtype,out_fil)),
                'spin':(lambda in_loc,in_fil,mtype,out_fil:
                        r'cp %s %s'%(os.path.join(in_loc,in_fil),out_fil)),

                'sc_loc':(lambda in_loc,in_fil,mtype,out_fil:
                        r'me_show %s -x header > %s'%(os.path.join(in_loc,in_fil),out_fil)),

                'star_sensor':(lambda in_loc,in_fil,mtype,out_fil:
                        r'me_show %s -x header > %s'%(os.path.join(in_loc,in_fil),out_fil)),
                'exposure':(lambda in_loc,in_fil,mtype,out_fil:
                        r'me_show %s -x header > %s'%(os.path.join(in_loc,in_fil),out_fil)),
                
                }
    

    if subdivide_output and len(list_itter(out_loc))==1:
        true_out = [('%s%s_%s'%(out_loc,dt,mt)).strip('_')+'/' for dt,mt in zip(list_itter(data_type),
                                                               list_itter(mtype))]
    else: 
        true_out = out_loc

    for ot in list_itter(true_out):
        if not os.path.exists(ot):
            print(ot)
            os.mkdir(ot)

    result_info = {}
    # Get/loop through orbit folders within database
    for rooti in os.listdir(oroot):
        # Extract the orbit number and check to see if its within orbit range 
        orbit_nam = rooti.split('/')[-1]
        if orbit_nam:  
            try:
                orbit_num = int(re.sub("[^0-9]",'',orbit_nam))
            except ValueError:
                orbit_num = -99999
                print('=====================')
                print('No orbit found: %s'%orbit_nam)
            if orbit_num >=orbit_range[0] and orbit_num < orbit_range[1]:
                print('=====================')
                print('Generating Data: %s'%orbit_nam)
                if orbit_nam not in result_info:
                    result_info[orbit_nam]={'loc':rooti,
                                            'data':[]}
                else:
                    print('Warning: repocess attempt on %s'%orbit_nam)

                # Loop through files within orbit folder identify files we want to process
                for root,dirs,filenames in os.walk(oroot+rooti):
                    for filename in filenames:
                        for dtype,mt,esteps,oloc in zip(list_itter(data_type),
                                                       list_itter(mtype),
                                                       list_itter(e_steps),
                                                       list_itter(true_out)):
                            try:
                                fil_dat = [filename,False]
                                fsplit = filename.split('.')
                                if len(fsplit)==2:
                                    lab = fsplit[-1] 
                                    fnam = fsplit[0]
                                    fil_iter = file_extensions[dtype]
                                    
                                    if fil_iter in lab:
                                        if esteps  != '':
                                            estep = lab.split('-')[-1]
                                        else: 
                                            estep = ''
                                        if estep in esteps or esteps.lower() == 'all':
                                            # print('Processing: %s'%filename)
                                            new_fil = output_files(oloc,fnam,dtype,mt,lab)
                                            cmd = output_commands[dtype](root,filename,mt,new_fil)

                                            if replace and os.path.exists(new_fil):
                                                print('Replacing %s using %s'%(new_fil.split('/')[-1],filename)) 
                                                os.remove(new_fil)
                                            elif replace or os.path.exists(new_fil)==False:
                                                print('Generating %s using %s'%(new_fil.split('/')[-1],filename))
                                            
                                            if os.path.exists(new_fil)==False:
                                                os.system(cmd)
                                                print(cmd)
                                                fil_dat[1] = True
                                            else:
                                                print('skipping %s'%(new_fil.split('/')[-1]))
                                                fil_dat[1] = 'File already exists'
                                            result_info[orbit_nam]['data'].append(fil_dat)
                            except:
                                print('Import failed: %s'%filename)
                                result_info[orbit_nam]['data'].append(fil_dat)


mtype_presets = {
                'hist':{
                        'tof2_doubles':'c6',
                        'H_triples':'lo_triple_H',
                        },
                 'tof':{
                        'H':'tofsH',
                        }
                }


def isn_data_fetch(data_base,orb_range,hist_dir,e2_dir):
    #
    get_orbit_data(data_base,data_type = 'hist',
                   out_loc = hist_dir,
                   orbit_range = orb_range,
                   e_steps = '1,2,3,4,5,6,7,A,B,C,D',replace = True)
    get_orbit_data(data_base,data_type = 'hist-double',
                   out_loc = e2_dir,
                   orbit_range = orb_range,
                   e_steps = '2,C',replace = True)


id_leops = '/usr/local/IBEX/id-leops/orbit/'
id_bas4 = '/usr/local/ibex/id-bas4/orbit/'
id_loops = '/usr/local/IBEX/id-loops/orbit/'
work_dir = '/usr/local/IBEX/work/jsx54'

hist_dir = '/usr/local/IBEX/work/jsx54/64_spin_ISN/ISN_meshow_textfiles/ISN_process_raw_data_4/hist/'
# hist_dir = '/home/jsx54/data/try_again/hist/'

e2_dir = '/usr/local/IBEX/work/jsx54/64_spin_ISN/ISN_meshow_textfiles/ISN_process_raw_data_4/e2_doubles/'
# e2_dir = '/home/jsx54/data/try_again/hist/'
tof_dir = '/usr/local/IBEX/work/jsx54/ISN_process_  aw_data_3/de/'


orb_range = [0,600]

# isn_data_fetch(id_loops,orb_range,hist_dir,e2_dir)
out_dir = './data/new_get_dat/'
# get_orbit_data(id_loops,
#                    out_loc = out_dir,
#                    orbit_range = [10,12],
#                    data_type = ['hist','hist','spin'],
#                    mtype = [
#                             mtype_presets['hist']['H_triples'],
#                             mtype_presets['hist']['tof2_doubles'],
#                             ''
#                                 ],
#                    e_steps= ['123','2,C',''],
#                    replace = False)

get_orbit_data(id_loops,
                   out_loc = out_dir,
                   orbit_range = [10,12],
                   data_type = 'exposure',
                   mtype = '',
                   e_steps= '',
                   replace = True)