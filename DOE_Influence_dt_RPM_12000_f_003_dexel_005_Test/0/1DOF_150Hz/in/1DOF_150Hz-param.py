# -*- coding: utf-8 -*-

import os

data_file = os.path.join('in','mdl_data.py')
exec(open(data_file).read())

param={
'glob':{
'd_t':d_t,
'usi':True,
'stability': stability,
'algo':0,
'draw':False,
'plot':False,
'd_inc_draw':2,
'plot_windows_time': 500 * d_t,
'plot_update_d_time':100 * d_t,
'd_inc_print':100,
'plot_res':True,
'plot_res_rt':False,
'plot_tor':False,
'plot_pow':False,
'plot_vol':False,
'plot_flo':False,
'plot_dyn':True,
'plot_dyn_o': True,
'serv':False,
'd_inc_serv':100,
'tbb_nth':0 # 0 : auto
},
'tool':{
'dyn':True,
'eff_ref': 1,
'eps_tl':1.e-9,
'draw_dep':True,
'draw_sweep':True,
'draw_cvol':True,
'draw_cf':True,
'coef_cf_plot': 5.e-5,
'coef_dep_plot': 100,
'delta_rev_out_pow':1/180.,
'delta_rev_out_flo':1./180.,
'out_tp':True,
'out_cd':False,
'out_in_rt':False,
 },
'wp':{
'dyn':False,
'draw_dep':False,
'coef_dep_plot':100.,
'eff_ref':10.
}
}