import os

data_file = os.path.join('in','mdl_data.py')
exec(open(data_file).read())

# ========================================================

# =======================================================

tool_axis_Z = [1.,0.,0.]

Origin_tool_beg = [L_cylindre/1.0, 0. , 0.]
Origin_tool_beg2 = [L_cylindre/1.0-2*f_tooth/1000., 0. , 0.]
# Origin_tool_end =[L_cylindre/1. - avancement_usinage * L_machining - L_machining , 0. , 0.]
Origin_tool_end =[L_cylindre/1.  - L_machining , 0. , 0.]

nor_pnt_f_spin = list()
nor_pnt_f_spin.append([ [ tool_axis_Z, Origin_tool_beg], [0., spin_rate] ])
nor_pnt_f_spin.append([ [ tool_axis_Z, Origin_tool_beg2], [Vf, spin_rate] ])
if stability : 
    for xx in lst_axis_pts_stab : 
        nor_pnt_f_spin.append([ [ tool_axis_Z, [xx, 0., 0.]], [Vf, spin_rate], True ])

nor_pnt_f_spin.append([ [ tool_axis_Z, Origin_tool_end], [Vf, spin_rate] ])

traj_def={'type': 'milling',
'nor_pnt_f_spin': nor_pnt_f_spin,
'x_init': [0.,0.,1.],
'f_unit': 'm/min',
's_unit': 'rev/min' }

# 'f_unit': 'mm/rev',
# 's_unit': 'rev/min' 