#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

n2m_bin_path = os.environ['n2m']
nessy2m_path = os.path.join(n2m_bin_path,'..','..')

sys.path.append(os.path.join(nessy2m_path,'pre','bin'))
import gent_user as GTU

# ==============================================================================

data_file = os.path.join('in','mdl_data.py')
exec(open(data_file).read())

draw_tool = False
if len(sys.argv) > 1 : 
    if sys.argv[1] != '-d' :
        raise Exception("Only option '-d' is allowed")
    draw_tool = True   

0.0275
0.0775
milieu = 0.05

ref_insert = GTU.gent_straight_insert( tool_width, rake_face_thick, 
                                       id_node_dyn = 0,
                                       width_eTool = width_eTool ) 

#  Positioning of the insert in the (O_T, X_T, Z_T) plane
origin_in_R_T = [R_int + 0.5*max(Ap_start, Ap_end), 0., 0.]
X_in_R_T = [  1., 0., 0.]
Y_in_R_T = [  0., 0., 1.]
Z_in_R_T = [  0.,-1., 0.]
frame_old_in_new = [ X_in_R_T, Y_in_R_T, Z_in_R_T, origin_in_R_T ]

GTU.transf_eTool_frame(frame_old_in_new, ref_insert, modify_original = True)

if draw_tool : 
    GTU.draw_eT(ref_insert, legend='ref_insert_in_plane_XT-ZT')

GTU.write_tool(ref_insert, 'in/1DOF_150Hz-tool_def.py')
