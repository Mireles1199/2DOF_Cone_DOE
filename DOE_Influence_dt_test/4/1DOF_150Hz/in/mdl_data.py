# -*- coding: utf-8 -*-

import pickle
import numpy as np

# ****************************
avancement_usinage = 0.05
# ****************************

# - Workpiece Geometry
Ap_start = 0.005 # m
Ap_end   = 0.015 # m
L_cylindre  = 150.e-3

R_int    = 50.e-3 # m
R_ext_1  = R_int + Ap_start # m
R_ext_2  = R_int + Ap_end   # m

use_truc_func = True
if not use_truc_func : 
    # Mesh coord error
    chord_error = 1.e-6

# - Operation parameters
f_tooth = 0.03 # mm/tooth 
# spin_rate = 12094. # rpm
spin_rate = 12000
#A = 0.
L_machining = L_cylindre/1.
Vf = spin_rate*f_tooth/1.e3 # m/mn

# - Tool Geometry
tool_width = 1.2*max(Ap_start, Ap_end)
rake_face_thick = 2.e-3
width_eTool = 1.5e-3

# Cutting law 
# K_f =  $K_f$

#***********
#bb = 119.1e-3 # m
K_f = 1000
Ap = "5mm to 15mm"

#

# - Dexel carpet parameters
nb_dxl_per_mcr_dxl = 200
dxl_size = 5e-05

# - Computed parameters
nb_dt_rev = 72
d_t = 60./spin_rate/nb_dt_rev

# - Stability
stability = False
if stability :
    lst_axis_pts_stab = [ L_cylindre * (1.- 0.1),
                          L_cylindre * (1.- 0.3),
                          L_cylindre * (1.- 0.5),
                          L_cylindre * (1.- 0.7),
                          L_cylindre * (1.- 0.9) ]
