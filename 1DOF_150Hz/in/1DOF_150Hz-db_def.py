# -*- coding: utf-8 -*-
import os
import math
#import debugpy

data_file = os.path.join('in','mdl_data.py')

exec(open(data_file).read())
# --------------------------------------------------------------------------
# - Dexel Data
# --------------------------------------------------------------------------
# Dexels in Y_P direction (in p direction of basis fps) :
#  u = Y_P, v = Z_P, z = X_P
# L_cylindre, l_cylindre  = (15.e-3,7.e-3)
# R_int, R_ext_2 = (5.e-3, 7.5e-3)# m



l_u_max = (R_ext_2*2)
nb_mcr_dxl_u = 1 + int( l_u_max / (nb_dxl_per_mcr_dxl*dxl_size) )
nb_dxl_per_mcr_dxl_u = int(l_u_max / (nb_mcr_dxl_u*dxl_size)) + 1
nb_tot_dexel_u = nb_mcr_dxl_u * nb_dxl_per_mcr_dxl_u
d_u = l_u_max / nb_tot_dexel_u

l_v_max = R_ext_2*2
nb_mcr_dxl_v = 1 + int( l_v_max / (nb_dxl_per_mcr_dxl*dxl_size) )
nb_dxl_per_mcr_dxl_v = int(l_v_max / (nb_mcr_dxl_v*dxl_size)) + 1
nb_tot_dexel_v = nb_mcr_dxl_v * nb_dxl_per_mcr_dxl_v
d_v = l_v_max / nb_tot_dexel_v  

print(" ------------------------")
print(" Dexel carpet data:")
print(" length u        : "+str(R_ext_2*2))
print(" length v        : "+str(R_ext_2*2))
print(" length w max    : "+str(L_cylindre))
print(" nb_mcr_dxl_u    : "+str(nb_mcr_dxl_u))
print(" nb_mcr_dxl_v    : "+str(nb_mcr_dxl_v))
print(" nb_dxl_per_mcr_dxl_u : "+str(nb_dxl_per_mcr_dxl_u))
print(" nb_dxl_per_mcr_dxl_v : "+str(nb_dxl_per_mcr_dxl_v))
print(" nb_tot_dexel_u: "+str(nb_tot_dexel_u))
print(" nb_tot_dexel_v: "+str(nb_tot_dexel_v))
print(" dexel size u  : "+str(d_u))
print(" dexel size v  : "+str(d_v))
print(" ------------------------")

w_max = L_cylindre
w_min = -L_cylindre

# ==============================================================================
# - Surface definition

if use_truc_func :
    def h_max_cono(u, v):
        r = math.sqrt(u*u + v*v)
        if r <= R_ext_2:
            # altura lineal: w_max en el centro, 0 en R_ext_2
            return w_max * (1.0 - r / R_ext_2)
        else:
            return 0.0

    def h_max_truncado(u, v):
        r2 = u*u + v*v
        r = math.sqrt(r2)

        # Fuera de la pieza
        if r < R_int or r > R_ext_2:
            return 0.0

        # Zona cilíndrica: altura constante
        if r <= R_ext_1:
            return w_max

        # Zona cónica (solo la parte exterior):
        # h(r) baja linealmente de w_max en r = R_cil a 0 en r = R_ext
        return w_max * (R_ext_2 - r) / (R_ext_2 - R_ext_1)

    def h_min_1(u,v):
        R_uv_2 = u*u + v*v
        if (R_uv_2>= R_int**2) and (R_uv_2 <= R_ext_2**2): return 0.
        else: return 0.

else : 

    # Data adaptation
    Axe_cone = 'X'
    Ab = 0.         # Axial begin position
    Ae = L_cylindre # Axial end position
    R0 = R_int      # Internal radius
    R1b = R_ext_2   # External radius, at begin position
    R1e = R_ext_1   # External radius, at end position


    # Triangles sizes

    LA = Ae - Ab  # Axial length
    R1 = min(R1b, R1e)

    cos = 1. - chord_error/R1
    alpha = 2. * np.arctan((1-cos**2)**0.5/cos)
    nb_tri_circ = int( np.ceil(2.*np.pi / ( alpha )) )
    angle_step = 2. * np.pi / nb_tri_circ
    L_edge_circ = 2. * R1 * np.sin(angle_step)
    nb_tri_axial = int(np.ceil(LA / L_edge_circ))
    axial_step = LA / nb_tri_axial

    print("  nb_tri_circ : "+str(nb_tri_circ))
    print("  nb_tri_axial: "+str(nb_tri_axial))
    print("  L_edge_circ (mm) : "+str(L_edge_circ*1000.))
    print("  axial_step (mm)  : "+str(axial_step*1000.))

    # - Nodes generation

    node_ext, node_int = [list(), list()] # External and internal skins

    P = {'X' : [2, 0, 1], 'Y' : [1, 2, 0], 'Z' : [0, 1, 2] }
    PA = P[Axe_cone]

    beta = (R1e-R1b) / LA
    zz = Ab
    for iax in range(nb_tri_axial+1) : 
        for jcir in range(nb_tri_circ) :
        
            RR = R1b + (zz-Ab) * beta
            XYZ = [RR*np.cos(jcir*angle_step), RR*np.sin(jcir*angle_step), zz]
            node_ext.append([XYZ[PA[ii]] for ii in range(3)])
    
            XYZ = [R0*np.cos(jcir*angle_step), R0*np.sin(jcir*angle_step), zz]
            node_int.append([XYZ[PA[ii]] for ii in range(3)])
    
        zz += axial_step
   
    node = node_ext+node_int

    nb_nod_cyl = nb_tri_circ * (nb_tri_axial+1) # = len(node_ext) = len(node_int)

    # - Triangles generation (external and internal skins)

    tri = list()
    nc = nb_tri_circ
    for ia in range(nb_tri_axial) : 
        for jc in range(nc) :
            tri.append([nc*ia+jc, nc*ia+(jc+1)%nc, nc*(ia+1)+jc])
            tri.append([tri[-1][0]+nb_nod_cyl, tri[-1][2]+nb_nod_cyl, tri[-1][1]+nb_nod_cyl])

            tri.append([nc*ia+(jc+1)%nc, nc*(ia+1)+(jc+1)%nc, nc*(ia+1)+jc])
            tri.append([tri[-1][0]+nb_nod_cyl, tri[-1][2]+nb_nod_cyl, tri[-1][1]+nb_nod_cyl])

    # - Triangles generation (begin and end disks)

    idx =  nb_nod_cyl

    dec = nb_tri_circ * nb_tri_axial
    for jc in range(nc) :
        # begin
        tri.append([ (jc+1)%nc     , jc            , jc+nb_nod_cyl        ])
        tri.append([ tri[-1][0]+dec, tri[-1][2]+dec, tri[-1][1]+dec       ])
        # end
        tri.append([ (jc+1)%nc     , jc+nb_nod_cyl , (jc+1)%nc+nb_nod_cyl ])
        tri.append([ tri[-1][0]+dec, tri[-1][2]+dec, tri[-1][1]+dec       ])

    # - tri mesh file writing

    node = np.array( node, dtype=np.float64, order='C')
    tri = np.array( tri, dtype=np.uint32, order='C')

    tri_wp_fn = os.path.join('in','tri_wp')
    file_tri_wp = open(tri_wp_fn,'wb')
    tri_mesh = [ [node, tri] ]
    pickle.dump(tri_mesh,file_tri_wp)
    file_tri_wp.close()        
# ==============================================================================


db_def=[{
    'bloc_id': 1,
    'axis': 0,
    'w_min': w_min,
    'w_max': w_max,
    'o_u': -R_ext_2,
    'o_v': -R_ext_2,
    'd_u': d_u,
    'd_v': d_v,
    'tab_size':[[nb_mcr_dxl_u, nb_mcr_dxl_v],[nb_dxl_per_mcr_dxl_u,nb_dxl_per_mcr_dxl_v]],
    'max_old_untouched_ln': 1000,
    },  
    ]

for db in db_def : 
    if use_truc_func :
        db['truc_func'] = {'h_min':h_min_1, 'h_max':h_max_truncado}
    else : 
        db['tri_fn'] = 'tri_wp'
