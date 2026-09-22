import numpy as np

data_file = os.path.join('in','mdl_data.py')

exec(open(data_file).read())

f1 = 250.
f2 = 150.

xsi1 = 0.012
xsi2 = 0.01

k1 = 2.26e8
k2 = 2.13e8

theta_1 =  60. * np.pi/180.
theta_2 = 135. * np.pi/180.

omega1 = f1 * 2. * np.pi
omega2 = f2 * 2. * np.pi

vp1 = omega1**2.
vp2 = omega2**2.

m1 = k1 / vp1
m2 = k2 / vp2

                                                                                                                                                                                                                                                           
phi1_x, phi1_z = (np.cos(theta_1)/m1**0.5, np.sin(theta_1)/m1**0.5)
phi2_x, phi2_z = (np.cos(theta_2)/m2**0.5, np.sin(theta_2)/m2**0.5)
#phi1_y, phi1_z = (-np.sin(theta_1), np.cos(theta_1))
#phi2_y, phi2_z = (np.sin(theta_2), np.cos(theta_2))

print("******************")
print(" m1     : "+str(m1))
print(" m2     : "+str(m2))
print(" phi1_z : "+str(phi1_z))
print(" phi2_z : "+str(phi2_z))
print("******************")


dyn_tool_def = {
    'fe-type':'beam',
    'eigs_val_ref': [vp2 ],
    'eigs_xsi_ref':[xsi2],
    'list_eigs_vec_ref':
    #   U___    V_____  W_________  RX____  RY____  RZ_______
   [
    # eig 1
    #[ [  0.0  ,  0.0 , phi1_z,   0.0  , 0.0   , 0.0      ]],  #NODE 1
      [ [  0.  ,  0.0 , phi2_z,   0.0  , 0.0   , 0.0      ]],  #NODE 1
   ],
    'list_pnt_ref': [ [0.,0.0,0.0] ], # coordinates of nodes 0 and 1
    #'list_r_plot': [],
    #'list_id_node_ref_plot': [],
    'list_r_plot': [R_ext_2],
    'list_id_node_ref_plot': [0],
    'sym_eigs':False,
    'nb_eigs_use': 1 }

