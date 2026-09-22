import os

data_file = os.path.join('in','mdl_data.py')
exec(open(data_file).read())

vol_ref = 1.e9
fcoulomb = 0.2 # Coulomb friction coefficient
#
laws_def={
'domains': [ {'name': 'my_domain', 'material': 'Altintlas_Mat', 'db_id_list': [1] } ],
'macro_cls': { 'MCL_rf': { 'Altintlas_Mat': 'CL_Altintlas_Mat'} },
'cls':{
  'CL_Altintlas_Mat': [ \
      { 'name': 'MCL_rf',
          'type': 'rake', 
          'V_dir': 'spindle', 
          '(b,h)': 'ec', 
          'h_ref': 1.e-3,
          'n': {'model': 'pow' , 'K':  0., 'p' : 0. , 'K_0': 0 },  
          'e': {'model': 'pow' , 'K':  0., 'p' : 0. , 'K_0': 0 },
          'c': {'model': 'pow' , 'K':  -K_f, 'p' : 1. , 'K_0': 0 }  },
         ]
   },
}


