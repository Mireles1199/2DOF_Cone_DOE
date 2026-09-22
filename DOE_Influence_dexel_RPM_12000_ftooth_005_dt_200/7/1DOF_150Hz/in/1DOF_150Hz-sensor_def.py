
sensor_disp = { 'name': 'Axial_disp',
                'support': 'tool', 
                'location': [0.0, 0.0, 0.0], 
                'basis': 'tool',
                'direction': [0., 0.0, 1.0], 
                'type': 'DIS',               
                'ref_length' : 10e-3 }  
sensor_vel = { 'name': 'Axial_vel',
                'support': 'tool', 
                'location': [0.0, 0.0, 0.0], 
                'basis': 'tool',
                'direction': [0., 0.0, 1.0], 
                'type': 'VEL',             
                'ref_length' : 10e-3 }  

sensors = [ sensor_disp, sensor_vel ]

units = { 'ACC' : 'SI',  # or 'm.s-2', 'g'
          'VEL' : 'SI',  # or 'm.s-1', 'm.min-1'
          'DIS' : 'SI' , # or 'm', 'mm', 'micron'
          'FLEX' : 'SI' } # or 'm/N', 'm/kN', 'micron/N'