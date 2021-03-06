# -*- coding: utf-8 -*-
"""###########################################################################
#
#                   kite validation script
#                     klh   
#
########################################################################### """

# TO compile in EMACS: M-x compile RET python3 <<NAME>>
print('Kite validation started \n')

# Import needed libraries
import numpy as np # Basic math library

from sys import path
import matplotlib.pyplot as plt

# Import ReadYalm
#from sys import path
#path.append("/home/lukas/Software/casadi/casadi-py35-np1.9.1-v3.2.3-64bit")
from casadi import *

# Personal functions
from kite_sim import *

import pickle


import yaml # import yaml files
with open('umx_radian.yaml') as yamlFile:
    aircraft = yaml.safe_load(yamlFile)

parameters = dict()

parameters['simulation'] = 0
parameters['plot'] = 0
parameters['vr'] = 0
parameters['int_type'] = 'cvodes'

start_sample = 10
# s_time = exp_telemetry(start_sample,end)
# f_time = exp_telemetry(end,end)
# parameters['t_span'] = [s_time, f_time, dt]
# parameters['x0'] = exp_telemetry(start_sample,1:13)'
parameters['u0'] = np.array([0,0,0])

start_sample = 10
s_time = 0 #exp_telemetry(start_sample,end);
f_time = 1 #exp_telemetry(end,end);
dT = 0.1

#parameters['t_span'] = []
parameters['t_span'] = [s_time, f_time, dT]
parameters['x0'] = [1.5,0,0, 0,0,0, -3,0,-2.0, 0.7071, 0, 0, 0.7071] #exp_telemetry(start_sample,1:13)';
parameters['u0'] = [0.1,0,0]

## Start simulation
num, flog, sym = kite_sim(aircraft, parameters)

##

#print(repr(sym))


#file = open('test.txt','w')
#print >> file, 'whatever and more'
#file.close()

#file = open('../var_sim_py.txt', 'w' )
#pickle.dump(' asdf asdf ', file)
#file.write(repr(sym))
#pickle.dump(' asdf asdf ', file)
#file.close()

print('End test script. \n')
