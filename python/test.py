# -*- coding: utf-8 -*-
"""###########################################################################
#
#                   kite validation script
#                     klh   
#
########################################################################### """

print('Kite validation started')
# clear all; ## To ensure clean workspace
##
# clc; close all; clear variables;

# Import needed libraries
import numpy as np # Basic math library
import yaml # import yaml files

from sys import path

# Import ReadYalm
path.append("/home/lukas/Software/casadi/casadi-py35-np1.9.1-v3.2.3-64bit")
from casadi import *

# Personal functions
from kite_sim import kite_sim

# CLASS definitions 
class ParameterContainer:
  simulation = 0
  plot = 0
  vr = 0 #list cannot be initialized here!
  int_type = ''
  t_span = ''
  x0 = ''
  u0 = ''

#create kite model
#aircraft = yaml.read('umx_radian.yaml');

YALM_stream = open('umx_radian.yaml')
aircraft = yaml.load_all(YALM_stream)
#for doc in docs:
#    for k,v in doc.items():
#        print( k, "->", v, '\n')
#    print("\n")

parameters = ParameterContainer()

parameters.simulation = 0
parameters.plot = 0
parameters.vr = 0
parameters.int_type = 'cvodes'

start_sample = 10
# s_time = exp_telemetry(start_sample,end)
# f_time = exp_telemetry(end,end)
# parameters.t_span = [s_time, f_time, dt]
# parameters.x0 = exp_telemetry(start_sample,1:13)'
parameters.u0 = [0,0,0]


## Start simulation
[num, flog, sym] = kite_sim(aircraft, parameters)



print('Kite validation finished')