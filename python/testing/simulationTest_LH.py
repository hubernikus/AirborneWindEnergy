######################################################################
#
#                   create kite model
#
#                   Lukas Huber 
#
######################################################################
print('Started casadi-test.py')

b# Add casadi to path
from casadi import * # casadi library


# Read  yaml to path
import yaml # import yaml files
with open('umx_radian.yaml') as yamlFile:
    aircraft = yaml.safe_load(yamlFile)

# Local
from kite_sim import * 

import time



start = time.time()

parameters = dict()

parameters['simulation']= 0
parameters['plot'] = 0
parameters['vr'] = 0
parameters['int_type'] = 'cvodes'

s_time = 0
f_time = 1
dt = 0.1
parameters['t_span'] = [s_time, f_time, dt]
parameters['x0'] = [1.5,0,0,0,0,0,-3,0,-2,0.7071,0,0,0.7071]
parameters['u0'] = [0,0,0]

num, flog, sym = kite_sim(aircraft, parameters)


end = time.time()
print('Simultion duration was ', end - start, 'seconds \n.')

#test
# #1 - dynamics
x0 = parameters['x0']
u0 = parameters['u0']
dynamics = num['DYN_FUNC']
res = dynamics(x0,u0) # x0 - state (13,1), u0 - control (3,1)
res1 = res
print('res1: ', res1)

# #2 - Jacobian
jacobian_num = num['DYN_JACOBIAN']
res = jacobian_num(x0, u0)
res2 = res[0]
print('res2: ', res2)

# #3 - Integrator
integrator_num = num['INT']

out = integrator_num(x0=x0, p=u0)
#res3 = full(out.xf)
res3 = out['xf']
print('res3: ', res3)


print('Finished casadi-test.py')
 
