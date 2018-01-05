"""
First Aircraft simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add casadi to path
from casadi import * # casadi library

# Read  yaml to path
import yaml # import yaml files
with open('umx_radian.yaml') as yamlFile:
    aircraft = yaml.safe_load(yamlFile)

# Local
from kite_sim import *
from quatlib import *

# Time span
t_start = 0
t_final = 5
dt = 0.1

# Simulation parameters
parameters = dict()
parameters['simulation']= 0
parameters['plot'] = 0
parameters['vr'] = 0
parameters['int_type'] = 'cvodes'
parameters['t_span'] = [t_start, t_final, dt]

# Import Initial Position and Control
with open('steadyCircle.yaml') as yamlFile:
#with open('steadyState_longitudial_steadyLevel.yaml') as yamlFile:
    initCond = yaml.safe_load(yamlFile)

# State: [velocity (BRF), angular rates (BRF), position (IRF), quaternions (IRF-BRF)]
#parameters['x0'] = [1.5,0,0,0,0,0,0,0,3,1,0,0,0]
vel0 = initCond['vel']
angRate0 =  initCond['angRate']

x0 = initCond['pos']
quat0 = initCond['quat']

parameters['x0'] = vel0 + angRate0 +  x0 +  quat0

# Steady Control input
thrust = initCond['T']
elevator = initCond['dE']
rudder = initCond['dR']
# Control: [Thrust, Elevevator, Rudder]
parameters['u0'] = [thrust, elevator, rudder]

# Algeabraic equation for system dynamics using CasADi
num, flogg, sym = kite_sim(aircraft, parameters)
integrator_num = num['INT']

# global variables
state = [parameters['x0']]

# Interpret parameters
vel = [state[-1][0:3]]
angRate = [state[-1][3:6]]
x = [state[-1][6:9]]
quat = [state[-1][9:13]]
eul = [quat2eul(quat[-1])]

u0 = parameters['u0']

time = [0]

## Create Animated Plots
fig, ax = plt.subplots()
ax_x = plt.subplot(4,1,1)
line_x, = plt.plot([], [], 'r-', animated=True, label = 'x')
line_y, = plt.plot([], [], 'g-', animated=True, label = 'y')
line_z, = plt.plot([], [], 'b-', animated=True, label = 'z')

ax_phi = plt.subplot(4,1,2)
line_pitch, = plt.plot([], [], 'r-', animated=True)
line_roll, = plt.plot([], [], 'g-', animated=True)
line_yaw, = plt.plot([], [], 'b-', animated=True)

ax_v = plt.subplot(4,1,3)
line_vx, = plt.plot([], [], 'r-', animated=True)
line_vy, = plt.plot([], [], 'g-', animated=True)
line_vz, = plt.plot([], [], 'b-', animated=True)

ax_omega = plt.subplot(4,1,4)
line_pRate, = plt.plot([], [], 'r-', animated=True)
line_rRate, = plt.plot([], [], 'g-', animated=True)
line_yRate, = plt.plot([], [], 'b-', animated=True)


def init():
    ax_x.set_ylim(-5, 50)
    ax_x.set_ylabel('Position')
    ax_x.set_xlim(t_start, t_final)
    ax_x.legend()

    ax_phi.set_ylim(-pi*2.1, pi*2.1)
    ax_phi.set_ylabel('Attitude')
    ax_phi.set_xlim(t_start, t_final)

    ax_v.set_ylim(-10, 10)
    ax_v.set_ylabel('Velocity')
    ax_v.set_xlim(t_start, t_final)

    ax_omega.set_ylim(-pi*0.5, pi*0.5)
    ax_omega.set_ylabel('Angular Rate')
    ax_omega.set_xlim(t_start, t_final)

    return  line_x, line_y, line_z 

def update_aircraft(frame):
    dt = frame
    time.append(time[-1]+dt) # update time vector
        
    # Simulation step
    print('x:', x[-1], '  vel:', vel[-1])
    print('eulerAngles:', eul[-1], '  angRate:', angRate[-1])
    print('quaternions:', quat[-1])
    
    print(u0)
    out = integrator_num(x0=state[-1], p=u0)
    state.append(out['xf'])
        
    vel.append(state[-1][0:3])
    angRate.append(state[-1][3:6])
    x.append(state[-1][6:9])
    quat.append(state[-1][9:13])

    eul.append(quat2eul(quat[-1]))

    # Draw to plot
    line_x.set_data(time,[x[i][0] for i in range(len(x))])
    line_y.set_data(time,[x[i][1] for i in range(len(x))])
    line_z.set_data(time,[x[i][2] for i in range(len(x))])

    line_pitch.set_data(time, [angRate[i][0]-angRate0[0] for i in range(len(eul))])
    line_roll.set_data(time, [angRate[i][1]-angRate0[1] for i in range(len(eul))])
    line_yaw.set_data(time, [angRate[i][2]-angRate0[2] for i in range(len(eul))])

    #line_vx.set_data(time, [vel[i][0]-vel0[0] for i in range(len(eul))])
    #line_vy.set_data(time, [vel[i][1]-vel0[1] for i in range(len(eul))])
    #line_vz.set_data(time, [vel[i][2]-vel0[2] for i in range(len(eul))])
    
    line_vx.set_data(time, [vel[i][0] for i in range(len(eul))])
    line_vy.set_data(time, [vel[i][1] for i in range(len(eul))])
    line_vz.set_data(time, [vel[i][2] for i in range(len(eul))])

    line_pRate.set_data(time, [angRate[i][0] for i in range(len(eul))])
    line_rRate.set_data(time, [angRate[i][1] for i in range(len(eul))])
    line_yRate.set_data(time, [angRate[i][2] for i in range(len(eul))])

    return  line_x, line_z, line_y, line_pitch, line_yaw, line_roll, line_vx, line_vy, line_vz, line_pRate, line_rRate, line_yRate



#ani = FuncAnimation(fig, update_limitCycle, frames=np.ones(int((t_final-t_start)/dt))*dt,
                    #init_func=init, blit=True)
ani = FuncAnimation(fig, update_aircraft, frames=np.ones(int((t_final-t_start)/dt))*dt,
                    init_func=init, blit=True)

plt.show()
 

