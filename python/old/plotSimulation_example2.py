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

# Simulation parameters
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

num, fig, sym = kite_sim(aircraft, parameters)

x0 = parameters['x0']
u0 = parameters['u0']
dynamics = num['DYN_FUNC']
res = dynamics(x0,u0) # x0 - state (13,1), u0 - control (3,1)
res1 = res

# global variables
xx = [parameters['x0']]

x = [[1,1,1]]
u = [parameters['u0']]
t = [0]

# Create Animated Plots
fig, ax = plt.subplots()
ax_xy = plt.subplot(2,2,1)
point_xy, = plt.plot([], [], 'ko', animated=True)
line_xy, = plt.plot([], [], 'k-', animated=True)

ax_xz = plt.subplot(2,2,3)
point_xz, = plt.plot([], [], 'ko', animated=True)
line_xz, = plt.plot([], [], 'k-', animated=True)

ax_yz = plt.subplot(2,2,2)
point_yz, = plt.plot([], [], 'ko', animated=True)
line_yz, = plt.plot([], [], 'k-', animated=True)

def init():
    ax_xy.set_xlim(-2, 2)
    ax_xy.set_ylim(-2, 2)
    ax_xy.set_ylabel('Y')

    ax_xz.set_xlim(-2, 2)
    ax_xz.set_ylim(-2, 2)
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')

    ax_yz.set_xlim(-2, 2)
    ax_yz.set_ylim(-2, 2)
    ax_yz.set_xlabel('Z')

    return point_xy, line_xy, point_xz, line_xz, point_yz, line_yz

def update_limitCycle(frame):
    dt = frame

    # Simulation step
    dx = limitCycle_DS(x[-1])
    xEnd =  [x[-1][i]+dx[i]*dt for i in range(len(dx))]
    
    x.append(xEnd)

    # Draw to plot
    point_xy.set_data(x[-1][0], x[-1][1])
    line_xy.set_data([x[i][0] for i in range(len(x))],[x[i][1] for i in range(len(x))])

    point_xz.set_data(x[-1][0], x[-1][2])
    line_xz.set_data([x[i][0] for i in range(len(x))],[x[i][2] for i in range(len(x))])

    point_yz.set_data(x[-1][2], x[-1][1])
    line_yz.set_data([x[i][2] for i in range(len(x))],[x[i][1] for i in range(len(x))])
    
    return point_xy, line_xy, point_xz, line_xz, point_yz, line_yz

def update_aircraft(frame):
    dt = frame

    # Simulation step
    dx = limitCycle_DS(x[-1])
    xEnd =  [x[-1][i]+dx[i]*dt for i in range(len(dx))]
    
    x.append(xEnd)

    # Draw to plot
    point_xy.set_data(x[-1][0], x[-1][1])
    line_xy.set_data([x[i][0] for i in range(len(x))],[x[i][1] for i in range(len(x))])

    point_xz.set_data(x[-1][0], x[-1][2])
    line_xz.set_data([x[i][0] for i in range(len(x))],[x[i][2] for i in range(len(x))])

    point_yz.set_data(x[-1][2], x[-1][1])
    line_yz.set_data([x[i][2] for i in range(len(x))],[x[i][1] for i in range(len(x))])
    
    return point_xy, line_xy, point_xz, line_xz, point_yz, line_yz
    

def limitCycle_DS(x):
    c1 = 1;

    dx = []
    dx.append(-x[1] + c1*x[0]*(1-(x[0]**2 + x[1]**2)))
    dx.append(x[0] + c1*x[1]*(1-(x[0]**2 + x[1]**2)))
    dx.append(-x[2])

    return dx

# Simulation parameters
t_start = 0
t_final = 10
dt = 0.2


#ani = FuncAnimation(fig, update_limitCycle, frames=np.ones(int((t_final-t_start)/dt))*dt,
                    #init_func=init, blit=True)
ani = FuncAnimation(fig, update_aircraft, frames=np.ones(int((t_final-t_start)/dt))*dt,
                    init_func=init, blit=True)

plt.show()
