"""
First Aircraft simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 3D Animation utils
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

# Add casadi to path
from casadi import * # casadi library

# Read  yaml to path
import yaml # import yaml files
with open('umx_radian.yaml') as yamlFile:
    aircraft = yaml.safe_load(yamlFile)

# Local libraries
from kite_sim import *
from quatlib import *


## SIMULATION PARAMETERS
# Time span
t_start = 0
t_final = 5
dt = 0.01

# motion: 'linear', 'circular'
motionType = 'circular'

# Choose sort of visualization
# '2' - 2D ; '3' - 3D
visual = 3

# Simulation parameters
parameters = dict()
parameters['simulation']= 0
parameters['plot'] = 0
parameters['vr'] = 0
parameters['int_type'] = 'cvodes'
parameters['t_span'] = [t_start, t_final, dt]

# Import Initial Position and Control
if motionType=='linear':
    #with open('steadyState_longitudial_steadyLevel.yaml') as yamlFile:
    with open('steadyState_longitudial.yaml') as yamlFile:
        initCond = yaml.safe_load(yamlFile)
    vel0 = initCond['vel']
    alpha0 = initCond['alpha']
    elevator = initCond['dE']
    thrust = initCond['T']
    gamma = initCond['gamma']

    angRate0 = [0, 0, 0]
    x0 = [-15, 0, 0]

    quat0 = [1, 0, 0, 0]
    euler0 = [0,alpha0+gamma, 0]
    print(euler0)
    eulDirect = eul2quat(euler0)
    quat0 = eulDirect
        
    parameters['x0'] = vel0 + angRate0 +  x0 +  quat0
    
    rudder = 0
    
elif motionType=='circular':
    with open('steadyCircle.yaml') as yamlFile:
        
        initCond = yaml.safe_load(yamlFile)
        
    vel0 = initCond['vel']
    angRate0 =  initCond['angRate']

    x0 = initCond['pos']
    x0 = [0,0,0]
    quat0 = initCond['quat']

    parameters['x0'] = vel0 + angRate0 +  x0 +  quat0

    rudder = initCond['dR']

# State: [velocity (BRF), angular rates (BRF), position (IRF), quaternions (IRF-BRF)]
#parameters['x0'] = [1.5,0,0,0,0,0,0,0,3,1,0,0,0]

# Steady Control input
thrust = initCond['T']
elevator = initCond['dE']

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

if visual ==2:
    ## Create Animated Plots - 2D
    fig, ax = plt.subplots() 
    ax_x = plt.subplot(4,1,1) #  Position
    line_x, = plt.plot([], [], 'r-', animated=True, label = 'x')
    line_y, = plt.plot([], [], 'g-', animated=True, label = 'y')
    line_z, = plt.plot([], [], 'b-', animated=True, label = 'z')

    ax_phi = plt.subplot(4,1,2) # Orientation
    line_pitch, = plt.plot([], [], 'r-', animated=True)
    line_roll, = plt.plot([], [], 'g-', animated=True)
    line_yaw, = plt.plot([], [], 'b-', animated=True)

    ax_v = plt.subplot(4,1,3) # Velocity
    line_vx, = plt.plot([], [], 'r-', animated=True)
    line_vy, = plt.plot([], [], 'g-', animated=True)
    line_vz, = plt.plot([], [], 'b-', animated=True)

    ax_omega = plt.subplot(4,1,4) # Angular Rate
    line_pRate, = plt.plot([], [], 'r-', animated=True)
    line_rRate, = plt.plot([], [], 'g-', animated=True)
    line_yRate, = plt.plot([], [], 'b-', animated=True)
    
elif visual == 3: # 3 Dimensional Plot
    fig=plt.figure() 
    ax_3d = fig.add_subplot(111, projection='3d')

    
    # Body geometry
    dirBody_B = np.array([1,0,0])
    dl1 = 0.7 # Length tail of plane
    dl2 = 0.3 # length noise of plane
    lWidth = 2 # thickness of body
    
    # Wing geometry
    dirWing_B = np.array([0, 1, 0])
    wingSpan = 0.35
    wingWidth = 0.22
    wingPos = 0

    # Tail geometry (-> direction is parallel as wing)
    dirTail_B = np.array([0, 0, 1])
    tailSpan = 0.15
    tailWidth = 0.15
    tailPos = -0.65
    tailPosz = 0.1

# Initialization
hLim = 10 # Boundaries of the Flying Machine Area
def init():
            
    ax_x.set_ylim(-20, 20)
    ax_x.set_ylabel('Position')
    ax_x.set_xlim(t_start, t_final)
    ax_x.legend()

    ax_phi.set_ylim(-pi*0.1, pi*0.1)
    ax_phi.set_ylabel('Attitude')
    ax_phi.set_xlim(t_start, t_final)

    ax_v.set_ylim(-30, 10)
    ax_v.set_ylabel('Velocity')
    ax_v.set_xlim(t_start, t_final)

    ax_omega.set_ylim(-pi*0.5, pi*0.5)
    ax_omega.set_ylabel('Angular Rate')
    ax_omega.set_xlim(t_start, t_final)

    return  line_x, line_y, line_z

def init3d():
    print('init useless...')
    #ax_3d.set_xlim(-hLim,hLim)
    #ax_3d.set_ylim(-hLim,hLim)
    #ax_3d.set_zlim(-5, 10)
    #ax_3d.set_xlabel('X')
    #ax_3d.set_ylabel('Y')

    
def update3d_aircraft(frame):
    dt = frame
    time.append(time[-1]+dt) # update time vector
    out = integrator_num(x0=state[-1], p=u0)
    state.append(out['xf'])
        
    vel.append(state[-1][0:3])
    angRate.append(state[-1][3:6])
    x.append(state[-1][6:9])
    quat.append(state[-1][9:13])

    # Clear frame
    ax_3d.clear()

    iter = -1
    planeBody, wingSurf, tailSurf = drawPlane(iter, quat[-1])

    planeBody, wingSurf, tailSurf = drawPlane(0, quat[0]) # keep first plane
    
    
    # Draw history of CM
    posHistory, = ax_3d.plot([x[i][0] for i in range(len(x))],
                             [x[i][1] for i in range(len(x))],
                             [x[i][2] for i in range(len(x))],
                             'k--', linewidth=1)

    if  motionType=='linear':
        vel_I = quatrot(vel0, np.array(quat0))
        dVel = vel_I*lPred/np.linalg.norm(vel_I)
        posPred, = ax_3d.plot([x0[0], x0[0]+dVel[0]],
                              [x0[1], x0[1]+dVel[1]],
                              [x0[2], x0[2]+dVel[2]],
                               'r--')
        
    elif motionType =='circular': # TODO implement circular trajectory
        vel_I = quatrot_inv(vel0, np.array(quat0))
        dVel = vel_I*lPred/np.linalg.norm(vel_I)
        posPred, = ax_3d.plot([x0[0], x0[0]+dVel[0]],
                              [x0[1], x0[1]+dVel[1]],
                              [x0[2], x0[2]+dVel[2]],
                               'r--')
        
    else:
        print('prediction not defined')

    # Set limits 3D-plot
    ax_3d.set_xlim(-hLim, hLim)
    ax_3d.set_ylim(-hLim, hLim)
    ax_3d.set_zlim(-10,20)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')

    return planeBody, wingSurf, tailSurf, posHistory, posPred
    
def update_aircraft(frame):
    dt = frame
    time.append(time[-1]+dt) # update time vector
        
    # Simulation step
    print('x:', x[-1], '  vel:', vel[-1])
    #print('eulerAngles:', eul[-1], '  angRate:', angRate[-1])
    #print('quaternions:', quat[-1])
    
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

    # #line_vx.set_data(time, [vel[i][0]-vel0[0] for i in range(len(eul))])
    # #line_vy.set_data(time, [vel[i][1]-vel0[1] for i in range(len(eul))])
    # #line_vz.set_data(time, [vel[i][2]-vel0[2] for i in range(len(eul))])
    
    line_vx.set_data(time, [vel[i][0] for i in range(len(eul))])
    line_vy.set_data(time, [vel[i][1] for i in range(len(eul))])
    line_vz.set_data(time, [vel[i][2] for i in range(len(eul))])

    line_pRate.set_data(time, [angRate[i][0] for i in range(len(eul))])
    line_rRate.set_data(time, [angRate[i][1] for i in range(len(eul))])
    line_yRate.set_data(time, [angRate[i][2] for i in range(len(eul))])


    return line_x ,line_z, line_y , line_pitch, line_yaw, line_roll, line_vx, line_vy, line_vz, line_pRate, line_rRate, line_yRate

def drawPlane(it, quat):
    # Draw airplane body
    q_IB =  [float(quat[i]) for i in range(4)]

    dBody = quatrot(dirBody_B, np.array(q_IB) )
    X_plane = [x[it][0]-dl1*dBody[0], x[it][0]+dl2*dBody[0]]
    Y_plane = [x[it][1]-dl1*dBody[1], x[it][1]+dl2*dBody[1]]
    Z_plane = [x[it][2]-dl1*dBody[2], x[it][2]+dl2*dBody[2]]
    
    planeBody, = ax_3d.plot(X_plane, Y_plane, Z_plane, 'k', linewidth = lWidth)

    
    # Draw Wing
    dirWing = quatrot(dirWing_B, np.array(q_IB))
    
    i = 0
    X_wing=np.array([[x[it][i]+wingPos*dBody[i]+wingSpan*dirWing[i], x[it][i]+wingPos*dBody[i]-wingSpan*dirWing[i]],
          [x[it][i]+(wingWidth+wingPos)*dBody[i]+wingSpan*dirWing[i], x[it][i]+(wingWidth+wingPos)*dBody[i]-wingSpan*dirWing[i]]])
    i = 1
    Y_wing=np.array([[x[it][i]+wingPos*dBody[i]+wingSpan*dirWing[i], x[it][i]+wingPos*dBody[i]-wingSpan*dirWing[i]],
          [x[it][i]+(wingWidth+wingPos)*dBody[i]+wingSpan*dirWing[i], x[it][i]+(wingWidth+wingPos)*dBody[i]-wingSpan*dirWing[i]]])
    i = 2
    Z_wing=np.array([[x[it][i]+wingPos*dBody[i]+wingSpan*dirWing[i], x[it][i]+wingPos*dBody[i]-wingSpan*dirWing[i]],
          [x[it][i]+(wingWidth+wingPos)*dBody[i]+wingSpan*dirWing[i], x[it][i]+(wingWidth+wingPos)*dBody[i]-wingSpan*dirWing[i]]])
    
    wingSurf = ax_3d.plot_surface(X_wing, Y_wing, Z_wing, color='k')

    # Draw Tail
    #dirWing = quatrot(dirWing_B, np.array(q_IB))
    dirTail = quatrot(dirTail_B, np.array(q_IB))
    i = 0
    X_tail=np.array([[x[it][i]+tailPos*dBody[i]+tailSpan*dirWing[i]+dirTail[i]*tailPosz,
                      x[it][i]+tailPos*dBody[i]-tailSpan*dirWing[i]]+dirTail[i]*tailPosz,
                     
                     [x[it][i]+(tailWidth+tailPos)*dBody[i]+tailSpan*dirWing[i]+dirTail[i]*tailPosz,
                      x[it][i]+(tailWidth+tailPos)*dBody[i]-tailSpan*dirWing[i]+dirTail[i]*tailPosz]])
    i = 1
    Y_tail=np.array([[x[it][i]+tailPos*dBody[i]+tailSpan*dirWing[i]+dirTail[i]*tailPosz,
                      x[it][i]+tailPos*dBody[i]-tailSpan*dirWing[i]]+dirTail[i]*tailPosz,
                     
                     [x[it][i]+(tailWidth+tailPos)*dBody[i]+tailSpan*dirWing[i]+dirTail[i]*tailPosz,
                      x[it][i]+(tailWidth+tailPos)*dBody[i]-tailSpan*dirWing[i]+dirTail[i]*tailPosz]])
    i = 2
    Z_tail=np.array([[x[it][i]+tailPos*dBody[i]+tailSpan*dirWing[i]+dirTail[i]*tailPosz,
                      x[it][i]+tailPos*dBody[i]-tailSpan*dirWing[i]]+dirTail[i]*tailPosz,
                     
                     [x[it][i]+(tailWidth+tailPos)*dBody[i]+tailSpan*dirWing[i]+dirTail[i]*tailPosz,
                      x[it][i]+(tailWidth+tailPos)*dBody[i]-tailSpan*dirWing[i]+dirTail[i]*tailPosz]])
    tailSurf = ax_3d.plot_surface(X_tail, Y_tail, Z_tail, color='k')

    # Draw Tail-holder
    i = 0
    X_tailHold=[x[it][i]+(tailWidth/2+tailPos)*dBody[i],
                x[it][i]+(tailWidth/2+tailPos)*dBody[i]+dirTail[i]*tailPosz]
    i=1
    Y_tailHold=[x[it][i]+(tailWidth/2+tailPos)*dBody[i],
                x[it][i]+(tailWidth/2+tailPos)*dBody[i]+dirTail[i]*tailPosz]
    i=2
    Z_tailHold=[x[it][i]+(tailWidth/2+tailPos)*dBody[i],
                x[it][i]+(tailWidth/2+tailPos)*dBody[i]+dirTail[i]*tailPosz]
    
    
    planeTailHold, = ax_3d.plot(X_tailHold, Y_tailHold, Z_tailHold, 'k', linewidth = lWidth)    
    return planeBody, wingSurf, tailSurf


# Simulation starts here
if visual == 2:
    ani = FuncAnimation(fig, update_aircraft, frames=np.ones(int((t_final-t_start)/dt))*dt,
                        init_func=init, blit=True)
elif visual == 3:
    ani = FuncAnimation(fig, update3d_aircraft, frames=np.ones(int((t_final-t_start)/dt))*dt,
                    init_func=init3d, blit=False)

#ani = FuncAnimation(fig, update_limitCycle, frames=np.ones(int((t_final-t_start)/dt))*dt,
                    #init_func=init, blit=True)
plt.show()
 


