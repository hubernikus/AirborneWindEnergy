"""
First Aircraft simulation

@author lukas huber
@date 2017-12-04

"""
# Automatically reload libraries, type in ipython shell:
#%load_ext autoreload
#%autoreload 2

## ----- Import Libraries ##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 3D Animation utils
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
#from matplotlib.patches import

import csv

from casadi import * # casadi library

import yaml # import yaml files

# Add path to local libraries
import sys
sys.path.append('./model/')
sys.path.append('./lib/')
sys.path.append('./') # TODO: change and move equilibrium search to library folder..., below,ooo 

# Local libraries
from kite_sim import *
from quatlib import *
from controllib import *
from visual_lib import *
from equilibriumSearch_steadyMotion import longitudinalFlight

# Direcotries
yamlDir = '../steadyState_modes/'
modelDir = './model/'
csvDir = '../simulationResults/'

##  --------- SIMULATION PARAMETERS ------------ 
# Simulation Time Parameters
t_start = 0
t_final = 10
dt = 0.01

# Visualization update interval (default 200 [miliseconds])
animationInterval = 10

# motion: 'linear', 'circular', 'MPC_simu', 'screw'
motionType = 'circular'

# Choose sort of visualization  -----  '2' - 2D ; '3' - 3D
visual = 3

# Choose sort of control ---- 'None', 'LQR', 'steadyStatePrediction' TODO: PID, nonlinear, MPC, etc.
#control  = 'LQR'
control  = 'None'
#control  = 'steadyStatePrediction'

# Dimension of LQR
if control  == 'steadyStatePrediction':
    linearSystemDim = 10 # DON'T TOUCH!!!
else:# Dimension of linear system and controlle [6, 9, 10, 13]
    linearSystemDim = 10
    
# Simulation output name to save to file
#simuName = 'None'
#simuName = 'linearSteadyLevel'
#simuName = 'lnearAscending'
#simuName = 'lnearDescending'
simuName = 'circular_2'
#simuName = 'MPC_simuluation'


if not(simuName =='None'):
    csvFile = open(csvDir + 'numeric_'+ simuName + '.csv', 'w')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['time','velolicity0','v1','v2', 'angularRate0','a1', 'a2', 'position0','p1','p2', 'quaternion0','q1','q2','q3', 'thrust', 'elevator' , 'rudder'])
    
# Simulation parameters
parameters = dict()
parameters['simulation']= 0
parameters['plot'] = 0
parameters['vr'] = 0
parameters['int_type'] = 'cvodes'
parameters['t_span'] = [t_start, t_final, dt]


## ------------------------------------------
# Physicial Limits of Controller

T_lim = [0, 0.3] # NewtonX
dE_lim = [-10/180*pi, 10/180*pi] # rad
dR_lim = [-10/180*pi, 10/180*pi] # rad


## --------------------------------------
# Import Initial Position and Control

if motionType=='linear':
    #with open(yamlDir + 'steadyState_longitudial_steadyLevel.yaml') as yamlFile:
    #with open('steadyState_longitudial.yaml') as yamlFile:
    #with open(yamlDir+ 'steadyState_longitudinal_gamma-30deg.yaml') as yamlFile:
    with open(yamlDir + 'steadyState_longitudinal_gamma15deg.yaml') as yamlFile:
        initCond = yaml.safe_load(yamlFile)
    vel0 = initCond['vel']
    alpha0 = initCond['alpha']
    elevator = initCond['dE']

    thrust = initCond['T']

    gamma = initCond['gamma']

    angRate0 = [0, 0, 0]
    x0 = [-3, 0, 0]

    euler0 = [0,alpha0+gamma,0]
    euler0 = [0,alpha0+gamma,0]
    
    quat0 = eul2quat(euler0)

    rudder = 0
else:# circular or MPC_simu
    if motionType=='circular':
        #with open( yamlDir + 'steadyCircle3.yaml') as yamlFile:
        #with open(yamlDir + 'circle_gammadeg.yaml') as yamlFile:
        with open( yamlDir + 'circle_gamma0deg_vel10_rad5.yaml') as yamlFile:
            initCond = yaml.safe_load(yamlFile)
            
        quat0 = initCond['quat']

        gamma = initCond['gamma'] # inclination
        print(gamma)
        
    elif motionType=='MPC_simu':
        with open(yamlDir + 'steadyCircle_simuMPC.yaml') as yamlFile:
            initCond = yaml.safe_load(yamlFile)
        motionType = 'circular' # circular motion was imulated
        quat0 = initCond['quat']

        gamma = 0

        # rotate to align with circle
        rotAng = eul2quat([0,0,-4*pi/16])
        quat0 = quatmul(rotAng, quat0)
        quat0 = [quat0[i] for i in range(4)]

    thrust = initCond['T']
    vel0 = initCond['vel']
    vel0 = [float(vel0[i]) for i in range(len(vel0))]
    angRate0 =  initCond['angRate']

    x0 = initCond['pos']
    posCenter = initCond['centerPos']
    
    rudder = initCond['dR']

    trajRad = initCond['radius']

# State: [velocity (BRF), angular rates (BRF), position (IRF), quaternions (IRF-BRF)]
#parameters['x0'] = [1.5,0,0,0,0,0,0,0,3,1,0,0,0]
print('Initial Conditions')
print('Velocity', vel0)
print('Angular Rate:', angRate0)
print('Position:', x0)
print('Quaternions:',quat0)
print('')

parameters['x0'] = vel0 + angRate0 +  x0 +  quat0

# Steady Control input
elevator = initCond['dE']

# Control: [Thrust, Elevevator, Rudder]
parameters['u0'] = [thrust, elevator, rudder]
print('Thrust', 'Elevattor', 'Rudder')
print(parameters['u0'])

dimStates = 13
dimController = 3

## -------  Algeabraic equation for System dynamics using CasADi ----
num, flogg, sym = kite_sim(parameters)
integrator_num = num['INT']

# Default input
U0 = DM(parameters['u0'])
X0 = np.matrix(parameters['x0'])
X0 = X0.transpose()

# global variables
state = [DM(X0)]

# Interpret parameters
q_rotX = eul2quat([pi,0,0])
vel = [state[-1][0:3]]
angRate = [state[-1][3:6]]
x = [state[-1][6:9]]
quat = [state[-1][9:13]]
eul = [quat2eul(quat[-1])]

print('Control with [T, dE, dR]', U0)

time = [0]

## ------------- Set up linearized system -----------------  
if control == 'None':
    K = np.zeros((3,3))
else:
    A,B, A0, B0 = linearizeSystem(sym, X0, U0, linearSystemDim)
    
    #C = np.hstack(( np.zeros((3,6)), np.eye((3)), np.zeros((3,4)) ))
    controllable = checkControllabily_lin(A,B)
    notStabilizable = checkStabilizability_lin(A,B)
    
    print('Number of non-controllable states', controllable)
    print('Not stabilizable states', notStabilizable)
    
    if control == 'LQR':
        # minimize J_bar = sum_k0^inf (x_k^T Q x_k) + (u_k^T R u_k)
        diagQ = SX([10,100,100,        # Velocity
                    10, 10, 10,       # Angular Rate
                    0.1,0.1,0.1,      # Position
                    10,100,100,100]) # quaternion

        if linearSystemDim == 10:
            Q = diag( vertcat(diagQ[0:6], diagQ[9:13] ) )
        else:
            Q = diag(diagQ[0:linearSystemDim])
            
        R = diag(SX([10,1000,100]))

        # Calculate control
        K, X,  eigVals = lqr(A,B,Q,R)
        print('Eigenvalues of closed loop:', eigVals)

    elif control == 'PolePlace':
        K = PolePlace(A,B)
    elif control == 'steadyStatePrediction':
        K = 0
        
## -------------------- Set up visualization --------------------
if visual == 2:
    fig, ax, ax_x, ax_phi, ax_v, ax_omega, line_x, line_y, line_z, line_pitch, line_roll, line_yaw, line_vx, line_vy, line_vz, line_pRate, line_rRate, line_yRate = initFigure_2d()
elif visual == 3:
    ax_3d, fig = initFigure_3d()

# Save predicited position
posPred, predLine = draw_posPred(motionType, x0, vel0, quat0, gamma, trajRad, posCenter, ax_3d)
with  open(csvDir + 'numeric_'+ simuName + '_posPred' + '.csv', 'w') as csvFile_posPred:
    csvWriter_posPred = csv.writer(csvFile_posPred)
    for i in range(len(predLine[0])):
        tempRow = []
        for dim in range(3):
            tempRow.append(predLine[dim][i])
        csvWriter_posPred.writerow(tempRow)


if not(simuName == 'None'):
    csvRow = [str(time[-1])]
    temp_csvRow = [time[-1]]

    for i in range(dimStates):
        csvRow.append(str(state[-1][i]))
        temp_csvRow.append(state[-1][i])

    for i in range(dimController):
        csvRow.append(str(U0[i]))
        temp_csvRow.append(U0[i])

    csvWriter.writerow(csvRow)

# Initialization

##  --- Functions ---
def init2D():
    ax_x.set_ylim(-20, 20)
    ax_x.set_ylabel('Position')
    ax_x.set_xlim(t_start, t_final)
    ax_x.legend()

    ax_phi.set_ylim(-pi*2, pi*2)
    ax_phi.set_ylabel('Attitude')
    ax_phi.set_xlim(t_start, t_final)

    ax_v.set_ylim(-5, 2)
    ax_v.set_ylabel('Velocity [Error]')
    ax_v.set_xlim(t_start, t_final)

    ax_omega.set_ylim(-7, 7)
    ax_omega.set_ylabel('Angular Rate [Error]')
    ax_omega.set_xlim(t_start, t_final)
    
    return line_y, line_z, line_x, line_pitch, line_yaw, line_roll, line_vx, line_vy, line_vz, line_pRate, line_rRate, line_yRate
    
def predictStat(state, state0,  gamma, desiredTrajRad, posCenter, dt, t):
    state = np.array(state)

    vel = state[0:3]
    angRate = state[3:6]
    x = state[6:9]
    q = state[9:13]
    
    state0 = np.array(state0)
    
    vel0 = state0[0:3]
    angRate0 = state0[3:6]
    x0 = state0[6:9]
    q0 = state0[9:13]

    posCenter = np.array([posCenter])
    posCenter = posCenter.transpose()
        
    vel_I = np.array([quatrot(vel[:,0],q[:,0])])
    vel_I = vel_I.T
            
    #vel_I = vel
    
    x = x + vel_I*dt*100 # move one time step
    
    draw_aimingPositions(state, [x], ax_3d)

    if not(motionType == 'linear'):
        dPhi = 30/180*pi
        
        relPos = x-posCenter # Relative position to center
        relPos0 = x0 - posCenter
        
        rotZ = atan2(relPos[1], relPos[0]) # Rotation around Z 
        rotZ0 = atan2(relPos0[1], relPos0[0]) 
        
        delta_qZ = eul2quat([0,0,rotZ-rotZ0]) # rotation from deviation from 0 pos

        angRateInt = eul2quat(angRate0*dt) # rotation from angular rate

        #anreRateInt = quatrot(angRateInt,quatinv(q0))
        angRateInt = np.array([angRateInt])
        angRateInt = angRateInt.transpose()
        delta_qZ = np.array([delta_qZ])
        delta_qZ = delta_qZ.transpose()

        q = delta_qZ * angRateInt * q0 # rotate initial quaternion

        # Project x onto trajection radius
        if motionType == 'circular':
            x[2] = x0[2]
        else:
            x[2] = x0[2] + vel0[2]*t

        actualTrajRad = sqrt(relPos[0]**2 + relPos[1]**2)
        x[0:2] = x[0:2]*desiredTrajRad/actualTrajRad

        #r_vec = [desiredTrajRad, 0, 0]
        #x[0:2] = quatrot()

    
    return x, q

def applyControl(control, state, time, gamma, trajRad, posCenter, dt, K, X0, U0):
    x_pred = 0
    u = 0
    
    if control == 'steadyStatePrediction':
        #dt = dt*50
        x_pred, q_pred = predictState(state[-1], state[0],
                                      gamma, trajRad, posCenter, dt, time[-1])
        dX = x_pred - state[-1][6:9]

        yaw = -atan2(dX[1], dX[0])
        gamma = atan2(dX[2],sqrt(dX[0]*dX[0] + dX[1]*dX[1]) ) 

        # approximation with straight line flight no rudder
        dR =0
 
        # choose one variable, here Elevator
        dE = np.array([0])
        
        # Calculate the steady state trajectory which leads to the point
        steadyState_pred = longitudinalFlight(dE, gamma)

        # Import values
        U0 = [float(steadyState_pred['T']), float(steadyState_pred['dE']), dR]
        
        vel = steadyState_pred['vel']
        vel = [float(vel[i]) for i in range(3)]
        angRate = [0,0,0] 
        pos = state[-1][6:9]
        pos = [float(pos[i]) for i in range(3)]
        quat = eul2quat([0,steadyState_pred['gamma'], yaw])
        
        X0 = vel + angRate + pos + quat
        
        #import pdb; pdb.set_trace() ## DEBUG ##

        A,B, A0, B0 = linearizeSystem(sym, X0, U0, linearSystemDim)
    
        #C = np.hstack(( np.zeros((3,6)), np.eye((3)), np.zeros((3,4)) ))

        controllable = checkControllabily_lin(A,B)
        notStabilizable = checkStabilizability_lin(A,B)

        if (controllable):
            print('Number of non-controllable states', controllable)
        if len(notStabilizable):
            print('Not stabilizable states', notStabilizable)

        diagQ = SX([10,10,100,        # Velocity
                    1, 1, 1000,       # Angular Rate
                    0.1,0.1,0.1,      # Position
                    1000,1000,1000,1000]) # quaternion
        
        if len(A) == 10:
            Q = diag( vertcat(diagQ[0:6], diagQ[9:13] ) )
        else:
            Q = diag(diagQ[0:linearSystemDim])
            
        #Q = diag(diagQ[0:linearSystemDim])
        
        R = diag(SX([10,1000,100]))

        # Calculate control
        K_i, X,  eigVals = lqr(A,B,Q,R)
        #print('Eigenvalues of closed loop:', eigVals)

        x_pred, q_pred = predictState(state[-1], state[0],
                                          gamma, trajRad, posCenter, dt, time[-1])

        x_k = state[-1]
        X0[9:13] = q_pred[:,0] 

        u =  -K_i*(np.vstack((x_k[0:6],x_k[9:13]))-np.array([np.hstack((X0[0:6],X0[9:13]))]).T )  # apply control law to first states
                
        #u = u1 + U0
        u = u + np.array(([U0])).T

        u = saturateControl(u)
        checkControlRange(u)

        print('[T, dE, dR] - ', u.T)
        
        out = integrator_num(x0=state[-1], p=u)

    elif control == 'LQR':
        x_k = np.matrix(state[-1])
        if K.size/3 == 13:
            x_pred, q_pred = predictState(state[-1], state[0],
                                          gamma, trajRad, posCenter, dt, time[-1])
            
            x_k[6:9] = x_pred
            x_k[9:13] = q_pred
            
            u =  -K*(x_k-X0)  #apply control law to first states

        elif K.size/3 == 9 :
            x_pred, q_pred = predictState(state[-1], state[0],
                                          gamma, trajRad, posCenter, dt, time[-1])

            x_k[6:9] = x_pred
            
            u =  -K*(x_k[0:9]-X0[0:9])  # apply control law to first states

        elif K.size/3 == 10:
            x_pred, q_pred = predictState(state[-1], state[0],
                                          gamma, trajRad, posCenter, dt, time[-1])

            x_k[9:13] = q_pred

            u =  -K*(np.vstack((x_k[0:6],x_k[9:13]))-np.vstack((X0[0:6],X0[9:13])) )  # apply control law to first states

        else :
            u =  -K*(x_k[0:6]-X0[0:6])  # apply control law to first states
            
        u = u + U0
        
        checkControlRange(u)
        u = saturateControl(u)
        
        print('Control [T,dE,dR]', u) # Clear frame
        out = integrator_num(x0=state[-1], p=u)

    else:
        print('Chosen control {} not defined'.format(control))
        return 0
    
    return out, x_pred, u


def update3d_aircraft(frame):
    ax_3d.clear()

    dt = frame

    u = U0
    if control == 'None':
        out = integrator_num(x0=state[-1], p=U0)
    else:
        out, x_pred, u = applyControl(control, state, time, gamma, trajRad, posCenter, dt, K, X0, U0)
        if control == 'steadyStatePrediction':
            posAim = draw_aimingPositions(state[-1], [x_pred], ax_3d)
        elif K.size/3 !=6:
            posAim = draw_aimingPositions(state[-1], [x_pred], ax_3d)

    time.append(time[-1]+dt) # update time vector
    state.append(out['xf'])
    
    vel.append(state[-1][0:3])
    angRate.append(state[-1][3:6])
    x.append(state[-1][6:9])
    quat.append(state[-1][9:13])
    
    # Draw current airplane iter = -1
    planeBody, wingSurf, tailSurf,  planeTailHold = drawPlane3D(-1, x, quat[-1], ax_3d)

    # Draw starting plane -- iter=0 (TODO: change color?)
    planeBody, wingSurf, tailSurf,  planeTailHold = drawPlane3D(0, x, quat[0], ax_3d)
    
    # Draw history of CM
    posHistory, = ax_3d.plot([x[i][0] for i in range(len(x))],
                             [x[i][1] for i in range(len(x))],
                             [x[i][2] for i in range(len(x))],
                             'k--', linewidth=1)

    if not(control == 'None'):
        posPred = draw_posPred(motionType, x0, vel0, quat0, gamma, trajRad, posCenter, ax_3d)

    if not(simuName == 'None'):
        csvRow = [str(time[-1])]
        temp_csvRow = [time[-1]]

        for i in range(dimStates):
            csvRow.append(str(state[-1][i]))
            temp_csvRow.append(state[-1][i])

        for i in range(dimController):
            csvRow.append(str(u[i]))
            temp_csvRow.append(u[i])

        csvWriter.writerow(csvRow)

        

    setAxis_3d(ax_3d)
    
    #return planeBody, wingSurf, tailSurf, planeTailHold, posHistory , posPred
    
def update_aircraft(frame):
    dt = frame

    # Simulation step
    if control == 'None':
        out = integrator_num(x0=state[-1], p=U0)
    else:
        out, x_pred = applyControl(control, state, time, gamma, trajRad, posCenter, dt, K, X0, U0)
        if control == 'steadyStatePrediction':
            posAim = draw_aimingPositions(state[-1], [x_pred], ax_3d)
        elif K.size/3 !=6:
            posAim = draw_aimingPositions(state[-1], [x_pred], ax_3d)
    
    state.append(out['xf'])

    time.append(time[-1]+dt) # update time vector

    vel.append(state[-1][0:3])
    angRate.append(state[-1][3:6])
    x.append(state[-1][6:9])
    quat.append(state[-1][9:13])

    eul.append(quat2eul(quat[-1]))

    # Draw to plot
    line_x.set_data(time,[x[i][0] for i in range(len(x))])
    line_y.set_data(time,[x[i][1] for i in range(len(x))])
    line_z.set_data(time,[x[i][2] for i in range(len(x))])

    line_pitch.set_data(time, [eul[i][0] for i in range(len(eul))])
    line_roll.set_data(time, [eul[i][1] for i in range(len(eul))])
    line_yaw.set_data(time, [eul[i][2] for i in range(len(eul))])

    line_vx.set_data(time, [vel[i][0]-vel0[0] for i in range(len(eul))])
    line_vy.set_data(time, [vel[i][1]-vel0[1] for i in range(len(eul))])
    line_vz.set_data(time, [vel[i][2]-vel0[2] for i in range(len(eul))])
    
    line_pRate.set_data(time, [angRate[i][0]-angRate0[0] for i in range(len(eul))])
    line_rRate.set_data(time, [angRate[i][1]-angRate0[1] for i in range(len(eul))])
    line_yRate.set_data(time, [angRate[i][2]-angRate0[2] for i in range(len(eul))])

    return line_y ,line_z, line_x, line_pitch, line_yaw, line_roll, line_vx, line_vy, line_vz, line_pRate, line_rRate, line_yRate

#def update_aircraft1(frame):
#    return 0

# ----------------- Simulation starts here------------
if visual == 2:
    ani = FuncAnimation(fig, update_aircraft, frames=np.ones(int((t_final-t_start)/dt))*dt, init_func=init2D, blit=True)

elif visual == 3:
    ani = FuncAnimation(fig, update3d_aircraft, interval=animationInterval, frames=np.ones(int((t_final-t_start)/dt))*dt, blit=False) # no init call !?

#csvFile.close()

#ani = FuncAnimation(fig, update_limitCycle, frames=np.ones(int((t_final-t_start)/dt))*dt,
                    #init_func=init, blit=True)
plt.show()

print('Python finished') 
 
