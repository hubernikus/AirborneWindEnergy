import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

import numpy as np

from quatlib import *

from math import sin, cos, tan


# Geometry for 3d viusalization
# Body geometry
dirBody_B = np.array([1,0,0])
dl1 = 1.2 # Length tail of plane
dl2 = 0.5 # length noise of plane
lWidth = 3 # thickness of body

# Wing geometry
dirWing_B = np.array([0, 1, 0])
wingSpan = 0.7
wingWidth = 0.4
wingPos = 0

# Tail geometry (-> direction is parallel as wing)
dirTail_B = np.array([0, 0, -1])
tailSpan = 0.3
tailWidth = 0.25
tailPos = -1.2
tailPosz = 0.5
# Prediction length
lPred = 15

# Horizontal limits for the 3d grid (x,y direction)
hLim0 = 10 # Boundaries of the Flying Machine Area


def  draw_posPred(motionType, x0, vel0, quat0, gamma, trajRad, posCenter, ax_3d):
    if  motionType=='linear':
        vel_I = quatrot(vel0, np.array(quat0))
        dVel = vel_I*lPred/np.linalg.norm(vel_I)
        posPred, = ax_3d.plot([x0[0], x0[0]+dVel[0]],
                              [x0[1], x0[1]+dVel[1]],
                              [x0[2], x0[2]+dVel[2]],
                               'r--')
        
    elif motionType =='circular': 
        #print('draw circle')
        N_circ = 20 # number of sample points
        
        #posCenter = [x0[0],x0[1]+trajRad,x0[2]] # TODO: more general center...
        dPhi = 2*pi/N_circ
        dZ = trajRad*dPhi*tan(gamma)

        if(gamma): # make to turns if there is a screw motion (gamma!=0)
            N_circ = 2*N_circ 
            
        xCirc = [trajRad*np.sin(dPhi*i)+posCenter[0] for i in range(N_circ+1)]
        yCirc = [trajRad*np.cos(dPhi*i)+posCenter[1] for i in range(N_circ+1)]
        zCirc = [posCenter[2]+dZ*i for i in range(N_circ+1)]
                                
        posPred, = ax_3d.plot(xCirc, yCirc, zCirc,'r--')
    else:
        print('prediction not defined')

    return posPred

def draw_aimingPoints(state, x_traj, ax_3d):
    x_aim = [state[6]]
    y_aim = [state[7]]
    z_aim = [state[8]]

    for i in range(len(x_traj)):
        x_aim.append(x_traj[i][0])
        x_aim.append(x_traj[i][1])
        x_aim.append(x_traj[i][2])
    
    posAim, = ax_3d.plot(x_aim, y_aim, z_aim,'g--o')
    
    return posAim


def initFigure_2d():
     ## Create Animated Plots - 2D
    fig, ax = plt.subplots() 
    ax_x = plt.subplot(4,1,1) #  Position
    line_x, = plt.plot([], [], 'r-', animated=True, label = 'x')
    line_y, = plt.plot([], [], 'g-', animated=True, label = 'y')
    line_z, = plt.plot([], [], 'b-', animated=True, label = 'z')

    ax_phi = plt.subplot(4,1,2) # Orientation    line_pitch, = plt.plot([], [], 'r-', animated=True)
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
    
    return ax_x, ax_phi, ax_v, ax_omega
    
def initFigure_3d():
    fig = plt.figure() 
    ax_3d = fig.add_subplot(111, projection='3d')
    
    return ax_3d, fig

def initFiugre_2d_axis(ax_x,ax_vel, ax_phi,ax_omega,  t_start, t_final):
    # todo
    return 1
    
def setAxis_3d(ax_3d, hLim = hLim0):
    # Set limits 3D-plot
    ax_3d.set_xlim(-hLim, hLim)
    ax_3d.set_ylim(-hLim, hLim)
    ax_3d.set_zlim(-20,10)
    ax_3d.invert_zaxis()
    ax_3d.invert_yaxis()
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')

    

def drawPlane3D(it, x, quat, ax_3d):
    
    # Draw airplane body
    q_IB =  [float(quat[i]) for i in range(4)]

    dBody = quatrot(dirBody_B, np.array(q_IB) ) # Direction of the plane
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
                      x[it][i]+tailPos*dBody[i]-tailSpan*dirWing[i]+dirTail[i]*tailPosz],
                     [x[it][i]+(tailWidth+tailPos)*dBody[i]+tailSpan*dirWing[i]+dirTail[i]*tailPosz,
                      x[it][i]+(tailWidth+tailPos)*dBody[i]-tailSpan*dirWing[i]+dirTail[i]*tailPosz]])
    i = 1
    Y_tail=np.array([[x[it][i]+tailPos*dBody[i]+tailSpan*dirWing[i]+dirTail[i]*tailPosz,
                      x[it][i]+tailPos*dBody[i]-tailSpan*dirWing[i]+dirTail[i]*tailPosz],
                     [x[it][i]+(tailWidth+tailPos)*dBody[i]+tailSpan*dirWing[i]+dirTail[i]*tailPosz,
                      x[it][i]+(tailWidth+tailPos)*dBody[i]-tailSpan*dirWing[i]+dirTail[i]*tailPosz]])
    i = 2
    Z_tail=np.array([[x[it][i]+tailPos*dBody[i]+tailSpan*dirWing[i]+dirTail[i]*tailPosz,
                      x[it][i]+tailPos*dBody[i]-tailSpan*dirWing[i]+dirTail[i]*tailPosz],
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

    
    return planeBody, wingSurf, tailSurf, planeTailHold
