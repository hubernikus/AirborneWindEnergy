"""
First Aircraft simulation

@author lukas huber
@date 2017-12-04

"""
## ----- Import Libraries ##
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
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
figDir = '../fig/'

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

time = []
vel = []
angRate = []
pos = []
quat = []
eul = []
u = []

# Simulation names
#simuName = 'linearSteadyLevel'
#simuName = 'lnearAscending'
#simuName = 'lnearDescending'
#simuName = 'circular'
simuName = 'circular_2'
#simuName = 'MPC_simuluation'


simuMax = 500

# Load predicted position
posPred = [[],[],[]]
with  open(csvDir + 'numeric_'+ simuName + '_posPred' + '.csv', 'r') as csvFile:
    csvReader = csv.reader(csvFile)
    for row in csvReader:
        posPred[0].append(float(row[0]))
        posPred[1].append(float(row[1]))
        posPred[2].append(float(row[2]))

it_import = 1        
# Load real file        
with open(csvDir + 'numeric_'+ simuName + '.csv', 'r') as csvFile:
    next(csvFile)
    csvReader = csv.reader(csvFile)
    #csvReader.next()
    for row in csvReader:
        time.append(float(row[0]))
        
        vel.append([])
        for i in range(3):
            vel[-1].append(float(row[1+i]))
        #i = 2
        #vel[-1].append(float(row[1+i])*(-1))
            
        angRate.append([])
        for i in range(3):
            angRate[-1].append(float(row[4+i]))
            
        pos.append([])
        for i in range(3):
            pos[-1].append(float(row[7+i]))
        #i = 2
        #pos[-1].append(float(row[7+i])*(-1))

        quat.append([])
        for i in range(4):
            quat[-1].append(float(row[10+i]))

        eul.append(quat2eul(quat[-1]))

        u.append([])
        for i in range(3):
            u[-1].append(float(row[14+i]))
            
        if(it_import > simuMax):
            break
        else:
            it_import += 1

simuNum = min(len(pos), simuMax)

nPlanes = 5



ax_3d, fig = initFigure_3d()
posPred, = ax_3d.plot(posPred[0],posPred[1],posPred[2],'r--')

posReal, = ax_3d.plot([pos[i][0] for i in range(simuNum)],
                      [pos[i][1] for i in range(simuNum)],
                      [pos[i][2] for i in range(simuNum)],
                      'k--')
# Draw last airplane
for i in range(0,nPlanes):
    n = int(simuNum/nPlanes*i)
    planeBody, wingSurf, tailSurf, planeTailHold = drawPlane3D(n, pos, quat[n], ax_3d, transp=0.5)

planeBody, wingSurf, tailSurf, planeTailHold = drawPlane3D(0, pos, quat[0], ax_3d, transp=1)
planeBody, wingSurf, tailSurf, planeTailHold = drawPlane3D(simuNum-1, pos, quat[simuNum-1], ax_3d, transp=1)

finalTime = time[simuNum-1]
setAxis_3d(ax_3d)
ax_3d.set_zlim(6,-2)
fig.set_size_inches(13, 9)
fig.savefig(figDir + simuName + '_fig3D_auto' + '.eps', dpi=100)

fig, ax = plt.subplots()
fig.set_size_inches(13, 9)
fig.subplots_adjust(wspace=.5)
ax_x = plt.subplot(2,2,1) #  Position

dv = 0.1
minv = [min([vel[i][j]-vel[0][j] for j in range(len(pos[0]))]) for i in range(simuNum)]
minv = min(minv)
maxv = [max([vel[i][j]-vel[0][j] for j in range(len(pos[0]))]) for i in range(simuNum)]
maxv = max(maxv)
ax_x.set_ylim(minv - dv, maxv +dv )

ax_x.set_xlim(time[0],finalTime)
ax_x.set_ylabel('Relative Velocity  [m/s]')
plt.plot(time[0:simuNum], [vel[i][0]-vel[0][0] for i in range(simuNum)], 'r', label='$v_{x0}$ = ' + str(round(vel[0][0],2) ) + '$m/s$' )
plt.plot(time[0:simuNum], [vel[i][1]-vel[0][1] for i in range(simuNum)],  'g', label='$v_{y0}$ = '+ str(round(vel[0][1],2)) + '$m/s$' )
plt.plot(time[0:simuNum], [vel[i][2]-vel[0][2] for i in range(simuNum)],  'b', label='$v_{z0}$ = '+ str(round(vel[0][2],2)) + '$m/s$' )
plt.gca().invert_yaxis()
ax_x.legend()


ax_phi = plt.subplot(2,2,2)
dAng = 0.1
minAng = [min([angRate[i][j]-angRate[0][j] for j in range(len(pos[0]))]) for i in range(simuNum)]
minAng = min(minAng)
maxAng = [max([angRate[i][j]-angRate[0][j] for j in range(len(pos[0]))]) for i in range(simuNum)]
maxAng = max(maxAng)
ax_phi.set_ylim(minAng-dAng, maxAng+dAng)
ax_phi.set_ylabel('Relative Angular Rate  [rad/s]')
ax_phi.set_xlim(time[0],finalTime)
plt.plot(time[0:simuNum], [angRate[i][0]-angRate[0][0] for i in range(simuNum)], 'r', label='$\omega_{x0}$ = ' + str(round(angRate[0][0],2)) + '$rad/s$')
plt.plot(time[0:simuNum], [angRate[i][1]-angRate[0][1] for i in range(simuNum)], 'g', label='$\omega_{y0}$ = ' + str(round(angRate[0][0],2))  + '$rad/s$')
plt.plot(time[0:simuNum], [angRate[i][2]-angRate[0][2] for i in range(simuNum)], 'b', label='$\omega_{z0}$ = ' + str(round(angRate[0][0],2)) + '$rad/s$')
ax_phi.legend()


ax_v = plt.subplot(2,2,3)
dAng = 0.1
minVal = [min([pos[i][j] for j in range(len(pos[0]))]) for i in range(simuNum)]
minVal = min(minVal)
maxVal = [max([pos[i][j] for j in range(len(pos[0]))]) for i in range(simuNum)]
maxVal = max(maxVal)
ax_v.set_ylim(minVal-dAng, maxVal+dAng)
ax_v.set_ylabel('Position [m]')
ax_v.set_xlabel('Time [s]')
ax_v.set_xlim(time[0],finalTime)
plt.plot(time[0:simuNum],[pos[i][0] for i in range(simuNum)], 'r', label ='x')
plt.plot(time[0:simuNum],[pos[i][1] for i in range(simuNum)], 'g', label ='y')
plt.plot(time[0:simuNum],[pos[i][2] for i in range(simuNum)], 'b', label ='z')
plt.gca().invert_yaxis()
ax_v.legend()


ax_omega = plt.subplot(2,2,4)
dOmega = 0.1
minVal = [min([eul[i][j] for j in range(len(pos[0]))]) for i in range(simuNum)]
minVal = min(minVal)
maxVal = [max([eul[i][j] for j in range(len(pos[0]))]) for i in range(simuNum)]
maxVal = max(maxVal)
ax_omega.set_ylim(minVal-dOmega, maxVal+dOmega)
ax_omega.set_xlabel('Time [s]')
ax_omega.set_ylabel('Attitude [rad]')
ax_omega.set_xlim(time[0],finalTime)
plt.plot(time[0:simuNum], [eul[i][0] for i in range(simuNum)], 'r', label='roll')
plt.plot(time[0:simuNum], [eul[i][1] for i in range(simuNum)], 'g', label='pitch')
plt.plot(time[0:simuNum], [eul[i][2] for i in range(simuNum)], 'b', label='yaw')
ax_omega.legend()

#line_roll.set_data(time, [eul[i][1] for i in range(simuNum)])
#line_yaw.set_data(time, [eul[i][2] for i in range(simuNum)])

#open(csvDir + 'numeric_'+ simuName + '_posPred' + '.csv', 'r') as csvFile:
fig.savefig(figDir + simuName + '_fig2D' + '.eps', dpi=100)

fig, ax = plt.subplots()
fig.set_size_inches(8, 5)

ax_thrust = plt.subplot(3,1,1)
ax_thrust.set_xlim(time[0],finalTime)
ax_thrust.set_ylim(0, 0.7)
ax_thrust.set_ylabel('Thrust [mN]')
plt.plot(time[0:simuNum], [u[i][0] for i in range(simuNum)], 'r', label='Thrust $[mN]$')

ax_dE = plt.subplot(3,1,2)
ax_dE.set_xlim(time[0],finalTime)
ax_dE.set_ylim(-0.4, 0.4)
ax_dE.set_ylabel('Elevator [rad]')
plt.plot(time[0:simuNum], [u[i][1] for i in range(simuNum)], 'r', label='Thrust $[mN]$')

ax_dR = plt.subplot(3,1,3)
ax_dR.set_xlim(time[0],finalTime)
ax_dR.set_ylim(-0.4, 0.4)
ax_dR.set_ylabel('Rudder [rad]')
plt.plot(time[0:simuNum], [u[i][2] for i in range(simuNum)], 'r', label='Thrust $[mN]$')
ax_dR.set_xlabel('Time [s]')

fig.savefig(figDir + simuName + '_control' + '.eps', dpi=100)

plt.show()
