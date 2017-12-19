import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

import numpy as np

from quatlib import *

class Aircraft:

    # Physical limit
    T_lim = [0, 0.3] # Newton
    dE_lim = [-10/180*pi, 10/180*pi] # rad
    dR_lim = [-10/180*pi, 10/180*pi] # rad



    def __init__(self):
        # Body geometry
        self.dirBody_B = np.array([1,0,0])
        self.dl1 = 1.2 # Length tail of plane
        self.dl2 = 0.5 # length noise of plane
        self.lWidth = 3 # thickness of body

        # Wing geometry
        self.dirWing_B = np.array([0, 1, 0])
        self.wingSpan = 0.7
        self.wingWidth = 0.4
        self.wingPos = 0

        # Tail geometry (-> direction is parallel as wing)
        self.dirTail_B = np.array([0, 0, -1])
        self.tailSpan = 0.3
        self.tailWidth = 0.25
        self.tailPos = -1.2
        self.tailPosz = 0.5


        # Simuulation paramaters
        self.motionType 'circular'
        self.visualization = 3
        self.control = 'None'

        
        # Simulation parameters
        
        parameters = dict()
        parameters['simulation']= 0
        parameters['plot'] = 0
        parameters['vr'] = 0
        parameters['int_type'] = 'cvodes'
        parameters['t_span'] = [t_start, t_final, dt]

        
    def importModel(self, fileName, motionType):
        if motionType=='linear':
            with open(yamlDir + 'steadyState_longitudial_steadyLevel.yaml') as yamlFile:
            #with open('steadyState_longitudial.yaml') as yamlFile:
                initCond = yaml.safe_load(yamlFile)
            self.vel0 = initCond['vel']
            self.alpha0 = initCond['alpha']
            self.elevator = initCond['dE']

            self.thrust = initCond['T']

            self.gamma = initCond['gamma']

            self.angRate0 = [0, 0, 0]
            self.x0 = [-15, 0, 0]

            #euler0 = [0,alpha0+gamma+pi, pi]
            self.euler0 = [0,alpha0+gamma,0]
            self.euler0 = [0,alpha0+gamma,0]


            self.quat0 = eul2quat(euler0)

            #dQuat = eul2quat([0,0.4,0])
            #quat0 = quatmul(quat0,dQuat)

            #quat0 = [quat0[i] for i in range(4)]

            #vel0  = [10, 0, 0.3]
            #angRate0 = [0,0.3,0]

            self.rudder = 0
            
        else:# circular or MPC_simu
            if motionType=='circular':
                with open( yamlDir + 'steadyCircle3.yaml') as yamlFile:
                #with open(yamlDir + 'circle_gammadeg.yaml') as yamlFile:
                    self.initCond = yaml.safe_load(yamlFile)

                self.quat0 = initCond['quat']

                self.gamma = initCond['gamma'] # inclination
                print(gamma)

            elif motionType=='MPC_simu':
                with open(yamlDir + 'steadyCircle_simuMPC.yaml') as yamlFile:
                    initCond = yaml.safe_load(yamlFile)
                self.motionType = 'circular' # circular motion was imulated
                self.quat0 = initCond['quat']


                self.gamma = 0

                # rotate to align with circle
                self.rotAng = eul2quat([0,0,-4*pi/16])
                self.quat0 = quatmul(rotAng, quat0)
                self.quat0 = [quat0[i] for i in range(4)]

            self.thrust = initCond['T']
            self.vel0 = initCond['vel']
            self.vel0 = [float(vel0[i]) for i in range(len(vel0))]
            self.angRate0 =  initCond['angRate']
            
            self.x0 = initCond['pos']
            self.posCenter = initCond['centerPos']

            self.rudder = initCond['dR']

            self.trajRad = initCond['radius']

        # State: [velocity (BRF), angular rates (BRF), position (IRF), quaternions (IRF-BRF)]

    def visual2d(self):
        return 1
    
    def drawPlane3d(self, it):
        
        x = self.x
        quat = self.quat[it]
        ax_3d = self.figAxis
    
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
        
        


    
