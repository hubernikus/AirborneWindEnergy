from quatlib import *

from math import sin, cos, tan, atan2, pi 
import matplotlib.pyplot as plt
import numpy as np


# import casadi library
#from casadi import *


import yaml # import yaml files
with open('umx_radian.yaml') as yamlFile:
    aircraft = yaml.safe_load(yamlFile)


# Import Data
#function [NUM, FLOG, SYM] = kite_sim(aircraft, params)

#casadi based kite dynamics simulation

# -------------------------
# Enviromental constants
# -------------------------
g = 9.80665 # gravitational acceleration [m/s2] [WGS84]
ro = 1.2985 # standart atmospheric density [kg/m3] [Standart Atmosphere 1976]

# ---------------------------
# Glider geometric parameters
# ---------------------------
b = aircraft['geometry']['b']
c = aircraft['geometry']['c']
AR = aircraft['geometry']['AR']
S = aircraft['geometry']['S']
lam = aircraft['geometry']['lam']
St = aircraft['geometry']['St'] 
lt = aircraft['geometry']['lt']
Sf = aircraft['geometry']['Sf'] 
lf = aircraft['geometry']['lf']   
Xac = aircraft['geometry']['Xac']
Xcg = 0.031/c               # Center of Gravity (CoG) wrt leading edge [1/c]
Vf = (Sf * lf) / (S * b)    # fin volume coefficient []
Vh = (lt * St) / (S * c)    # horizontal tail volume coefficient [] 

#---------------------------
#Mass and inertia parameters
#---------------------------
Mass = aircraft['inertia']['mass']
Ixx = aircraft['inertia']['Ixx']
Iyy = aircraft['inertia']['Iyy'] 
Izz = aircraft['inertia']['Izz']
Ixz = aircraft['inertia']['Ixz']

#-------------------------------
#Static aerodynamic coefficients
#-------------------------------
# All characteristics assumed linear
CL0 = aircraft['aerodynamic']['CL0']
CL0_t = aircraft['aerodynamic']['CL0_tail']
CLa_tot = aircraft['aerodynamic']['CLa_total']  
CLa_w = aircraft['aerodynamic']['CLa_wing'] 
CLa_t = aircraft['aerodynamic']['CLa_tail']
e_o = aircraft['aerodynamic']['e_oswald']
dw = CLa_tot / (pi * e_o * AR) # downwash acting at the tail []
CD0_tot = aircraft['aerodynamic']['CD0_total'] 
CD0_w = aircraft['aerodynamic']['CD0_wing']
CD0_t = aircraft['aerodynamic']['CD0_tail'] 
CYb  = aircraft['aerodynamic']['CYb']
CYb_vt = aircraft['aerodynamic']['CYb_vtail']
Cm0 = aircraft['aerodynamic']['Cm0']
Cma = aircraft['aerodynamic']['Cma']
Cn0 = aircraft['aerodynamic']['Cn0']
Cnb = aircraft['aerodynamic']['Cnb']
Cl0 = aircraft['aerodynamic']['Cl0']
Clb = aircraft['aerodynamic']['Clb']

CLq = aircraft['aerodynamic']['CLq']
Cmq = aircraft['aerodynamic']['Cmq']
CYr = aircraft['aerodynamic']['CYr'] 
Cnr = aircraft['aerodynamic']['Cnr'] 
Clr = aircraft['aerodynamic']['Clr']
CYp = aircraft['aerodynamic']['CYp']
Clp = aircraft['aerodynamic']['Clp'] 
Cnp = aircraft['aerodynamic']['Cnp'] 

## ------------------------------
# Aerodynamic effects of control
# ------------------------------
CLde = aircraft['aerodynamic']['CLde']
CYdr = aircraft['aerodynamic']['CYdr']
Cmde = aircraft['aerodynamic']['Cmde'] 
Cndr = aircraft['aerodynamic']['Cndr'] 
Cldr = aircraft['aerodynamic']['Cldr']
CDde = aircraft['aerodynamic']['CDde'] # (assume negligible)

CL_daoa = -2 * CLa_t * Vh * dw # aoa-rate effect on lift (from Etkin) [] Stengel gives positive estimation !!!
Cm_daoa = -2 * CLa_t * Vh * (lt/c) * dw #aoa-rate effect on pitch moment [] Stengel gives positive estimation !!!
    

def steadyLevel(dE, gamma):
    N_res = dE.shape[0]
    
    # Pitch equilibrium
    alpha = -(Cm0+Cmde*dE)/Cma

    dR = 0
    
    # Aerodynamic Coefficients
    C_L = CL0 + CLa_tot*alpha + CLde*dE
    C_Y = CYdr*dE
    C_D = CD0_tot + C_L*C_L/(pi * e_o * AR)
    
    vel2 = 2*Mass*g/(S*ro*(C_L+C_D*np.tan(alpha))) #velocity squared

    a = np.vstack(( np.cos(alpha+gamma),np.zeros((1,N_res)), 1*np.sin(alpha+gamma) ))
    v_matrix = np.tile(np.sqrt(vel2),[3,1])
    vel = np.multiply(a,v_matrix)

    T = S*ro*vel2*C_D/(2*np.cos(alpha))

    # Create output dictionnary
    steadyState = {}
    steadyState['dE'] = dE
    steadyState['alpha'] = alpha
    steadyState['vel'] = vel
    steadyState['T'] = T
    steadyState['gamma'] = gamma
    
    return steadyState


def longitudinalFlight(dE, gamma):
    N_res = dE.shape[0]

    # TODO. implement equations correctly...
    gamma = - gamma

    
    # Pitch equilibrium
    alpha = -(Cm0+Cmde*dE)/Cma

    dR = 0
    
    # Aerodynamic Coefficients
    C_L = CL0 + CLa_tot*alpha + CLde*dE
    C_Y = CYdr*dE
    C_D = CD0_tot + C_L*C_L/(pi * e_o * AR)
        
    #vel2 = 2*Mass*g/(S*ro*(C_L+C_D*np.tan(alpha))) #velocity squared
    vel2 =  2*Mass*g*(np.cos(gamma) - np.tan(alpha)*np.sin(gamma))/(S*ro*(C_L+C_D*np.tan(alpha))) #velocity squared
    #print(gamma)
    a = np.vstack(( np.cos(alpha),np.zeros((1,N_res)), 1*np.sin(alpha) ))
    v_matrix = np.tile(np.sqrt(vel2),[3,1])
    vel = np.multiply(a,v_matrix)
        
    #T = S*ro*vel2*C_D/(2*np.cos(alpha))
    T = 1/np.cos(alpha)*(Mass*g*np.sin(gamma) + C_D *0.5*ro*S*vel2)

    # Create output dictionnary
    steadyState = {}
    steadyState['dE'] = dE
    steadyState['alpha'] = alpha
    steadyState['vel'] = vel
    steadyState['T'] = T
    steadyState['gamma'] = gamma
    
    return steadyState


def steadyLevel_circle(mu, vel):
    # mu > 0 -> omega>0
    # mu > 0 -> omega<0
    gamma = 0 # horizontql circle

    dyn_press = 0.5*ro*vel**2 # dynamics pressure
    
    # FORMULAS 'steady aircraft flight'
    r = vel**2/(g*tan(mu))

    omega = vel/r

    w_bar = [0,
             c*omega*sin(mu)/(2*vel),
             b*omega*cos(mu)/(2*vel)]

    w = [0, sin(mu), cos(mu)]

        #alpha = 0 # TODO: what is it really..

    # Moment equilibirium
    C_n0_bar = (Cnr*w_bar[2] + Cnp*w_bar[0]) + Cn0
    C_l0_bar = (Clr*w_bar[2] + Clp*w_bar[0]) + Cl0
    
    dR = (C_l0_bar*Cnb - C_n0_bar*Clb)/(Cndr*Clb - Cldr*Cnb)

    C_m0_bar = w_bar[1]*Cmq + Cm0


    # Force equilibirium
    L = 0.5          * (1/sin(mu)*Mass*vel**2/r - 1/cos(mu)*Mass*g)

    # Angle of Attack - Force equilibrium
    b0 = CL0 + CLq*w_bar[1] - CLde/Cmde*C_m0_bar
    b1 = CLa_tot - CLde/Cmde*Cma
    alpha = 1/b1* ( 2*L/(ro*vel**2) - b0)

    T = 0.5/sin(alpha)*(1/sin(mu)*Mass*vel**2/r + 1/cos(mu)*Mass*g)


    dE = - (C_m0_bar + Cma * alpha)/Cmde

    beta = -(C_l0_bar + Cldr*dR)/Clb


    
    vel = [cos(alpha)*cos(beta)*vel,
           sin(beta)*vel,
           sin(alpha)*cos(beta)*vel]
    x = [0, r ,0]
    
    q = eul2quat([mu, gamma,0])
    
    # Create output dictionnary
    steadyState = {}
    steadyState['dE'] = dE
    steadyState['dR'] = dR
    steadyState['alpha'] = alpha
    steadyState['beta'] = beta
    steadyState['T'] = T
    steadyState['gamma'] = gamma
    steadyState['mu'] = mu
    steadyState['vel'] = vel
    steadyState['angRate'] = w
    steadyState['pos'] = x
    steadyState['quat'] = q
    
    return steadyState



def steadyLevel_longitudial(elevator):
    N_res = elevator.shape[0]
    gamma = 0 # No altitude change

    steadyState = longitudinalFlight(elevator, gamma)

    return steadyState

#def steadyLevel_circle(alpha, mu):
    # gamma = 0

    # omega_bar = 1/Mass*(S*ro*0.5)*(C_L)

    

def testEquationsOfMotion_model(x, alpha, vel, u):
    # TODO: more generalized
    
    # Extract Controller values
    dE, dR, T = u
    
    #Cl = Cl0 +Clb*beta +
    CL = CL0 + CLa_tot*alpha + CLde*dE
    CY = CYdr*dR
    CD = CD0_tot + CL*CL/e_o

    # Velocity
    vel2 = vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]
    dyn_press = ro*vel2*0.5

    # Force Coefficients
    F_lift = CL*dyn_press*S
    F_sideforce = CY*dyn_press*S
    F_drag = CD*dyn_press*S

    # Simplified Equations of motions
    F_x = -F_drag + T*cos(alpha)
    F_z = -F_lift - T*sin(alpha) + Mass*g

    if(F_x or F_z):
        print('Force equilibrium wrong. F_x:', F_x, ', F_z:', F_z)

    # Momentum Coefficicients
    Cm = Cm0 + Cma * alpha + Cmde * dE

    #  Momentum Equilibrium
    M = Cm * dyn_press*S*c

    pitch = 0 
    roll = M
    yaw = 0

    if(pitch or roll or yaw):
        print('Momentum equilibrium not satisfied. Roll:', roll)

        
def testEquationsOfMotion_script(x, alpha, vel, u):
    return 1
# ---------------------------
print('Start script')



def writeToYamlFile(fileName, initValues):
    initVal = {}
    initVal['dE'] = float(initValues['dE'][0])
    initVal['alpha'] = float(initValues['alpha'][0])
    initVal['gamma'] = float(initValues['gamma'])
    vel = initValues['vel'].tolist()
    #print([vel[i][0] for i in range(len(vel))])
    initVal['vel'] = [float(vel[i][0]) for i in range(len(vel))]
    initVal['T'] = float(initValues['T'][0])

    if 'q' in initValues:
        initVal['q'] = [float(vel[i][0]) for i in range(len(q))]

    with open(fileName + '.yaml', 'w') as outfile:
        yaml.dump(initVal, outfile, default_flow_style=False)

def writeToYamlFile_singleValue(fileName, initValues):
    # initVal = {}
    # initVal['dE'] = float(initValues['dE'])
    # initVal['alpha'] = float(initValues['alpha'])
    # initVal['gamma'] = float(initValues['gamma'])
    # initVal['vel'] = [float(vel[i]) for i in range(len(vel))]
    # initVal['T'] = float(initValues['T'])

    # if 'q' in initValues:
    #     initVal['q'] = [float(q[i]) for i in range(len(q))]

    with open(fileName + '.yaml', 'w') as outfile:
        yaml.dump(initValues, outfile, default_flow_style=False)

    
figDir =  '../fig/'
dE = np.linspace(-0.05,0.05,41)

initValues = steadyLevel_longitudial(dE)

elevator = initValues['dE']*180/pi # in degrees
alpha = initValues['alpha']
vel = initValues['vel']
T = initValues['T']

# plt.figure()
# plt.subplot(3,1,1)
# plt.title('Steady Level Longitudinal Flight')
# plt.plot(elevator, alpha*180/pi)
# plt.ylabel('Angle of attack [deg]')
# plt.xlim(elevator[0],elevator[-1])
# plt.subplot(3,1,2)
# plt.plot(elevator, vel[0], 'r', label='x')
# plt.plot(elevator, vel[1], 'g', label='y')
# plt.plot(elevator, vel[2], 'b', label ='z')
# plt.xlim(elevator[0],elevator[-1])
# plt.ylabel('Velocity [m/s]')
# plt.subplot(3,1,3)
# plt.plot(elevator, T)
# plt.xlabel('Elevator [deg]')
# plt.ylabel('Thrust [N]')
# plt.xlim(elevator[0],elevator[-1])

elevator0 = np.array([0])
initValues = steadyLevel_longitudial(elevator0)

elevator0 = initValues['dE'][0]
alpha0 = initValues['alpha'][0]
vel0 = initValues['vel']
T0 = initValues['T'][0]

print('Stable Longitudinal flight with:')
print('Elevator:', elevator0)
print('Angle of attack:', alpha0)
print('Velocities:')
print(vel0)
print('Thrust:', T0)


## Flight moving up
gamma = 5/180*pi
initValues = steadyLevel(dE, gamma)

writeToYamlFile('steadyState_longitudial_steadyLevel', initValues)

# plt.figure()
# plt.subplot(3,1,1)
# plt.title('Longitudinal Flight with Slope of {} deg'.format(round(180/pi*gamma,2)))
# plt.plot(elevator, alpha*180/pi)
# plt.ylabel('Angle of attack [deg]')
# plt.xlim(elevator[0],elevator[-1])
# plt.subplot(3,1,2)
# plt.plot(elevator, vel[0], 'r', label='x')
# plt.plot(elevator, vel[1], 'g', label='y')
# plt.plot(elevator, vel[2], 'b', label ='z')
# plt.ylabel('Velocity [m/s]')
# plt.xlim(elevator[0],elevator[-1])
# plt.legend( loc='upper right')
# plt.subplot(3,1,3)
# plt.plot(elevator, T)
# plt.xlabel('Elevator [deg]')
# plt.ylabel('Thrust [N]')
# plt.xlim(elevator[0],elevator[-1])


dE = np.array([0])
initValues = longitudinalFlight(dE, gamma)
writeToYamlFile('steadyState_longitudial', initValues)

elevator0 = initValues['dE'][0]
alpha0 = initValues['alpha'][0]
vel0 = initValues['vel']
T0 = initValues['T'][0]

# print('Stable Longitudinal flight with:')
# print('Elevator:', elevator0)
# print('Angle of attack:', alpha0)
# print('Velocities:')
# print(vel0)
# print('Thrust:', T0)

Vel = 7 # m/s
gamma = 0

mu = 10/180*pi # rad

initValues_circ = steadyLevel_circle(mu, Vel)
#initValues_circ = steadyLevel_circle(mu, Vel)

print('')
print('Stable Circle flight with:')
print('Elevator:', initValues_circ['dE'])
print('Angle of attack:', initValues_circ['alpha'])
print('Sideslip:', initValues_circ['beta'])
print('Velocities:')
print(initValues_circ['vel'])
print('Thrust:', initValues_circ['T'])

with open('steadyCircle' + '.yaml', 'w') as outfile:
        yaml.dump(initValues_circ, outfile, default_flow_style=False)


plt.show()
print('End script')
