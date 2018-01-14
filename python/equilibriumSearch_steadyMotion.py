from math import sin, cos, tan, atan2, pi 
import matplotlib.pyplot as plt
import numpy as np

import cmath

# import casadi library
#from casadi import *
yamlDir = '../steadyState_modes/'
modelDir = './model/'


# Add path to local libraries
import sys
sys.path.append('./model/')
sys.path.append('./lib/')

# Local libraries
from kite_sim import *
from quatlib import *
from controllib import *

import yaml # import yaml files
with open(modelDir + 'umx_radian.yaml') as yamlFile:
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
    SteadyState['alpha'] = alpha
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

    
    steadyState['dR'] = 0 # sraight line flight, no rudder
    steadyState['angRate'] = [0,0,0] # no rotation, straight line flight

    
    return steadyState


def steadyLevel_circle2(mu, vel):
    gamma = 0 # horizontal circle
    
    dyn_press = 0.5*ro*vel**2 # dynamics pressure
    
    # FORMULAS 'steady aircraft flight'
    r = vel**2/(g*tan(mu))

    omega = vel/r

    # WRONG ASSUMPTION - no side slip...
    w = [0, -omega*sin(mu), -omega*cos(mu)]

    w_bar = [b*w[0]/(2*vel),
             c*w[1]/(2*vel),
             b*w[2]/(2*vel)]

    K = 1/(pi*e_o*AR)

    T = 0.5*ro*vel**2*S*CD0_tot + 2*K*(Mass*g)**2/(ro*vel**2*S*(cos(mu))**2)

    C_n0_bar = (Cnr*w_bar[2] + Cnp*w_bar[0]) + Cn0
    C_l0_bar = (Clr*w_bar[2] + Clp*w_bar[0]) + Cl0
    
    dR = (C_l0_bar*Cnb - C_n0_bar*Clb)/(Cndr*Clb - Cldr*Cnb)
    beta = -(C_l0_bar + Cldr*dR)/Clb

    
    
        
    # Create output dictionnary
    steadyState = {}
    #steadyState['dE'] = dE
    # steadyState['dR'] = dR
    #steadyState['alpha'] = alpha
    # steadyState['beta'] = beta
    # steadyState['T'] = T
    # steadyState['gamma'] = gamma
    # steadyState['mu'] = mu
    # steadyState['vel'] = vel
    # steadyState['angRate'] = w
    # steadyState['pos'] = x
    # steadyState['quat'] = q
    #steadyState['radius'] = r
    
    return steadyState
    
def steadyLevel_circle(vel, r):
    gamma = 0 # steady level
    return steadyState_circle(vel, r, gamma)
    
def steadyState_circle(vel, r, gamma):
    # mu > 0 -> omega>0
    # mu > 0 -> ome ga<0
    print(gamma)
    #gamma = 0 # horizontal circle
    cosGam = cos(gamma)

    dyn_press = 0.5*ro*vel**2 # dynamics pressure
    
    # FORMULAS 'steady aircraft flight'
    #r = vel**2/(g*tan(mu))
    #mu = atan(vel**2/(g*r))
    mu = atan(vel**2*cosGam/(g*r))
    
    omega = vel/r
    
    # WRONG ASSUMPTION - no side slip...
    w = [0, omega*sin(mu), - omega*cos(mu)]

    w_bar = [b*w[0]/(2*vel),
             c*w[1]/(2*vel),
             b*w[2]/(2*vel)]
    
    #alpha = 0 # TODO: what is it really..

    # Moment equilibrium
    C_n0_bar = (Cnr*w_bar[2] + Cnp*w_bar[0]) + Cn0
    C_l0_bar = (Clr*w_bar[2] + Clp*w_bar[0]) + Cl0
    
    dR = (C_l0_bar*Cnb - C_n0_bar*Clb)/(Cndr*Clb - Cldr*Cnb)

    C_m0_bar = w_bar[1]*Cmq + Cm0

    # Angle of Attack - Force equilibrium
    b0 = CL0 + CLq*w_bar[1] - CLde/Cmde*C_m0_bar
    b1 = CLa_tot - CLde/Cmde*Cma
    
    # Solve cubic equation
    epsilon = pi*e_o*AR
    
    dd = -((cosGam*vel)**2*Mass)/(dyn_press*r*S*sin(mu)) + b0
    cc = (b1 + CD0_tot + b0**2/(epsilon))
    bb = 2*b1*b0/epsilon
    aa = b1**2/epsilon

    # Cubic equation solver - Wikipedia
    D = 18*aa*bb*cc*dd - 4*bb**3*dd - 4*aa*cc**3 - 27*aa*aa*dd*dd
    D0 = bb*bb - 3*aa*cc
    D1 = 2*bb*bb*bb - 9*aa*bb*cc + 27*aa*aa*dd

    # Check nubmer of solutes
    if D0 == 0:
      print('attention, D0 =0')
      C1 = (D1 + (D1**2 - 4*D0**3 + 0j)**(0.5))
      C2 = (D1 - (D1**2 - 4*D0**3 + 0j)**(0.5))
      if abs(C1) > abs(C2) :
        C = [(C1*0.5)**(1/3)]
      else:
        C = [(C2*0.5)**(1/3)]
    else: 
        C = [((D1 + (D1**2 - 4*D0**3 + 0j)**(0.5))*0.5)**(1/3.) ]

    C.append( (-0.5 + 0.5*cmath.sqrt(3)*1j)*C[0])
    C.append( (-0.5 - 0.5*cmath.sqrt(3)*1j)*C[0])

    alpha = [-1/(3*aa)*(bb+C[i] + D0/C[i]) for i in range(len(C))]

    it = 0
    for i in range(len(alpha)): # Check for complex results
        if(abs(alpha[it].imag) > 1e-10): # tolerance for numerical error
            del alpha[it]
        else:
            alpha[it] = alpha[it].real
            it += 1

    if(len(alpha) > 1):
        print('WARNING: alpha is a vector')

    #T = [0.5/sin(alpha[i])*(1/sin(mu)*Mass*vel**2/r + 1/cos(mu)*Mass*g) for i in range(len(alpha))]
    alpha = alpha[0] # TODO --- better solution for this mutliple alpha
    

    dE = -(C_m0_bar + Cma * alpha)/Cmde # 
    #dE = -(C_m0_bar + Cma * alpha)/Cmde # - not exact 
    
    #beta = [-(C_l0_bar + Cldr*dR)/Clb for i in range(len(alpha))]
    beta = -(C_l0_bar + Cldr*dR)/Clb

    CD = CD0_tot + (CL0 + CLa_tot * alpha)**2 / (pi * e_o * AR) #total drag coefficient

    F_D = dyn_press*S*CD
    F_L = ( (CL0 + CLa_tot * alpha) * dyn_press * S +
            (0.25 * CLq * c * S * ro) * vel * w[1] )

    
    T = F_D/cos(alpha)
    T1 = (Mass*g/cos(mu) - F_L)/sin(alpha)
    
    K = 1/(pi*e_o*AR)
    T2 = 0.5*ro*vel**2*S*CD0_tot + 2*K*(Mass*g)**2/(ro*vel**2*S*(cos(mu))**2)

    i = 0
    vel = [cos(alpha)*cos(beta)*vel,
           sin(beta)*vel,
           sin(alpha)*cos(beta)*vel]
    
    x = [0, r ,0]
    x_center = [0,0,0]
    q = eul2quat([mu+pi, alpha,0])
    
    
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
    steadyState['centerPos'] = x_center
    steadyState['quat'] = q
    steadyState['radius'] = r
    
    return steadyState



def steadyLevel_longitudial(elevator):
    N_res = elevator.shape[0]
    gamma = 0 # No altitude change

    steadyState = longitudinalFlight(elevator, gamma)

    return steadyState

#def steadyLevel_circle(alpha, mu):
    # gamma = 0

    # omega_bar = 1/Mass*(S*ro*0.5)*(C_L)

    
def testEquationsOfMotion_circle(x, alpha, vel, u):
    V = sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2] )
    ss = asin(vel[1] / V) #side slip angle [rad] (v(3)/v(1)) // small angle assumption
    aoa = atan2(vel[2] , vel[0])  # angle of attack definition [rad] (v(2)/L2(v))
    dyn_press = 0.5 * ro * V*V #dynamic pressure
    
    CD_w = CD0_w + (CL0 + CLa_w * aoa )**2 / (pi * e_o * AR) #wing drag coefficient
    CD = CD0_tot + (CL0 + CLa_tot * aoa)**2 / (pi * e_o * AR) #total drag coefficient 

    LIFT = ( (CL0 + CLa_tot * aoa) * dyn_press * S +
            (0.25 * CLq * c * S * ro) * V * w[1] )
    DRAG = CD * dyn_press * S
    SF = ( (CYb * ss + CYdr * dR) * dyn_press * S +
            0.25 * (CYr * w[2] + CYp * w[0]) * (b * ro * S) * V )

    
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

    with open(yamlDir + fileName + '.yaml', 'w') as outfile:
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

        
######################################################################################        
print('Start script')

yamlDir ='../steadyState_modes/'

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
print('')

#writeToYamlFile('steadyState_longitudial_steadyLevel', initValues)

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
gammaDeg = 15
gamma = gammaDeg*pi/180 #angle in rad 
dE = np.array([0])
initValues = longitudinalFlight(dE, gamma)
writeToYamlFile('steadyState_longitudinal_' + 'gamma' + str(gammaDeg) + 'deg', initValues)

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
Vel =  14 # m/s
r = 3 # m/s

gamma = 0
initValues_circ = steadyState_circle(Vel,r, gamma)
#initValues_circ2 = steadyLevel_circle2(mu, Vel)
#initValues_circ = steadyLevel_circle(mu, Vel)

print('')
print('Stable Circle flight with:')

print('Circle Radius:', initValues_circ['radius'])
print('Angle of attack:', initValues_circ['alpha'])
print('Sideslip:', initValues_circ['beta'])
#print('mu', mu)
print(' --- State --- ')
print('Angular Rate',initValues_circ['angRate'])
print('Position,', initValues_circ['pos'])
print('Velocities:', initValues_circ['vel'])
print('Quaternion', initValues_circ['quat'])
print('--- Control --- ')
print('Thrust:', initValues_circ['T'])
print('Elevator:', initValues_circ['dE'])
print('Rudder:', initValues_circ['dR'])
print('')


with open(yamlDir + 'circle_gamma0deg_vel' + str(Vel) + '_rad' + str(r) + '.yaml', 'w') as outfile:
    yaml.dump(initValues_circ, outfile, default_flow_style=False)

#with open('steadyCircle2' + '.yaml', 'w') as outfile:
#    yaml.dump(initValues_circ2, outfile, default_flow_style=False)


gamDeg = 15
gamma = gamDeg*pi/180
initValues_circ = steadyState_circle(Vel,r, gamma)
with open(yamlDir+'circle_gamma{}deg.yaml'.format(gamDeg) , 'w') as outfile:
    yaml.dump(initValues_circ, outfile, default_flow_style=False)

print('')
print('Stable Circle flight with:')

print('Circle Radius:', initValues_circ['radius'])
print('Angle of attack:', initValues_circ['alpha'])
print('Sideslip:', initValues_circ['beta'])
#print('mu', mu)
print(' --- State --- ')
print('Angular Rate',initValues_circ['angRate'])
print('Position,', initValues_circ['pos'])
print('Velocities:', initValues_circ['vel'])
print('Quaternion', initValues_circ['quat'])
print('--- Control --- ')
print('Thrust:', initValues_circ['T'])
print('Elevator:', initValues_circ['dE'])
print('Rudder:', initValues_circ['dR'])
print('')


plt.show()
print('End script')
