'''

LQR  controller for the quite system

@author Lucas Huber
@date 04-12-2017

'''
# find control law -F_k*x
# which minimizes J = sum_k0^inf (x_k^T Q x_k + u_k^T R u_k + 2* x_k^T *N*u_k)

# -------------------------------------------------------------------------
# Libraries

import numpy as np
import numpy.linalg as LA

# CasADi
from casadi import *

# Local libraries
from kite_sim import *
from quatlib import *


# -------------------------------------------------------------------------

dim = 13 # number of states

# Simultion Time
t_start = 0
t_final = 10
dt = 0.1

# Parameters
parameters = dict()
parameters['simulation']= 0
parameters['plot'] = 0
parameters['vr'] = 0
parameters['int_type'] = 'cvodes'
parameters['t_span'] = [t_start, t_final, dt]

#state = vel0 + angRate0 +  x0 +  quat0
vel0 = [10,0,0] # Velocity in BRF
angRate0 = [0,0,0] # Angular rate in BRF
pos0 = [0,0,0] # Position in IRF
quat0 = [1,0,0,0] # Orienation from IRF to BRF

parameters['x0'] = vel0 + angRate0 +  pos0 +  quat0

# Control: [Thrust, Elevevator, Rudder]
parameters['u0'] = [0,0,0]

# Algeabraic equation for system dynamics using CasADi
num, flogg, sym = kite_sim(parameters)
kite_dynamicalSystem = sym['DYNAMICS']

X = sym['X']
U = sym['U']

# x_(k+1) = A*x_k _ B*u_k
# Linearize with CasADi
A_x = jacobian(kite_dynamicalSystem, X)
B_x  = jacobian(kite_dynamicalSystem, U)

A_X = Function('A_X', [X,U], [jacobian(kite_dynamicalSystem, X)]) # how to doooooo... read quickly
B_X = Function('B_X', [X,U], [jacobian(kite_dynamicalSystem, U)])

# Default input
U0 = [0.15, 0, 0]

def LQR_controller(x_k):

    A = A_X(x_k,U0)
    A_T = np.transpose(A)
    B = B_X(x_k,U0)
    B_T = np.tranpose(B)

    # minimize J_bar = sum_k0^inf (x_k^T Q x_k)
    Q = np.zeros((dim)) # Optimization matrix Q

    # Choose which states to optimize
    for i in range(6): # index 0-5 > velocity & angular rate
        Q[i][i] = 1  # states are punished


    # Initial P matrix
    P = np.eye(dim)

    # Iterational solution finding
    iterMax = 1e3
    tolP = 1e-3
    while(iter < iterMax):
        P = A_T*P*A - A_T*P*B * LA.inv(R + B_T*P*B)*B_T*P*A + Q
        iter +=1
        if( sum(sum(abs(P))) < tolP): # stopping condition
            break

    # find control law u = -F_k*x
    F = (LA.inverse(R + B_T*P*B) * B_T*P*A)
    u = -F*x_k
    
    return u

