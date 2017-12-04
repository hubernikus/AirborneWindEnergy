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

#state = vel0 + angRate0 +  x0 +  quat0
vel0 = [10,0,0] # Velocity in BRF
angRate0 = [0,0,0] # Angular rate in BRF
pos0 = [0,0,0] # Position in IRF
quat0 = [1,0,0,0] # Orienation from IRF to BRF

parameters['x0'] = vel0 + angRate0 +  x0 +  quat0


# Control: [Thrust, Elevevator, Rudder]
parameters['u0'] = [0,0,0]

# Algeabraic equation for system dynamics using CasADi
num, flogg, sym = kite_sim(aircraft, parameters)


# x_(k+1) = A*x_k _ B*u_k

# Linearize with CasADi
A = []
A_T = np.transpose(A)
B = []
B_T = np.tranpose(B)

# minimize J_bar = sum_k0^inf (x_k^T Q x_k)
Q = np.zeros((dim)) # Optimization matrix Q

# Choose which states to optimize
for i in range(6): # index 0-5 > velocity & angular Rate
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
