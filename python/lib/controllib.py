'''
Controller Library for system

@author Lucas Huber
@date 04-12-2017

'''
# find control law -F_k*x
# which minimizes J = sum_k0^inf (x_k^T Q x_k + u_k^T R u_k + 2* x_k^T *N*u_k)

# -------------------------------------------------------------------------
# Libraries
from casadi import * 

import numpy as np
import scipy.linalg

# ------------------------------------------------------------------------

dim = 13 # number of states

#
T_lim0 = [0, 0.3] # Newton
dE_lim0 = [-10/180*pi, 10/180*pi] # rad
dR_lim0 = [-10/180*pi, 10/180*pi] # rad


def checkControlRange(u,T_lim = T_lim0, dE_lim= dE_lim0, dR_lim = dR_lim0):
    if(u[0]< T_lim[0]): # Check range of Thrust control
        print('Warning: Control T={} out of range'.format(u[0]))
    elif(u[0] > T_lim[1]):
        print('Warning: Control T={} out of range'.format(u[0]))

    if(u[1]< dE_lim[0]): # Check range of Elevator control
        print('Warning: Control dE={} out of range'.format(u[1]))
    elif(u[1] > dE_lim[1]):
        print('Warning: Control dE={} out of range'.format(u[0]))

    if(u[2]< dR_lim[0]): # Check range of Rudder cojntrol
        print('Warning: Control dR={} out of range'.format(u[1]))
    elif(u[2] > dR_lim[1]):
        print('Warning: Control dR={} out of range'.format(u[0]))

def saturateControl(u, T_lim = T_lim0, dE_lim= dE_lim0, dR_lim = dR_lim0):
    if(u[0]< T_lim[0]): # Check range of Thrust control
        u[0] = T_lim[0]
    elif(u[0] > T_lim[1]):
        u[0] = T_lim[1]

    if(u[1]< dE_lim[0]): # Check range of Elevator control
        u[1] = dE_lim[0]
    elif(u[1] > dE_lim[1]):
        u[1] = dE_lim[1]

    if(u[2]< dR_lim[0]): # Check range of Rudder cojntrol
        u[2] = dR_lim[0]
    elif(u[2] > dR_lim[1]):
        u[2] = dR_lim[1]

    return u

def linearizeSystem(sym, X0, U0,  linDim = dim):
    # Linearize such that
    # x(t+1) = A*x(t) - B*u(t)

    kite_dynamicalSystem = sym['DYNAMICS']

    # CASAdi variables
    state_SYM = sym['state']
    control_SYM = sym['control']

    A_X = Function('A_X', [state_SYM, control_SYM], [jacobian(kite_dynamicalSystem, state_SYM)],['x','u'], ['A_X']) 
    B_X = Function('B_X', [state_SYM, control_SYM], [jacobian(kite_dynamicalSystem, control_SYM)],['x','u'], ['B_X'])

    #u = LQR_controller(state[-1], A_X, B_X, X0, U0)
    A = A_X(x=X0 , u=U0)
    A0 = A['A_X']
    A = A0[0:linDim,0:linDim]
        

    B = B_X(x=X0 , u=U0)
    B0 = B['B_X']
    B  =  B0[0:linDim,:]
    
    return A,B, A0, B0 
    
    
def dPID(A,B,Q,R):

    K = 1
    
    return K

def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
     
    dx/dt = A x + B u
     
    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151

    # convert CASAdi to numpy matrix format
    A = np.matrix(A)
    B = np.matrix(B)
    Q = np.matrix(DM(Q))
    R = np.matrix(DM(R))

    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    
    return K, X, eigVals

 
def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
     
     
    x[k+1] = A x[k] + B u[k]
     
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
    
    # convert CASAdi to numpy matrix format
    A = np.matrix(A)
    B = np.matrix(B)
    Q = np.matrix(DM(Q))
    R = np.matrix(DM(R))

    #first, try to solve the ricatti equation

    X = np.matrix(scipy.linalg.solve_discrete_are(A,B,Q,R))
                  
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
     
    eigVals, eigVecs = scipy.linalg.eig(A - B*K)
    
    return K, X, eigVals
 

def dLQR_controller(A,B,Q,R):
    dim = 13
    # Initial P matrix
    P = SX.eye(dim)
    P_norm2 = 2
    for i in range(dim):
            for j in range(dim):
                P_norm2 += fabs(P[i,j])
                
    # Iterative ricatti equation solver
    iterMax = 1e2
    tolP = 1e-3
    iter = 0
    while(1):
        #P = A_T*P*A - A_T*P*B * LA.inv(R + B_T*P*B)*B_T*P*A + Q
        P1 = mtimes(mtimes(A.T,P),A)
        P2 = mtimes(mtimes(A.T,P),B)
        P3 = mtimes(mtimes(B.T,P),B)
        P3 = inv(P3)
        P4 = mtimes(mtimes(B.T,P),A)
        P5 = Q
        P = P1 - mtimes(mtimes(P2,P3),P4) - P5

        P_norm= 0
        for i in range(dim):
            for j in range(dim):
                P_norm += fabs(P[i,j])

        # Stopping condition
        if((P_norm-P_norm2)/P_norm < tolP):
            print('P={} converged after iter={}'.format(P_norm, iter))
            break
        if(iter > iterMax):
            print('No convergence reached with P={} after iter={}'.format(P_norm, iter))
            break

        iter +=1 # increment counter
        P_norm_old   = P_norm
    
    # find control law u = -F_k*x
    K1 = mtimes(mtimes(B.T,P),B)
    K1 = inv(F1)
    K2 = mtimes(mtimes(B.T,P), A)
    K = mtimes(F1,F2)
    
    return K


def LQR_controller_ricatti(x_k, A_X, B_X, X0,U0):

    A = A_X(x=X0 , u=U0)
    A = A['A_X']
    
    B = B_X(x=X0 , u=U0)
    B = B['B_X']

    # minimize J_bar = sum_k0^inf (x_k^T Q x_k)
    Q = diag(SX([1,1,1, # Velocity
                 1,1,1, # angulr rotation
                 0,0,0, # Position
                 0,0,0,0])) # quaternion

    # R = zeros, N = zeros
    dimControl = 3
    R = SX.zeros(dimControl,dimControl)

    K,X,eigVals = dlqr(A, B, Q, R)

    x_k = np.matrix(x_k)
    X0 = np.matrix(X0)
        
    x_bar = x_k.T - X0
        
    u = - (K *  x_bar.T)
    
    print(u.shape)
    
    return u.T + U0


def LQR_controller(x_k, A_X, B_X, X0,U0):
    A = A_X(x=x_k, u=U0)
    A = A['A_X']
    
    B = B_X(x=x_k, u=U0)
    B = B['B_X']

    # minimize J_bar = sum_k0^inf (x_k^T Q x_k)
    # R = zeros, N = zeros
    Q = diag(SX([1,1,1, # Velocity
                        1,1,1, # angulr rotation
                        0,0,0, # Position
                        0,0,0,0])) # quaternion

    # Initial P matrix
    P = SX.eye(dim)
    P_norm2 = 2
    for i in range(dim):
            for j in range(dim):
                P_norm2 += fabs(P[i,j])
                
    # Iterative ricatti equation solver
    iterMax = 1e2
    tolP = 1e-3
    iter = 0
    while(1):
        #P = A_T*P*A - A_T*P*B * LA.inv(R + B_T*P*B)*B_T*P*A + Q
        P1 = mtimes(mtimes(A.T,P),A)
        P2 = mtimes(mtimes(A.T,P),B)
        P3 = mtimes(mtimes(B.T,P),B)
        P3 = inv(P3)
        P4 = mtimes(mtimes(B.T,P),A)
        P5 = Q
        P = P1 - mtimes(mtimes(P2,P3),P4) - P5

        P_norm= 0
        for i in range(dim):
            for j in range(dim):
                P_norm += fabs(P[i,j])

        # Stopping condition
        if((P_norm-P_norm2)/P_norm < tolP):
            print('P={} converged after iter={}'.format(P_norm, iter))
            break
        if(iter > iterMax):
            print('No convergence reached with P={} after iter={}'.format(P_norm, iter))
            break

        iter +=1 # increment counter
        P_norm_old   = P_norm
    
    # find control law u = -F_k*x
    F1 = mtimes(mtimes(B.T,P),B)
    F1 = inv(F1)
    F2 = mtimes(mtimes(B.T,P), A)
    F = mtimes(F1,F2)
    u = -mtimes(F,x_k-np.array(X0))

    dimControl = 3
    return [float(u[i])+U0[i] for i in range(dimControl)]
