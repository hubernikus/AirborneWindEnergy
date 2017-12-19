"""
##### Nonlinear Controll Library #####

Author: Lukas Huber
Date: 2017-11-18

"""
import numpy as np

from casadi import *

def LieDerivative(f,h,X, N = 1):
    # Lie derivative of higher order L_f^N
    print('Lie Derivative with N =', N) #TODO - remove this iteration counter output

    # Order is treated as integer (N =  floor(max(N,0))
    if N >= 2:
        # L_f^N h = L_f (L_f^(N-1) h) 
        return LieDerivative(f, LieDerivative(f,h,X, N-1), X)
    elif N >= 1: # order 1
        # Lie derivative - directional derivative
        # L_f h = \nabla \CDT f
        return mtimes(jacobian(h,X), f)
    else: # order 0
        return h

def LieBracket(f, g, X, N = 1):
    # Lie bracket of higher order
    #  ad_f^N g
    print('Lie Bracket with N =', N) # TODO -remove this iteration counter output
    
    # Order is treated as integer (N =  floor(max(N,0))
    if N >= 2:
        # ad_f^N g = [f, ad_f^(N-1) g]
        return LieBracket(f, LieBracket(f, g, X, N-1), X)
    elif N >= 1: # order 1
        # ad_f g =  [f, g] = L_f g - L_g f
        return (LieDerivative(f,g,X) - LieDerivative(g,f,X))
    else: # order 0
        return g

    
########################################################################

print('')    
print('Start test script')
print('')

# Reconstruct example 6.7 - Slotine
X = SX.sym('X', 2)

#a = SX.sym('a', 1)
a = 1

f = np.array([-2*X[0] + a*X[1] + sin(X[0]),
              -X[1]*cos(X[0])])

g = np.array([0, cos(2*X[0]) ])

bracket_fg = LieBracket(f,g,X)
F_bracket = Function('F_bracket', [X], [bracket_fg])

x = [1,1]
f_mat = F_bracket(x)
print(f_mat)


print([-a*cos(2*x[0]),
       cos(x[0])*cos(2*x[0]) - 2*sin(2*x[0])*(-2*x[0] + a*x[1] + sin(x[0]))] )
print('')


# Implementation example 6.9 for testing
Z = SX.sym('Z',3)
z = [4,2,1]
f_1 = np.array([4*Z[2], -1, 0])
f_2 = np.array([-Z[0], (Z[2])**2-3*Z[1], 2*Z[2]])
bracket_ff = LieBracket(f_1, f_2, Z)
F_bracketff = Function('F_bracketff', [Z], [bracket_ff])
ff_mat = F_bracketff(z)
print(ff_mat)
print([-12*z[2],+3,0])
print('')

# Lerivatives
# Lie der
L_f1_f2 = LieDerivative(f_1,f_2,Z)
F_L_f12 = Function('F_bracketff', [Z], [L_f1_f2])
ff_mat = F_L_f12(z) 
print(ff_mat)
print([-4*z[2],+3,0])
print('')


print('')
print('End test script')
print('')    

####################################################################
