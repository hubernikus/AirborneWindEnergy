##
##
##
##

import cmath

#d = -vel**2*Mass/(dyn_ress**r*S*sin(mu)) + b0
#c = (b1 + Cd0_tot + b0**2/(epsilon))
#b = 2*b1*b0/epsilon
#a = b1**2/espilon

#a = 1
#b = 3
#c = -6
#d = -8

a = 2
b = -3
c = -3
d = 2

# Cubic equation solver - Wikipedia
D = 18*a*b*c*d - 4*b**3*d - 4*a*c**3 - 27*a*a*d*d
D0 = b*b - 3*a*c
D1 = 2*b*b*b - 9*a*b*c + 27*a*a*d

# Check nubmer of soluts
if D0 == 0:
  print('attention, D0 =0')

C = []
C.append( ((D1 + (D1**2 - 4*D0**3 + 0j)**(0.5))*0.5)**(1/3.) )
C.append( (-0.5 + 0.5*cmath.sqrt(3)*1j)*C[0])
C.append( (-0.5 - 0.5*cmath.sqrt(3)*1j)*C[0])

x = [-1/(3*a)*(b+C[i] + D0/C[i]) for i in range(len(C))]

print(x)

# Check iiiit...
f_x = [a*x[i]**3 + b*x[i]**2 +c*x[i] + d  for i in range(len(x))]

print([abs(f_x[i]) for i in range(len(f_x))])

