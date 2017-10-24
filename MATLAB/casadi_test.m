%create kite model
clc; close all; clear variables; 

% Add casadi to path
addpath('/home/lukas/Software/casadi/casadi-matlabR2014a-v3.2.3')
import casadi.*

% add yaml to path
addpath(genpath('/home/lukas/Software/MATLAB/AddOns/YAMLMatlab_0.4.3'))

aircraft = ReadYaml('umx_radian.yaml');
parameters.simulation = 0;
parameters.plot = 0;
parameters.vr = 0;
parameters.int_type = 'cvodes';

s_time = 0;
f_time = 1;
dt = 0.1;
parameters.t_span = [s_time, f_time, dt];
parameters.x0 = [1.5,0,0,0,0,0,-3,0,-2,0.7071,0,0,0.7071]';
parameters.u0 = [0;0;0];

[num, flog, sym] = kite_sim(aircraft, parameters);

%test
% #1 - dynamics
x0 = parameters.x0;
u0 = parameters.u0;
dynamics = num.DYN_FUNC;
res = dynamics(x0,u0); % x0 - state (13,1), u0 - control (3,1)
res1 = full(res)

% #2 - Jacobian
jacobian_num= num.DYN_JACOBIAN;
res = jacobian_num(x0, u0);
res2 = res{1}

% #3 - Integrator
integrator_num = num.INT;

%
out = integrator_num('x0',x0,'p',u0);

res3 = full(out.xf)

