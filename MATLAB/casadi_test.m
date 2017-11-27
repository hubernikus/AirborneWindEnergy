%create kite model
clc; close all; clear variables; 

%%
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

% Time parameters
s_time = 0;
f_time = 10;
dt = 0.1;
parameters.t_span = [s_time, f_time, dt];

% State
parameters.x0 = [1.5,0,0,0,0,0,-3,0,-2,0.7071,0,0,0.7071]';

% Control
parameters.u0 = [0.1;0;0];

[num, flog, sym] = kite_sim(aircraft, parameters);

%test
% #1 - dynamics
x0 = parameters.x0;
u0 = parameters.u0;
dynamics = num.DYN_FUNC;
res = dynamics(x0,u0); % x0 - state (13,1), u0 - control (3,1)
res1 = full(res)

% #2 - Jacobian
jacobian_num = num.DYN_JACOBIAN;
res = jacobian_num(x0, u0);
res2 = res{1}

% #3 - Integrator
integrator_num = num.INT;

%
out = integrator_num('x0',x0,'p',u0);

res3 = full(out.xf)


%% Simualtion
%close all; 

figure;
h1 = subplot(1,1,1);

vel = [7, 0, 0]
angRate = [0, 0, 0]
x0 = [0, 0, 0]
quat = [1, 0, 0, 0]

%vel = [1.5, 0, 0];
%angRate = [0, 0 ,0];
%x0 = [0 , 0, 3];
%quat = [1, 0, 0, 0];

vel = [8.40584779, 0, 0.1020674]
%vel0 = [1.5, 0, 0]
angRate = [0, 0, 0]
x0 = [0, 0, 0]
euler0 = [0, 0.0121418, 0]

state = [vel, angRate, x0, quat]';
x = x0';
vel = vel';
t = [0];
u0 = [0.060057058,0,0.0];

for i = 1:(f_time-s_time)/dt
    t(i+1) = dt+t(i);
    fprintf('time: %2.2f \n', t(i+1))
    
    % Run simulation
    res = integrator_num('x0',state(:,i), 'p', u0);
    
    % Get values
    state(:,i+1) = full(res.xf);
    
    % Store values in matrices
    x(:,i+1) = state(7:9,i+1);
        
    % Visualize
    plot(h1, t, x(1,:),'r'); hold on;
    plot(h1, t, x(2,:),'g')
    plot(h1, t, x(3,:),'b')
    xlim([s_time, f_time]);
    ylim([-10,30])
    pause(0.11)
end
