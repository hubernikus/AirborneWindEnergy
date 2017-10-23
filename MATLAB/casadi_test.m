%create kite model
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
res = dynamics({x0,u0});
res1 = res{1}

% #2 - Jacobian
jacobian = num.DYN_JACOBIAN;
res = jacobian({x0, u0});
res2 = res{1}

% #3 - Integrator
integrator = num.INT;
out = integrator(struct('x0',x0, 'p',u0));
res3 = full(out.xf)

