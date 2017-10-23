%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                   kite validation script
%                   Version 1 - Lukas Huber
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all; %% To ensure clean workspace
%%
clc; close all; clear variables;


% Import ReadYalm
addpath(genpath('/home/lukas/Software/MATLAB/AddOns/YAMLMatlab_0.4.3'))

% Import Casadi
addpath('/home/lukas/Software/casadi/casadi-matlabR2014a-v3.2.3')
import casadi.*

%%
%create kite model
aircraft = ReadYaml('umx_radian.yaml');
parameters.simulation = 0;
parameters.plot = 0;
parameters.vr = 0;
parameters.int_type = 'cvodes';

start_sample = 10;
s_time = 0; %exp_telemetry(start_sample,end);
f_time = 1; %exp_telemetry(end,end);
parameters.t_span = [s_time, f_time, 0.1];
parameters.x0 = [1.5;0;0; 0;0;0; -3;0;-2.0; 0.7071; 0; 0; 0.7071]; %exp_telemetry(start_sample,1:13)';
parameters.u0 = [0.1;0;0];


%% First test
[num, flog, sym] = kite_sim(aircraft, parameters);

%%

sym.DYNAMICS

%num.INT(x0, u0)

%%


