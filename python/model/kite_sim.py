#import matplotlib.pyplot as plt

import yaml # import yaml files

from casadi import * # import casadi library

def kite_sim(params, aircraft = []):
    #function [NUM, FLOG, SYM] = kite_sim(aircraft, params)
    #casadi based kite dynamics simulation

    
    if not aircraft:
        modelPath = './model/'
        with open(modelPath + 'umx_radian.yaml') as yamlFile: # import aircraft
            aircraft = yaml.safe_load(yamlFile)
    
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
    
    #--------------------------
    #State variables definition
    #--------------------------
    r = SX.sym('r', 3) # position of the CoG in the Inertial Reference Frame (IRF) [m]
    q = SX.sym('q', 4) # body attitude relative to IRF [unit quaternion]
    v = SX.sym('v', 3) # linear velocity of the CoG in the Body Reference Frame (BRF) [m/s]
    w = SX.sym('w', 3) # glider angular velocity in BRF [rad/s]
    
    #----------------------------
    #Control variables definition
    #----------------------------
    #TODO: more detatailed propeller model will be added later
    T = SX.sym('T') # propeller propulsion : applies along X-axis in BRF [N]
    dE = SX.sym('dE') # elevator deflection [positive down] [rad]
    dR = SX.sym('dR') # rudder deflection [rad]
    dA = SX.sym('dA') # aileron deflection [reserved, but not used] [rad]
    dF = SX.sym('dF') # flaps deflection [reserved, but not used]
    
    #Read algorithm and simulation parameters
    #DEFAULTS
    SIMULATION = 0
    USE_CVODES = 1
    VR_SIM = 0
    PLOT_RESULTS = 0
    X0 = []
    U0 = []
    T_START = 0.0
    T_FINISH = 1.0
    DT = 0.01
    
    if('simulation' in params):
        SIMULATION = params['simulation']
    
    if('int_type' in params):
        if(params['int_type'] == 'cvodes'):
            USE_CVODES = 1
        else:
            #TODO CLEANER THAN THIS STATMENT
            assert (params['int_type'] != 'rk4'), 'ERROR: Unknown integrator type:' + type(params['int_type'] + '\n' + 'Use "cvodes" or "rk4" instead')

            USE_CVODES = 0
        
            
            
    
    #simulation time span
    if('t_span' in params):
        assert (params['t_span'] != 3), 'ERROR: time span format should be: [start, finish, dT]'
        
        T_START = params['t_span'][0]
        T_FINISH = params['t_span'][1]
        DT = params['t_span'][2]
        assert(T_START < T_FINISH), 'ERROR: T_START > T_FINISH'
 
    
    if(SIMULATION):
        #VR visualisation
        if('vr' in params):
            VR_SIM = params['vr']
        
        #initial condition and input
        assert not(('x0' in params) and ('u0' in params)), 'ERROR: initial conditions <<x0>> and control <<u0>> should be specified for simulation'
        
        X0 = params['x0']
        U0 = params['u0']
        
        
        if('plot' in prams):
            PLOT_RESULTS = params['plot']
        
    
    
    #TODO: for now assume there is no wind, this functionality will be added later
    V = L2(v)
    V2 = dot(v,v)
    
    ss = asin(v[1] / V) #side slip angle [rad] (v(3)/v(1)) // small angle assumption
    aoa = atan2(v[2] , v[0])  # angle of attack definition [rad] (v(2)/L2(v))
    dyn_press = 0.5 * ro * V2 #dynamic pressure
    
    CD_w = CD0_w + (CL0 + CLa_w * aoa )**2 / (pi * e_o * AR) #wing drag coefficient
    CD = CD0_tot + (CL0 + CLa_tot * aoa)**2 / (pi * e_o * AR) #total drag coefficient 
    
    #-------------------------
    #Dynamic Equations: Forces
    #-------------------------
    LIFT = ( (CL0 + CLa_tot * aoa) * dyn_press * S +
            (0.25 * CLq * c * S * ro) * V * w[1] )
    DRAG = CD * dyn_press * S
    SF = ( (CYb * ss + CYdr * dR) * dyn_press * S +
            0.25 * (CYr * w[2] + CYp * w[0]) * (b * ro * S) * V )
    
    #Compute transformation betweeen WRF and BRF: qw_b
    #qw_b = q(aoa) * q(-ss)
    q_aoa = SX(np.array([cos(aoa/2), 0, sin(aoa/2), 0]))
    q_ss = SX(np.array([cos(-ss/2), 0, 0, sin(-ss/2)]))
    
    qw_b = quat_mul(q_aoa, q_ss)
    qw_b_inv = quat_inv(qw_b)
    
    #The following is a complete disaster
    #Aerodynamic forces in BRF: Faer0_b = qw_b * [0 -DRAG SF -LIFT] * qw_b_inv
    qF_tmp = quat_mul(qw_b_inv, np.array([0, -DRAG, 0, -LIFT]))
    qF_q = quat_mul(qF_tmp, qw_b)
    Faero_b = qF_q[1:4]
    
    Zde = (-CLde) * dE * dyn_press * S
    FdE_tmp = quat_mul(quat_inv(q_aoa), np.array([0, 0, 0, Zde]))
    FdE = quat_mul(FdE_tmp, q_aoa)
    FdE = FdE[1:4]
    
    Faero_b = Faero_b + FdE + np.array([0, SF, 0])
    
    #Gravity force in BRF
    qG = quat_mul(quat_inv(q),SX([0,0,0,g]))
    qG_q = quat_mul(qG, q)
    G_b = qG_q[1:4]
    
    #Propulsion force in BRF
    T_b = np.array([T,0,0])
    
    #Tether force
    #value: using smooth ramp approximation
    Lt = 2.31 #tether length
    d_ = L2(r)
    
    #spring term
    #Rv = ((d_ - Lt)) * Heaviside(d_ - Lt, 1)
    Ks = 500 * Mass #300 Ks: 800
    Kd = 10 * Mass #30 Kd: 500
    Rv = ((d_ - Lt))
    Rs = -Rv * (r / d_)
    
    # #damping vterm
    # qvi = quat_mul(q, vertcat(SX([0]),v) )
    # #qvi = quat_mul(q, np.insert(v, 0, 0))
    # qvi_q = quat_mul(qvi, quat_inv(q))
    # vi = qvi_q[1:4]
    # # TODO: check line
    # Rd = (-r / d_) * dot(r,vi) / d_
    # R = ( Ks * Rs + Kd * Rd) * Heaviside(d_ - Lt, 1)
    
    
    # #BRF
    # qR = quat_mul(quat_inv(q), vertcat(SX([0]),R))
    # qR_q = quat_mul(qR, q)
    # R_b = qR_q[1:4]

    #v_dot = (Faero_b + T_b  + R_b)/Mass + G_b  - cross(w,v)
    
    #Total external forces devided by glider's mass (linear acceleration)
    v_dot = (Faero_b + T_b)/Mass + G_b  - cross(w,v)
    

    #-------------------------
    #Dynamic Equation: Moments
    #-------------------------
    #Rolling Aerodynamic Moment
    L = ( (Cl0 + Clb * ss + Cldr * dR) * dyn_press * S * b + 
         (Clr * w[2] + Clp * w[0]) * (0.25 * ro * b**2 * S) * V )
    
    
    #Pitching Aerodynamic Moment
    M = ( (Cm0 + Cma * aoa  + Cmde * dE) * dyn_press * S * c + 
         Cmq * (0.25 * S * c**2 * ro) * w[1] * V )
    
    #Yawing Aerodynamic Moment
    N = ( (Cn0 + Cnb * ss + Cndr * dR) * dyn_press * S * b + 
         (Cnp * w[0] + Cnr * w[2]) * (0.25 * S * b**2 * ro) * V )
     
    #Aircraft Inertia Matrix 
    J = SX([[Ixx, 0, Ixz], [0, Iyy, 00], [Ixz, 0, Izz]])
    
    #Angular motion equationin BRF
    Maero = vertcat(L, M, N)

    #Moments rotation SRF -> BRF
    T_tmp = quat_mul(quat_inv(q_aoa), vertcat(SX([0]),Maero) )
    Trot = quat_mul(T_tmp, q_aoa)
    Maero = Trot[1:4]

    w_dot = mtimes( inv(J) , (Maero - cross(w, mtimes(J,w) ) ) ) 
    
    #-----------------------------
    #Kinematic Equations: Position
    #-------------p----------------
    # Aircraft position in the IRF
    qv = quat_mul(q, vertcat(SX([0]),v))
    qv_q = quat_mul(qv, quat_inv(q))
    r_dot = qv_q[1:4]
    
    #-----------------------------
    #Kinematic Equations: Attitude
    #-----------------------------
    #Aircraft attitude wrt to IRF 
    q_dot = 0.5 * quat_mul(q, vertcat(SX([0]),w))
    
    #Now we completely define dynamics of the aircraft
    state = vertcat(v, w, r, q)
    control = vertcat(T, dE, dR)
    dynamics = vertcat(v_dot, w_dot, r_dot, q_dot)

    #   dyn_func = Function('dynamics', [state, control], [dynamics])
    dyn_func = Function('dynamics', [state, control], [dynamics], ['state', 'control'], ['dynamics'])

    #compute dynamics state Jacobian
    #d_jacobian = dyn_func.jacobian('state')
    d_jacobian = jacobian(dynamics, state)

    dyn_jac = Function('dyn_jacobian', [state, control], [d_jacobian],['state', 'control'], ['d_jacobian'])
    
    #define RK4 integrator scheme
    X = SX.sym('X',13)  
    U = SX.sym('U',3)
    dT = SX.sym('dT')
    
    # get symbolic expression for an integrator
    integrator_RK4 = RK4_sym(X, U, dyn_func, dT)
    RK4_INT = Function('RK4', [X,U,dT],[integrator_RK4],['X','U','dT'],['integratorFunc'])


    #get Jacobian of the RK4 Integrator
    integrator_jacobian = jacobian(integrator_RK4,X)
    rk4_jacobian = Function('RK4_JACOBIAN', [X, U, dT], [integrator_jacobian],['X', 'U', 'dT'], ['integrator_jacobian'])

    h = DT
    x0 = X0
    u0 = U0
    
    #CVODES integrator
    # TODO, check ode = struct('x',state, 'p',control, 'ode',dynamics)
    #ode = {'x': state, 'p': control, 'ode': dynamics}
    ode = {'x': state, 'p': control, 'ode': dynamics}
    opts = {'tf': h}

    CVODES_INT = integrator('CVODES_INT','cvodes', ode, opts)
    
    aoa_func = Function('AoA',[v],[aoa],['v'],['aoa'])
    ss_func = Function('SS',[v],[ss],['v'],['ss'])
    
    #return integrator function\
    NUM = {}
    if(USE_CVODES):
        NUM['INT'] = CVODES_INT
    else:
        NUM['INT'] = RK4_INT
    
    #Symbolic expression for the model
    SYM = {}
    SYM['INT']= integrator_RK4
    SYM['DYNAMICS'] = dynamics
    SYM['DYN_JAC'] = d_jacobian
    
    SYM['state'] = state
    SYM['control'] = control
    
    SYM['X'] = X
    SYM['U'] = U
    SYM['dT'] = dT

    # Function expressions for numeric evaluation
    NUM['DYN_FUNC'] = dyn_func
    NUM['DYN_JACOBIAN'] = dyn_jac
    NUM['RK4_JACOBIAN'] = rk4_jacobian
    
    if(VR_SIM):
        #create simulation scene
        print('TODO: implement Backview')
        #scene = CreateScene('BackView',1)
    
    
    #-----------------------------
    #Open-Loop Flight Simulation
    #-----------------------------
    
    #simulation & visualisation
    FLOG = []
    if (SIMULATION):
        t = T_START
        log = []
        flight_angles = []
        res = []
        while (t <= T_FINISH):
            #simulation loop
            print('Simulation looop. Implement iiiiit')
            #logging data
            #log = np.append(np.append(log, x0), t)
            log = vertcat(log,x0,t)
            
            aoa_t = aoa_func({x0[0:3]})
            #TODO: check if coherent - full(aoa_t{1})
            full(aoa_t[0])
            
            ss_t = ss_func({x0[0:3]})
            #full(ss_t{1})
            #TODO - check coherency ---- flight_angles = [flight_angles full(aoa_t{1}) full(ss_t{1})]
            flight_angles = [flight_angles, full(aoa_t[0]), full(ss_t[0])]
            
            if (USE_CVODES):
                out = dyn_func({x0,u0})
                #TODO - check coherency --- dyn =full(out{1})
                dyn =full(out[0])
                out = CVODES_INT(struct('x0',x0, 'p',u0))
                res = full(out.xf)
            else:
                out = RK4_INT({x0, u0, h})
                #TODO - check coherency ---- res = full(out{1})
                res = full(out[0])
            
            t = t + h
            x0 = res
            
            if(VR_SIM):
                UpdateScene(scene, res[6:9], res[9:13],t)
                #a = res([6:9]
        
        #return flight telemetry
        FLOG = log
        
        if(PLOT_RESULTS):
            #trajectory 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(log[:][7], log[:][8], -log[:][9])
            #axis equal
            plt.grid(True)
            
            #velocities in BRF
            fig = plt.figure('Name','Linear Velocities in BRF','NumberTitle','off')
            plt.plot(log[:][-1], log[:][0:3])
            plt.grid(True)
            
            #angle of attack
            fig = plt.figure('Name','Angle of Attack & Sideslip Angle','NumberTitle','off')
            #plot(rad2deg(flight_angles))
            plt.plot(log[:][-1], flight_angles)
            plt.grid(True)
            
            #attitude
            fig = plt.figure('Name','Attitude (EulerAngles)','NumberTitle','off')
            # TODO: ZYX ????
            plt.plot(log[:][-1], quat2eul(log[:][9:13],'ZYX'))
            plt.grid(True)
            
            #angular rates
            fig = plt.figure('Name','Angular rates in BRF','NumberTitle','off')
            plt.plot(log[:][-1], log[:][4:6])
            plt.grid(True)

    return NUM, FLOG, SYM
    # kite_sim functions end here

    #Casadi helper functions
    
    #L2-norm of a vector
def L2(x):
    # L2 =
    return sqrt(dot(x,x))

   
    #quaternion multiplication
def quat_mul(q1,q2):
    s1 = q1[0]
    v1 = q1[1:4]
    
    s2 = q2[0]
    v2 = q2[1:4]
    
    s = s1*s2 - dot(v1,v2)
    v = cross(v1,v2) + s1*v2 + s2*v1
    # q = np.array([s v(1) v(2) v(3)])

    return vertcat(s,v[0:3])
       
    #inverse quaternion
def quat_inv(q):
    # q_inv =
    return  vertcat(q[0], -q[1], -q[2], -q[3])
    
    #Runge-Kutta 4th-order symbolic integration method
def RK4_sym(x, u, f, h):
    # this integrator is problem specific
    #k1 = f({x,u})
    #k2 = f({x + 0.5 * h * k1{1},u})
    #k3 = f({x + 0.5 * h * k2{1},u})
    #k4 = f({x + h * k3{1},u})
    k1 = f(x,u)
    # TODO: correct syntax ...
    k2 = f(x + 0.5 * h * k1[0],u)
    k3 = f(x + 0.5 * h * k2[0],u)
    k4 = f(x + h * k3[0],u)
    
    #  x_h 
    return  (x + (h/6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]))
    
    #Heaviside smooth approximation
def Heaviside(x,k):
    #H = k / (1 + exp(-4*x))
    return  k / (1 + exp(-4*x))
    
def CreateScene(viewpoint, is_recording):
    #create 3D scene
    scene = struct
    
    scene.world = vrworld('world.wrl','new')
    open(scene.world)
    fig = vrfigure(scene.world)
    # go to a viewpoint suitable for user navigation
    set(fig, 'Viewpoint', viewpoint)
    
    # get the manipulated airplane node
    scene.position = vrnode(scene.world, 'GliderTranslational')
    scene.attitude = vrnode(scene.world, 'GliderRotational')
    
    if(is_recording):
        set(scene.world, 'RecordMode', 'manual')
        set(fig, 'Record2DFileName', 'flight_sim.avi')
        set(fig, 'Record2D', 'on')
        set(fig, 'Record2DCompressQuality', 100)
    
def UpdateScene(scene, position, attitude, time):
    #frame transformations with VR simulator
    q_sim = eul2quat(SX([0, 0, pi/2]))
    
    Xi = position
    Qi = attitude
    
    #apply simulation and draw
    x_sim = quatmul(quatmul(q_sim, np.insert(Xi, 0 ,0)),quatinv(q_sim))
    q_disp = quatmul(quatmul(q_sim, Qi),quatinv(q_sim))
    
    scene.position.translation = x_sim[1:4]
    scene.attitude.rotation = quat2axang(q_disp)
    
    set(scene.world, 'Time', time)
    
    vrdrawnow
