%% IndirectAdaptiveControl.m
% Adaptive gain scheduling script for the Simulink model

clear all; clc;
load("ann_model_paramsVI_Realtime.mat")

% Define Simulink sampling time
params.Tsam = 1e-5;
% Grid nominal frequency
params.f0 = 50;

%% LCL filter design
% DC link voltage
params.Vdc = 800;
% Switching frequency
params.fs = 10e3;
params.Ts = 1/params.fs;
params.Filter.Ls_1 = 1e-3;
params.Filter.Ls_2 = 0.5e-3;
params.Filter.Cs   = 50e-6;

%% PI current controller
% Current-loop response time
params.Tresc = 1e-3;
params.e     = 1;
params.omega = 2*pi/params.Tresc;
params.k     = 1/params.Filter.Ls_1;
params.Kpc   = 2*params.e*params.omega/params.k;
params.Kic   = params.omega^2/params.k;

%% PI voltage controller
% Voltage-loop response time
params.Tresv  = 10e-3;
params.ev     = 1;
params.omegav = 2*pi/params.Tresv;
params.kv     = 1/params.Filter.Cs;
params.Kpv    = 2*params.ev*params.omegav/params.kv;
params.Kiv    = params.omegav^2/params.kv;

%% Power and voltage references
% Active power reference (4.5 kW)
params.Pref = 4.5e3;
% Reactive power reference (1 kVAR)
params.Qref = 1.5e3;
% Nominal phase voltage (RMS)
params.Vn   = 110;
params.wn   = 2*pi*50;
params.Vj   = 110;

%% Pole-placement specification
zeta    = 1;
T_setl  = 1;
omega_n = 4/(zeta*T_setl);
p1      = -zeta*omega_n + 1i*omega_n*sqrt(1-zeta^2);
poles   = [p1, conj(p1), -80, -90, -100];

%% Initial controller gain K (2×5)
K_now = -[...
    339.437515294446   -37.1962550049616  -155.433559437714   37.2641713114898   3289.76434330323;...
    287.980448278270    35.5843781184771  -282.198010268474   58.4116660247888  -4058.33626972146...
];

%% Conventional P- and Q-loop fixed gains
Kp = 0.006383815167581;
Dp = 2.506338228783948e3;
Kq = 0.076670166944346;
Dq = 1.033099625954584;

%% 2. Load model, configure, and start simulation
mdl     = "FullModelAddANN_Final_ANN_revised";
gainBlk = 'FullModelAddANN_Final_ANN_revised/Power_control/Subsystem6/K';

load_system(mdl);
set_param(mdl, "StopTime", "40");    % simulate for 100 s
set_param(mdl, "SimulationCommand", "start");
disp("Model started ...");

%% 3. Monitoring loop & gain update
while strcmp(get_param(mdl,'SimulationStatus'),'running')

    % Wait for the simulation run object
    runObj = [];
    while isempty(runObj)
        runObj = Simulink.sdi.getCurrentSimulationRun(mdl);
        pause(0.01);
    end

    % Wait until the first two signals each have at least one sample
    id1 = runObj.getSignalIDByIndex(1);
    id2 = runObj.getSignalIDByIndex(2);
    sig1 = Simulink.sdi.getSignal(id1);
    sig2 = Simulink.sdi.getSignal(id2);
    while isempty(sig1.Values.Data) && isempty(sig2.Values.Data)
        pause(0.01);
        sig1 = Simulink.sdi.getSignal(id1);
        sig2 = Simulink.sdi.getSignal(id2);
    end

    % Extract latest Rg and Lg values
    tsRg = sig1.Values;
    tsLg = sig2.Values;
    Rg = tsRg.Data(end);
    Lg = tsLg.Data(end);
    fprintf("Lg = %f, Rg = %f\n", Lg, Rg);

    % Compute updated gain via pole-placement
    K_now = compute_gain_Lg_Rg(Lg, Rg, params, poles);
    fprintf("K = [%s]\n", num2str(K_now));

    % Update the Gain block in Simulink
    set_param(gainBlk, 'Gain', mat2str(-K_now));

    pause(0.5);  % poll at ~2 Hz
end

disp("Simulation finished.");

%% --- Local function: compute K using pole-placement ---
function [K] = compute_gain_Lg_Rg(Lg, Rg, params, poles)
    % Unpack parameters
    f0   = params.f0;
    Pref = params.Pref;
    Qref = params.Qref;
    Vn   = params.Vn;
    Vj   = params.Vj;

    % Compute grid reactance Xg
    Xg = 2*pi*f0*Lg;

    % Solve for steady-state voltage and angle
    [Ess, delta_ss] = solve_E_delta(Pref, Qref, Vn, Rg, Lg, f0);
    Vi0    = Ess;
    theta0 = delta_ss;

    % Build KPQ matrix
    D   = Rg^2 + Xg^2;
    k   = 3/D;
    K11 = k*( Rg*Vi0*Vj*sin(theta0) + Xg*Vi0*Vj*cos(theta0) );
    K12 = k*( Rg*(2*Vi0 - Vj*cos(theta0)) + Xg*Vj*sin(theta0) );
    K21 = k*( Xg*Vi0*Vj*sin(theta0) - Rg*Vi0*Vj*cos(theta0) );
    K22 = k*( Xg*(2*Vi0 - Vj*cos(theta0)) - Rg*Vj*sin(theta0) );
    KPQ = [K11 K12; K21 K22];

    % Use fixed P/Q-loop gains
    Kp = 0.006383815167581;
    Dp = 2.506338228783948e3;
    Kq = 0.076670166944346;
    Dq = 1.033099625954584;

    % System matrices for pole-placement
    A = [ ...
        0 0 Dp*Kp          Kp*KPQ(1,2)      Kp*KPQ(1,1); ...
        0 0 0              Dq*Kq+Kq*KPQ(2,2) Kq*KPQ(2,1); ...
        0 0 0              0                0; ...
        0 0 0              0                0; ...
        0 0 1              0                0  ...
    ];
    B = [1 0; 0 1; 1 0; 0 1; 0 0];

    % Check controllability
    if rank(ctrb(A,B)) < size(A,1)
        warning("System not controllable: Lg=%.3e, Rg=%.3e", Lg, Rg);
        K = zeros(2,5);
        return;
    end

    % Compute gain matrix
    K = place(A, B, poles);
end

%% --- Local function: solve E_ss & delta_ss via fsolve ---
function [Ess, delta_ss] = solve_E_delta(Pref, Qref, Vn, Rg, Lg, f)
    % Compute grid reactance
    Xg = 2*pi*f*Lg;
    D  = Rg^2 + Xg^2;
    k  = 3/D;

    % Steady-state power equations
    fun = @(x)[ ...
        Pref - k*x(1)*( Rg*(x(1) - Vn*cos(x(2))) + Vn*Xg*sin(x(2)) ); ...
        Qref - k*x(1)*( Xg*(x(1) - Vn*cos(x(2))) - Vn*Rg*sin(x(2)) ) ...
    ];

    % Initial guess
    Pslope = Pref*D/(3*Vn^2);
    delta0 = asin(max(min(Pslope,0.99),-0.99));
    x0 = [Vn, delta0];

    opts = optimoptions('fsolve','Display','off', ...
                       'FunctionTolerance',1e-12, ...
                       'StepTolerance',    1e-12);
    sol = fsolve(fun, x0, opts);
    Ess     = sol(1);
    delta_ss= sol(2);
end
