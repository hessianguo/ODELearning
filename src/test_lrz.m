clear
close all
clc
debugflag = 2;
if debugflag == 1
    % test for lorenze
    polyorder = 5;
    usesine = 0;
    sigma = 10;  % Lorenz's parameters (chaotic)
    beta = 8/3;
    rho = 45;
    %rho = 28;
    n = 3;
    %x0=[-8; 8; 27];  % Initial condition
    x0 = [1, 1, 1];

    % Integrate
    dt = 0.01;
    tspan=[dt:dt:4];
    N = length(tspan);
    options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
    [t,x]=ode45(@(t,x) lorenz(t,x,sigma,beta,rho),tspan,x0,options);
    xclean = x;


    %% compute Derivative
    % compute clean derivative  (just for comparison!)
    for i=1:length(x)
        dxclean(i,:) = lorenz(0,xclean(i,:),sigma,beta,rho);
    end

    normalizeflag = 0;
    if normalizeflag == 1
        x = x./20;
        dxclean = dxclean/20;
        xclean = xclean/20;
    end

    % add noise
    eps = .5;
    x = x + eps*randn(size(x));
elseif debugflag == 2
    load tv.mat;
    x = x_noise(:, 2:end)';
    dxclean = x_dot_true(:, 2:end)';
    tspan = t(2:end);
    dt = t(2) - t(1);
end

%%  Total Variation Regularized Differentiation
dxt(:,1) = TVRegDiff( x(:,1), 50, .2, [], 'small', 1e12, dt, 1, 1 );
dxt(:,2) = TVRegDiff( x(:,2), 50, .2, [], 'small', 1e12, dt, 1, 1 );
dxt(:,3) = TVRegDiff( x(:,3), 50, .2, [], 'small', 1e12, dt, 1, 1 );

close all
subplot(131)
plot(tspan, dxclean(:,1),'.-r')
hold on
plot(tspan, dxt(2:end,1), 'b');
hold off
subplot(132)
plot(tspan, dxclean(:,2),'.-r')
hold on
plot(tspan, dxt(2:end,2), 'b');
hold off
subplot(133)
plot(tspan, dxclean(:,3),'.-r')
hold on
plot(tspan, dxt(2:end,3), 'b');
hold off

%% Save data
%dataname = sprintf('LRZDATA_N%d_nsr%.1f.mat', N, eps);
%save(dataname, 'xclean', 'x', 't', 'dxt', 'dxclean')
