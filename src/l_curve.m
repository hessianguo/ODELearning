function [reg_corner,rho,eta,reg_param] = l_curve(U,s,B)
% Plot the L-curve and find its "corner".
%
% Plots the L-shaped curve of eta, the solution norm || x || or
% semi-norm || L x ||, as a function of rho, the residual norm
% || A x - b ||, for the following methods:
% The corresponding reg. parameters are returned in reg_param.  If no
% method is specified then 'Tikh' is default.  For other methods use plot_lc.
%
% If any output arguments are specified, then the corner of the L-curve
% is identified and the corresponding reg. parameter reg_corner is
% returned.  Use routine l_corner if an upper bound on eta is required.

% Reference: [1] P. C. Hansen & D. P. O'Leary, "The use of the L-curve in the regularization 
% of discrete ill-posed problems",  SIAM J. Sci. Comput. 14 (1993), pp. 1487-1503.

% Set the range of lambda
% Initialization.
n = size(U,1);   % U is nxn
% n = size(s,1)  % s is nx1
% [d, n] = size(B)

npoints = 200;  % Number of points on the L-curve
smin_ratio = 1000*eps;  % Smallest regularization parameter.
reg_param(npoints) = max([s(n), s(1)*smin_ratio]);  % smallest value of lambda
ratio = (s(1)/reg_param(npoints))^(1/(npoints-1)); 
% values of lambda from smallest to largest
for i=npoints-1:-1:1
    reg_param(i) = ratio*reg_param(i+1); 
end

% if (nargout > 0), locate = 1; else locate = 0; end

beta = U'*b; beta2 = norm(b)^2 - norm(beta)^2;
% s = sm; beta = beta(1:p);
xi = beta(1:p)./s;

%-----------------------
eta = zeros(npoints,1); 
rho = zeros(npoints,1); 
rho = eta; 
reg_param = eta; 
s2 = s.^2;



% values of solution norms and residual norms
for i=1:npoints
    f = s2./(s2 + reg_param(i)^2);
    eta(i) = norm(f.*xi);  % ||x||
    rho(i) = norm((1-f).*beta(1:p));  % ||Ax-b||
end

if (m > n & beta2 > 0)
    rho = sqrt(rho.^2 + beta2); 
end
 
marker = '-'; txt = 'Tikh.';


% Locate the "corner" of the L-curve, if required.
if (locate)
  [reg_corner,rho_c,eta_c] = l_corner(rho,eta,reg_param,U,sm,b,method);
end

% Make plot.
plot_lc(rho,eta,marker,ps,reg_param);
if locate
  ax = axis;
  HoldState = ishold; hold on;
  loglog([min(rho)/100,rho_c],[eta_c,eta_c],':r',[rho_c,rho_c],[min(eta)/100,eta_c],':r')
  title(['L-curve, ',txt,' corner at ',num2str(reg_corner)]);
  axis(ax)
  if (~HoldState), hold off; end
end


