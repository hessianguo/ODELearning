function u = TVRegDiff( data, iter, alph, u0, scale, ep, dx, plotflag, diagflag )
% u = TVRegDiff( data, iter, alph, u0, scale, ep, dx, plotflag, diagflag );
% u = tvdiff( e, dx, iter, ep, alph );
% Rick Chartrand (rickc@lanl.gov), Apr. 10, 2011
% Please cite Rick Chartrand, "Numerical differentiation of noisy,
% nonsmooth data," ISRN Applied Mathematics, Vol. 2011, Article ID 164564, 
% 2011. 
%
% Inputs:  (First three required; omitting the final N parameters for N < 7
%           or passing in [] results in default values being used.) 
%       data        Vector of data to be differentiated.
%
%       iter        Number of iterations to run the main loop.  A stopping
%                   condition based on the norm of the gradient vector g
%                   below would be an easy modification.  No default value.
%
%       alph        Regularization parameter.  This is the main parameter
%                   to fiddle with.  Start by varying by orders of
%                   magnitude until reasonable results are obtained.  A
%                   value to the nearest power of 10 is usally adequate.
%                   No default value.  Higher values increase
%                   regularization strenght and improve conditioning.
%
%       u0          Initialization of the iteration.  Default value is the
%                   naive derivative (without scaling), of appropriate
%                   length (this being different for the two methods).
%                   Although the solution is theoretically independent of
%                   the intialization, a poor choice can exacerbate
%                   conditioning issues when the linear system is solved.
%
%       scale       'large' or 'small' (case insensitive).  Default is
%                   'small'.  'small' has somewhat better boundary
%                   behavior, but becomes unwieldly for data larger than
%                   1000 entries or so.  'large' has simpler numerics but
%                   is more efficient for large-scale problems.  'large' is
%                   more readily modified for higher-order derivatives,
%                   since the implicit differentiation matrix is square.
%
%       ep          Parameter for avoiding division by zero.  Default value
%                   is 1e-6.  Results should not be very sensitive to the
%                   value.  Larger values improve conditioning and
%                   therefore speed, while smaller values give more
%                   accurate results with sharper jumps.
%
%       dx          Grid spacing, used in the definition of the derivative
%                   operators.  Default is the reciprocal of the data size.
%
%       plotflag    Flag whether to display plot at each iteration.
%                   Default is 1 (yes).  Useful, but adds significant
%                   running time.
%
%       diagflag    Flag whether to display diagnostics at each
%                   iteration.  Default is 1 (yes).  Useful for diagnosing
%                   preconditioning problems.  When tolerance is not met,
%                   an early iterate being best is more worrying than a
%                   large relative residual.
%                   
% Output:
%
%       u           Estimate of the regularized derivative of data.  Due to
%                   different grid assumptions, length( u ) = 
%                   length( data ) + 1 if scale = 'small', otherwise
%                   length( u ) = length( data ).

%% Copyright notice:
% Copyright 2010. Los Alamos National Security, LLC. This material
% was produced under U.S. Government contract DE-AC52-06NA25396 for
% Los Alamos National Laboratory, which is operated by Los Alamos
% National Security, LLC, for the U.S. Department of Energy. The
% Government is granted for, itself and others acting on its
% behalf, a paid-up, nonexclusive, irrevocable worldwide license in
% this material to reproduce, prepare derivative works, and perform
% publicly and display publicly. Beginning five (5) years after
% (March 31, 2011) permission to assert copyright was obtained,
% subject to additional five-year worldwide renewals, the
% Government is granted for itself and others acting on its behalf
% a paid-up, nonexclusive, irrevocable worldwide license in this
% material to reproduce, prepare derivative works, distribute
% copies to the public, perform publicly and display publicly, and
% to permit others to do so. NEITHER THE UNITED STATES NOR THE
% UNITED STATES DEPARTMENT OF ENERGY, NOR LOS ALAMOS NATIONAL
% SECURITY, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY,
% EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
% RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF
% ANY INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR
% REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
% RIGHTS. 

%% BSD License notice:
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions
% are met: 
% 
%      Redistributions of source code must retain the above
%      copyright notice, this list of conditions and the following
%      disclaimer.  
%      Redistributions in binary form must reproduce the above
%      copyright notice, this list of conditions and the following
%      disclaimer in the documentation and/or other materials
%      provided with the distribution. 
%      Neither the name of Los Alamos National Security nor the names of its
%      contributors may be used to endorse or promote products
%      derived from this software without specific prior written
%      permission. 
%  
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
% CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
% INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
% USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
% AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
% LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
% ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE. 

%% code starts here
% Make sure we have a column vector.
data = data( : );
% Get the data size.
n = length( data );

% Default checking. (u0 is done separately within each method.)
if nargin < 9 || isempty( diagflag )
    diagflag = 1;
end
if nargin < 8 || isempty( plotflag )
    plotflag = 1;
end
if nargin < 7 || isempty( dx )
    dx = 1 / n;
end
if nargin < 6 || isempty( ep )
    ep = 1e-6;
end
if nargin < 5 || isempty( scale )
    scale = 'small';
end

% Different methods for small- and large-scale problems.
switch lower( scale )
    
    case 'small'
        % Construct differentiation matrix.
        c = ones( n + 1, 1 ) / dx;
        D = spdiags( [ -c, c ], [ 0, 1 ], n, n + 1 );
        clear c
        DT = D';
        % Construct antidifferentiation operator and its adjoint.
        A = @(x) chop( cumsum( x ) - 0.5 * ( x + x( 1 ) ) ) * dx;
        AT = @(w) ( sum( w ) * ones( n + 1, 1 ) - [ sum( w ) / 2; cumsum( w ) - w / 2 ] ) * dx;
        % Default initialization is naive derivative.
        if nargin < 4 || isempty( u0 )
            u0 = [ 0; diff( data ); 0 ];
        end
        u = u0;
        % Since Au( 0 ) = 0, we need to adjust.
        ofst = data( 1 );
        % Precompute.
        ATb = AT( ofst - data );
        
        % Main loop.
        for ii = 1 : iter
            % Diagonal matrix of weights, for linearizing E-L equation.
            Q = spdiags( 1 ./ ( sqrt( ( D * u ).^2 + ep ) ), 0, n, n );
            % Linearized diffusion matrix, also approximation of Hessian.
            L = dx * DT * Q * D;
            % Gradient of functional.
            g = AT( A( u ) ) + ATb + alph * L * u;
            % Prepare to solve linear equation.
            tol = 1e-4;
            maxit = 100;
            % Simple preconditioner.
            P = alph * spdiags( spdiags( L, 0 ) + 1, 0, n + 1, n + 1 );
            if diagflag
                s = pcg( @(v) ( alph * L * v + AT( A( v ) ) ), g, tol, maxit, P );
                fprintf( 'iteration %4d: relative change = %.3e, gradient norm = %.3e\n', ii, norm( s ) / norm( u ), norm( g ) );
            else
                [ s, ~ ] = pcg( @(v) ( alph * L * v + AT( A( v ) ) ), g, tol, maxit, P );
            end
            % Update solution.
            u = u - s;
            % Display plot.
            if plotflag
                plot( u, 'ok' ), drawnow;
            end
        end
        
    case 'large'
        % Construct antidifferentiation operator and its adjoint.
        A = @(v) cumsum(v);
        AT = @(w) ( sum(w) * ones( length( w ), 1 ) - [ 0; cumsum( w( 1 : end - 1 ) ) ] );
        % Construct differentiation matrix.
        c = ones( n, 1 );
        D = spdiags( [ -c c ], [ 0 1 ], n, n ) / dx;
        D( n, n ) = 0;
        clear c
        DT = D';
        % Since Au( 0 ) = 0, we need to adjust.
        data = data - data( 1 );
        % Default initialization is naive derivative.
        if nargin < 4 || isempty( u0 )
            u0 = [ 0; diff( data ) ];
        end
        u = u0;
        % Precompute.
        ATd = AT( data );
        
        % Main loop.
        for ii = 1 : iter
            % Diagonal matrix of weights, for linearizing E-L equation.
            Q = spdiags( 1./ sqrt( ( D * u ).^2 +  ep ), 0, n, n );
            % Linearized diffusion matrix, also approximation of Hessian.
            L = DT * Q * D;
            % Gradient of functional.
            g = AT( A( u ) ) - ATd;
            g= g + alph * L * u;
            % Build preconditioner.
            c = cumsum( n : -1 : 1 ).';
            B = alph * L + spdiags( c( end : -1 : 1 ), 0, n, n );
            droptol = 1.0e-2;
            R = cholinc( B, droptol );
            % Prepare to solve linear equation.
            tol = 1.0e-4;
            maxit = 100;
            if diagflag
                s = pcg( @(x) ( alph * L * x + AT( A( x ) ) ), -g, tol, maxit, R', R );
                fprintf( 'iteration %2d: relative change = %.3e, gradient norm = %.3e\n', ii, norm( s ) / norm( u ), norm( g ) );
            else
                [ s, ~ ] = pcg( @(x) ( alph * L * x + AT( A( x ) ) ), -g, tol, maxit, R', R );
            end
            % Update current solution
            u = u + s;
            % Display plot.
            if plotflag
                plot( u, 'ok' ), drawnow;
            end
        end
end

% Utility function.
function w = chop( v )
w = v( 2 : end );

function h = color_line3(x, y, z, c, varargin)
% color_line3 plots a 3-D "line" with c-data as color
%
%       h = color_line(x, y, z, c)
%       by default: 'LineStyle','-' and 'Marker','none'
%
%          or
%       h = color_line(x, y, z, c, mark) 
%          or
%       h = color_line(x, y, z, c, 'Property','value'...) 
%             with valid 'Property','value' pairs for a surface object
%
%  in:  x      x-data
%       y      y-data
%       z      z-data
%       c      4th dimension for colouring
%       mark   for scatter plots with no connecting line
%
% out:  h   handle of the surface object


h = surface(...
  'XData',[x(:) x(:)],...
  'YData',[y(:) y(:)],...
  'ZData',[z(:) z(:)],...
  'CData',[c(:) c(:)],...
  'FaceColor','none',...
  'EdgeColor','flat',...
  'Marker','none');
  
if nargin ==5
    switch varargin{1}
        case {'+' 'o' '*' '.' 'x' 'square' 'diamond' 'v' '^' '>' '<' 'pentagram' 'p' 'hexagram' 'h'}
            set(h,'LineStyle','none','Marker',varargin{1})
        otherwise
            error(['Invalid marker: ' varargin{1}])
    end

elseif nargin > 5
    set(h,varargin{:})
end


function dy = hopf(t,y,mu,omega,A)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

dy = [
mu*y(1) - omega*y(2) - A*y(1)*(y(1)^2+y(2)^2);
omega*y(1) + mu*y(2) - A*y(2)*(y(1)^2+y(2)^2);
];

function dy = logistic(t,y,r)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

dy = [
    r*y*(1-y);
];


function dy = lorenz(t,y,sigma,beta,rho)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

dy = [
sigma*(y(2)-y(1));
y(1)*(rho-y(3))-y(2);
y(1)*y(2)-beta*y(3);
];

function yout = poolData(yin,nVars,polyorder,usesine)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

n = size(yin,1);
% yout = zeros(n,1+nVars+(nVars*(nVars+1)/2)+(nVars*(nVars+1)*(nVars+2)/(2*3))+11);

ind = 1;
% poly order 0
yout(:,ind) = ones(n,1);
ind = ind+1;

% poly order 1
for i=1:nVars
    yout(:,ind) = yin(:,i);
    ind = ind+1;
end

if(polyorder>=2)
    % poly order 2
    for i=1:nVars
        for j=i:nVars
            yout(:,ind) = yin(:,i).*yin(:,j);
            ind = ind+1;
        end
    end
end

if(polyorder>=3)
    % poly order 3
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k);
                ind = ind+1;
            end
        end
    end
end

if(polyorder>=4)
    % poly order 4
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                for l=k:nVars
                    yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k).*yin(:,l);
                    ind = ind+1;
                end
            end
        end
    end
end

if(polyorder>=5)
    % poly order 5
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                for l=k:nVars
                    for m=l:nVars
                        yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k).*yin(:,l).*yin(:,m);
                        ind = ind+1;
                    end
                end
            end
        end
    end
end

if(usesine)
    for k=1:10;
        yout = [yout sin(k*yin) cos(k*yin)];
    end
end

function yout = poolDataLIST(yin,ahat,nVars,polyorder,usesine)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

n = size(yin,1);

ind = 1;
% poly order 0
yout{ind,1} = ['1'];
ind = ind+1;

% poly order 1
for i=1:nVars
    yout(ind,1) = yin(i);
    ind = ind+1;
end

if(polyorder>=2)
    % poly order 2
    for i=1:nVars
        for j=i:nVars
            yout{ind,1} = [yin{i},yin{j}];
            ind = ind+1;
        end
    end
end

if(polyorder>=3)
    % poly order 3
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                yout{ind,1} = [yin{i},yin{j},yin{k}];
                ind = ind+1;
            end
        end
    end
end

if(polyorder>=4)
    % poly order 4
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                for l=k:nVars
                    yout{ind,1} = [yin{i},yin{j},yin{k},yin{l}];
                    ind = ind+1;
                end
            end
        end
    end
end

if(polyorder>=5)
    % poly order 5
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                for l=k:nVars
                    for m=l:nVars
                        yout{ind,1} = [yin{i},yin{j},yin{k},yin{l},yin{m}];
                        ind = ind+1;
                    end
                end
            end
        end
    end
end

if(usesine)
    for k=1:10;
        yout{ind,1} = ['sin(',num2str(k),'*yin)'];
        ind = ind + 1;
        yout{ind,1} = ['cos(',num2str(k),'*yin)'];
        ind = ind + 1;
    end
end


output = yout;
newout(1) = {''};
for k=1:length(yin)
    newout{1,1+k} = [yin{k},'dot'];
end
% newout = {'','xdot','ydot','udot'};
for k=1:size(ahat,1)
    newout(k+1,1) = output(k);
    for j=1:length(yin)
        newout{k+1,1+j} = ahat(k,j);
    end
end
newout


function dy = sparseGalerkin(t,y,ahat,polyorder,usesine)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

yPool = poolData(y',length(y),polyorder,usesine);
dy = (yPool*ahat)';


function ykplus1 = sparseGalerkinDiscrete(t,y,ahat)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

yPool = poolData1D(y',length(y));
ykplus1 = (yPool*ahat)';


function Xi = sparsifyDynamics(Theta,dXdt,lambda,n)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

% compute Sparse regression: sequential least squares
Xi = Theta\dXdt;  % initial guess: Least-squares

% lambda is our sparsification knob.
for k=1:10
    smallinds = (abs(Xi)<lambda);   % find small coefficients
    Xi(smallinds)=0;                % and threshold
    for ind = 1:n                   % n is state dimension
        biginds = ~smallinds(:,ind);
        % Regress dynamics onto remaining terms to find sparse Xi
        Xi(biginds,ind) = Theta(:,biginds)\dXdt(:,ind); 
    end
end