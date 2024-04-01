function [phifun,m,p] = get_phi_handle(N,varargin)

    default_phi_class = 1;
    default_phi_fun = [];
    default_tau = 10^-10;
    default_tauhat = 2;
    default_m = floor((N-1)/8);
    default_p = [];
    default_k = [];
    default_maxd = 4;
    
    inPars = inputParser;
    validfcnhandle = @(x) isequal(class(x),'function_handle');
    validScalarPosNum = @(x) isnumeric(x) && all(x >=0);
    addParameter(inPars,'phi_class',default_phi_class);
    addParameter(inPars,'phi_fun',default_phi_fun,validfcnhandle);
    addParameter(inPars,'tau',default_tau,validScalarPosNum);
    addParameter(inPars,'tauhat',default_tauhat,validScalarPosNum);
    addParameter(inPars,'m',default_m,validScalarPosNum);
    addParameter(inPars,'p',default_p,validScalarPosNum);
    addParameter(inPars,'k',default_k,validScalarPosNum);
    addParameter(inPars,'maxd',default_maxd,validScalarPosNum);
    parse(inPars,varargin{:});
        
    phi_class = inPars.Results.phi_class;
    phi_fun = inPars.Results.phi_fun;
    tau = inPars.Results.tau;
    tauhat = inPars.Results.tauhat;
    m = inPars.Results.m;
    p = inPars.Results.p;
    k = inPars.Results.k;
    maxd = inPars.Results.maxd;
    
    if ~isempty(phi_fun)
        phifun = phi_fun;
    else
        if phi_class == 1
            if ~isempty(k)
                if isempty(tauhat)
                    tauhat = default_tauhat;
                end
                if ~isempty(p)
                    l = @(m,k,N) log((2*m-1)./m.^2).*((4*pi^2*k^2*m.^2-3*N^2*tauhat^2)-2*N^2*tauhat^2*p);
                    mstar1 = sqrt(3)/pi*N/2/k*tauhat;
                    mstar2 = 1/pi*tauhat*(N/2)/k*sqrt(3+8*p);
                    m = floor(min(fzero(@(m)l(m,k,N), [mstar1 mstar2]),(N-1)/2));
                    tau = ((2*m-1)./m.^2)^p;
                else
                    if isempty(tau)
                        tau = default_tau;
                    end
                    l = @(m,k,N) log((2*m-1)./m.^2).*(4*pi^2*k^2*m.^2-3*N^2*tauhat^2)-2*N^2*tauhat^2*log(tau);
                    mstar1 = sqrt(3)/pi*N/2/k*tauhat;
                    mstar2 = sqrt(log(exp(1)^3/tau^8))/pi*N/2/k*tauhat;
                    m = floor(min(fzero(@(m)l(m,k,N), [mstar1 mstar2]),(N-1)/2));
                    p = max(maxd+1,ceil(log(tau)/log(1-(1-1/m)^2)));
                end
            elseif isempty(p)
                p = ceil(max(log(tau)/log((2*m-1)/m^2),maxd+1));
            else 
            end
            phifun = @(x) (1-x.^2).^p;
        elseif phi_class == 2
            if ~isempty(k)
                m = floor(min(1+N*tauhat/2/pi/k*sqrt(-2*log(tau)),(N-1)/2));
                p = 2*pi*k/tauhat/N;
            elseif ~isempty(tau)
                a = sqrt(-2*log(tau));
                p = (1-1/m)/a;
            end
            phifun = @(x) exp(-(m*p*x).^2/2); % p = dx/sig
        end 
    end
end