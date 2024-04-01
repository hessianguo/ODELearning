function [U_exact,xs,lhs,true_nz_weights,dx,dt,Ntot] = hist1storder(Xscell,t,varargin)

    defaultnumx = 2^7+1;
    defaultexps = 1;
    defaultbw = 0;
    defaultnumsdv = 4;
    defaultsubsamp = 1;
    defaultcustdom = [];
    defaultnoise = [0 1];

    p = inputParser;
    addParameter(p,'numx',defaultnumx);
    addParameter(p,'exps',defaultexps);
    addParameter(p,'bw',defaultbw);
    addParameter(p,'numsdv',defaultnumsdv);
    addParameter(p,'subsamp',defaultsubsamp);
    addParameter(p,'custdom',defaultcustdom);
    addParameter(p,'noise',defaultnoise);
    
    parse(p,varargin{:});   

    numx = p.Results.numx;
    exps = p.Results.exps;
    bw = p.Results.bw;
    numsdv = p.Results.numsdv;
    subsamp = p.Results.subsamp;
    custdom = p.Results.custdom;
    nz = p.Results.noise;

    [~,d,~] = size(Xscell{1});
    numt = length(t);
    dt = t(2)-t(1);
    
    true_nz_weights = {};
    L = length(exps);
    Ns = floor(subsamp*cellfun(@(x) size(x,1), Xscell(exps)));
    Nscum = [0 cumsum(Ns)];
    Ntot = Nscum(end);
    Xall = zeros(Nscum(end),d,numt);
    for j=1:L
        X = Xscell{exps(j)};
        [N,~,~] = size(X);
        subinds = randperm(N,Ns(j));
        X = X(subinds,:,:);
        if nz(1)>0
            subsubs = randperm(length(subinds),floor(length(subinds)*nz(2)));
            X(subsubs,:,:) = X(subsubs,:,:) + nz(1)*rms(reshape(X(subsubs,:,:),[],1))*randn(size(X(subsubs,:,:)));
        end
        Xall(Nscum(j)+1:Nscum(j+1),:,:)=X;
    end

    if d ==1
        if isempty(custdom)
            mux = mean(cellfun(@(x)mean(reshape(x(:,1,:),[],1)),Xscell(exps)));
            stdx = mean(cellfun(@(x)std(reshape(x(:,1,:),[],1)),Xscell(exps)));
            minX = min(cellfun(@(x)min(reshape(x(:,1,:),[],1)),Xscell(exps)));
            maxX = max(cellfun(@(x)max(reshape(x(:,1,:),[],1)),Xscell(exps)));
            x = linspace(max(mux-numsdv*stdx,minX),min(mux+numsdv*stdx,maxX),numx);
        else
            x = linspace(custdom(1),custdom(2),numx);
        end
        dx = x(2)-x(1);
        lhs = [1 0 1];
        xs = {(x(1:end-1)+x(2:end))/2,t};
        U_exact = {zeros(length(x)-1,length(t))};       
        for tt=1:numt
            if bw == 0
                U_exact{1}(:,tt) = histcounts(Xall(:,:,tt),x,'normalization','pdf');
            elseif bw >0
                [U_exact{1}(:,tt),~,~] = ksdensity(Xall,xs{1},'kernel','epanechnikov','bandwidth',bw);
            end
        end        
    elseif d==2
        if isempty(custdom)
            mux = mean(cellfun(@(x)mean(reshape(x(:,1,:),[],1)),Xscell(exps)));
            stdx = mean(cellfun(@(x)std(reshape(x(:,1,:),[],1)),Xscell(exps)));
            minX = min(cellfun(@(x)min(reshape(x(:,1,:),[],1)),Xscell(exps)));
            maxX = max(cellfun(@(x)max(reshape(x(:,1,:),[],1)),Xscell(exps)));
            muy = mean(cellfun(@(x)mean(reshape(x(:,2,:),[],1)),Xscell(exps)));
            stdy = mean(cellfun(@(x)std(reshape(x(:,2,:),[],1)),Xscell(exps)));
            minY = min(cellfun(@(x)min(reshape(x(:,2,:),[],1)),Xscell(exps)));
            maxY = max(cellfun(@(x)max(reshape(x(:,2,:),[],1)),Xscell(exps)));
            x = linspace(max(mux-numsdv*stdx,minX),min(mux+numsdv*stdx,maxX),numx);
            dx = mean(diff(x));
            y = max((muy-numsdv*stdy),minY):dx:min((muy+numsdv*stdy),maxY);
        else
            x = linspace(custdom(1),custdom(2),numx);
            y = x;
            dx = mean(diff(x));
        end
        lhs = [1 0 0 1];
        xs = {(x(1:end-1)+x(2:end))/2,(y(1:end-1)+y(2:end))/2,t};
        true_nz_weights = {[]};
        L = length(exps);
        if bw>0
            [xkd,ykd]= meshgrid(xs{1:2});
        end
        U_exact = {zeros(length(x)-1,length(y)-1,length(t))};
        for tt=1:numt
            if bw == 0
                U_exact{1}(:,:,tt) = histcounts2(Xall(:,1,tt),Xall(:,2,tt),x,y,'normalization','pdf');
            elseif bw >0
                Utemp = ksdensity(Xall(:,:,tt),[xkd(:) ykd(:)],'kernel','epanechnikov','bandwidth',bw);
                U_exact{1}(:,:,tt) = reshape(Utemp,length(xs{2}),[])';
            end
        end        
    end
end
