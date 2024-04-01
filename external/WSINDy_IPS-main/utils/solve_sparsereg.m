function [W,resid,its_all,lossvals,thrs_EL,lambda_hat,G_fin,b_fin] = solve_sparsereg(G,b,varargin)

defaultmeth = 'STLS';
defaultkeeprows = 1:size(G,1);
defaultcov = ones(size(G,1),1);
defaultlambda = 10.^(linspace(-4,0,50));
defaultgamma = 0;
defaultM = [];
defaultmaxits = size(G,2);
defaultsparsity_scale=0;
defaultalpha=0.5;
defaultexcl_inds=repmat({[]},1,size(b,2));
defaultAineq=repmat({[]},1,size(b,2));
defaultbineq=repmat({[]},1,size(b,2));
defaulttol=[];
defaultmaxQPits = 1000;
defaultdispQP='off';
defaultmuDR=1;

inp = inputParser;
addParameter(inp,'meth',defaultmeth);
addParameter(inp,'keeprows',defaultkeeprows);
addParameter(inp,'cov',defaultcov);
addParameter(inp,'lambda',defaultlambda);
addParameter(inp,'gamma',defaultgamma);
addParameter(inp,'M',defaultM);
addParameter(inp,'maxits',defaultmaxits);
addParameter(inp,'sparsity_scale',defaultsparsity_scale);
addParameter(inp,'alpha',defaultalpha);
addParameter(inp,'excl_inds',defaultexcl_inds);
addParameter(inp,'Aineq',defaultAineq);
addParameter(inp,'bineq',defaultbineq);
addParameter(inp,'tol',defaulttol);
addParameter(inp,'maxQPits',defaultmaxQPits);
addParameter(inp,'dispQP',defaultdispQP);
addParameter(inp,'muDR',defaultmuDR);

parse(inp,varargin{:});  

meth = inp.Results.meth;
inds_keep = inp.Results.keeprows;
cov = inp.Results.cov;
lambda = inp.Results.lambda;
gamma = inp.Results.gamma;
M = inp.Results.M;
maxits = inp.Results.maxits;
sparsity_scale = inp.Results.sparsity_scale;
alpha = inp.Results.alpha;
excl_inds = inp.Results.excl_inds;
Aineq = inp.Results.Aineq;
bineq = inp.Results.bineq;
tol = inp.Results.tol;
maxQPits = inp.Results.maxQPits;
muDR = inp.Results.muDR;
dispQP = inp.Results.dispQP;

G_fin = (1./cov(inds_keep)).*G(inds_keep,:);
b_fin = (1./cov(inds_keep)).*b(inds_keep,:);

if isequal(meth,'STLS')
    if and(length(lambda)==1,all(lambda>=0))
        [W,resid,its_all,thrs_EL] = wsindy_pde_RGLS(lambda,gamma,G_fin,b_fin,M,maxits);
        lambda_hat = lambda;
        lossvals = [];
    else
        if sparsity_scale ==0
            [W,resid,its_all,lossvals,thrs_EL] = wsindy_pde_RGLS_seq(lambda,gamma,G_fin,b_fin,M,maxits,alpha);
        elseif sparsity_scale ==1
            [W,resid,its_all,lossvals,thrs_EL] = wsindy_pde_RGLS_seq2(lambda,gamma,G_fin,b_fin,M,maxits,alpha);
        end
        lambda_hat = lossvals(min(end,4),lossvals(min(end,4),:)>0);
        lambda_hat = lambda_hat(end);
    end
    if gamma>0
        for i=1:size(b_fin,2)
            if ~isempty(M)
                W(W(:,i)~=0,i) = M(W(:,i)~=0,i).*(G_fin(:,W(:,i)~=0) \ b_fin(:,i));
            else
                W(W(:,i)~=0,i) = G_fin(:,W(:,i)~=0) \ b_fin(:,i);
            end
        end
    end
elseif isequal(meth,'STLSQP')
    [W,resid,its_all,lossvals,thrs_EL] = wsindy_pde_RGLS_seq_qp(lambda,gamma,G_fin,b_fin,M,maxits,alpha,Aineq,bineq,excl_inds,tol,maxQPits,dispQP);
    lambda_hat = lossvals(min(end,4),lossvals(min(end,4),:)>0);
    if ~isempty(lambda_hat)
        lambda_hat = lambda_hat(end);
    else
        lambda_hat=lambda(1);
    end
elseif isequal(meth,'DR')
    [W,its_all] = dougrach(G_fin,b_fin,gamma,lamma,muDR,maxits,tol,alpha,M);
    if ~isempty(M)
        W = W.*M;
    end
elseif isequal(meth,'QR')
    [W,Wcell,res,~,~] = qr_sparse_reg(G_fin,b_fin,M,lambda,gamma,tol);
%     figure(10)
%     subplot(1,2,1)
%     plot(res,'o-')
%     subplot(1,2,2)
%     spy(Wcell{1})
end

if ~isempty(M)
    resid = (G_fin*(W./M)-b_fin)./vecnorm(b_fin); 
else
    resid = (G_fin*W-b_fin)./vecnorm(b_fin); 
end

end


function [W,resid,its_all,thrs_EL] = wsindy_pde_RGLS(lambda,gamma,G,b,M,maxits)

[~,J] = size(G);
[~,num_eq] = size(b);
W = zeros(J,num_eq);

its_all = zeros(num_eq,1);
resid = b*0;
for k=1:num_eq
    if isempty(M)
        [W(:,k),its,thrs_EL] = sparsifyDynamics(G, b(:,k), lambda, gamma, M,maxits);
        resid(:,k) = (b(:,k) - G*W(:,k))/norm(b(:,k)); 
    else
        [W(:,k),its,thrs_EL] = sparsifyDynamics(G, b(:,k), lambda, gamma, M(:,k), maxits);
        resid(:,k) = (b(:,k) - G*(W(:,k)./M(:,k)))/norm(b(:,k)); 
    end
    its_all(k) = its;
end
end

function [W,resid,its_all,lossvals,thrs_EL] = wsindy_pde_RGLS_seq(lambdas,gamma,G,b,M,maxits,alpha)

[~,m] = size(G);
[~,num_eq] = size(b);

W_ls = [G;gamma*eye(m)] \ [b;zeros(m,num_eq)];
GW_ls = norm(G*W_ls);

proj_cost = [];
overfit_cost = [];
lossvals = [];

if isempty(lambdas)
    lam_max = max(max(abs(G'*b),[],2)./vecnorm(G).^2');
    lam_min = min(vecnorm(G*W_ls))/size(G,2)/max(vecnorm(G));
    lambdas = 10.^linspace(log10(lam_min), log10(lam_max),50);
end

if and(length(lambdas)==1,all(lambdas<0))
    lam_max = max(max(abs(G'*b),[],2)./vecnorm(G).^2');
    lam_min = min(vecnorm(G*W_ls))/size(G,2)/max(vecnorm(G));
    lambdas = 10.^linspace(log10(lam_min), log10(lam_max),-lambdas);
end

W = zeros(m,num_eq);
for l=1:length(lambdas)
    lambda = lambdas(l);
    for k=1:num_eq
        if isempty(M)
            [W(:,k),~,~] = sparsifyDynamics(G, b(:,k), lambda, gamma, [], maxits);
        else
            [W(:,k),~,~] = sparsifyDynamics(G, b(:,k), lambda, gamma, M(:,k), maxits);
            W(:,k) = W(:,k)./M(:,k);
        end
    end
    proj_cost = [proj_cost 2*alpha*norm(G*(W-W_ls))/GW_ls];
    overfit_cost = [overfit_cost 2*(1-alpha)*length(find(W~=0))/length(find(W_ls~=0))];
    lossvals = [lossvals proj_cost(end) + overfit_cost(end)];
end

l = find(lossvals == min(lossvals),1);

lambda = lambdas(l);
its_all = zeros(num_eq,1);

resid = b*0;
for k=1:num_eq
    if isempty(M)
        [W(:,k),its,thrs_EL] = sparsifyDynamics(G, b(:,k), lambda, gamma, []);
        resid(:,k) = (b(:,k) - G*W(:,k))/norm(b(:,k)); 
    else
        [W(:,k),its,thrs_EL] = sparsifyDynamics(G, b(:,k), lambda, gamma, M(:,k), maxits);
        resid(:,k) = (b(:,k) - G*(W(:,k)./M(:,k)))/norm(b(:,k)); 
    end
    its_all(k) = its;
end
lossvals = [lossvals;lambdas; [[lossvals(1:l);lambdas(1:l)] zeros(2,length(lambdas)-l)]; proj_cost; overfit_cost];
end

function [W,resid,its_all,lossvals,thrs_EL] = wsindy_pde_RGLS_seq2(lambdas,gamma,G,b,M,maxits,alpha)

[~,m] = size(G);
[~,num_eq] = size(b);
    
W_ls = [G;gamma*eye(m)] \ [b;zeros(m,num_eq)];
GW_ls = norm(G*W_ls);

if length(lambdas)<=1
%     if isempty(lambdas)
%         lambdas = 100;
%     end
%     Wtemp = sort(abs(G'*b./vecnorm(G).^2'));
%     ind = 3;%findchangepts(log10(Wtemp));
%     lambdas = 10.^linspace(log10(mean(Wtemp))-3, log10(mean(Wtemp)),lambdas);
    if isempty(lambdas)
        num_lam = 100;
    else
        num_lam = -lambdas;
    end
    lam_max = min(max(max(abs(G'*b),[],2)./vecnorm(G).^2'),1);
    lam_min = min(vecnorm(G*W_ls))/size(G,2)/max(vecnorm(G));
    lambdas = 10.^linspace(log10(lam_min), log10(lam_max),num_lam);
end

proj_cost = [];
overfit_cost = [];
lossvals = [];

W = zeros(m,num_eq);

for l=1:length(lambdas)
    lambda = lambdas(l);
    for k=1:num_eq
        [W(:,k),~,~] = sparsifyDynamics(G, b(:,k), lambda, gamma, ones(m,1), maxits);
    end    
    proj_cost = [proj_cost 2*alpha*norm(G*(W-W_ls))/GW_ls];
    overfit_cost = [overfit_cost 2*(1-alpha)*length(find(W~=0))/length(W(:))];
    lossvals = [lossvals proj_cost(end) + overfit_cost(end)];
end

l = find(lossvals == min(lossvals),1);

lambda = lambdas(l);
its_all = zeros(num_eq,1);

resid = b*0;
for k=1:num_eq
    [W(:,k),its,thrs_EL] = sparsifyDynamics(G, b(:,k), lambda, gamma, ones(m,1));
    resid(:,k) = (b-G*W(:,k))/norm(b);
    if ~isempty(M)
        W(:,k) = W(:,k).*M(:,k);
    end
    its_all(k) = its;
end
lossvals = [lossvals;lambdas; [[lossvals(1:l);lambdas(1:l)] zeros(2,length(lambdas)-l)]; proj_cost; overfit_cost];
end

function [W,resid,its_all,lossvals,thrs_EL] = wsindy_pde_RGLS_seq_qp(lambdas,gamma,G,b,M,maxits,alpha,A,c,excl_inds,const_tol,max_its,disp_opt)

maxits=min(maxits,size(G,2));

[~,m] = size(G);
[~,num_eq] = size(b);

W_ls = [G;gamma*eye(m)] \ [b;zeros(m,num_eq)];
GW_ls = norm(G*W_ls);

proj_cost = [];
overfit_cost = [];
lossvals = [];

if isempty(lambdas)
    lam_max = max(max(abs(G'*b),[],2)./vecnorm(G).^2');
    lam_min = min(vecnorm(G*W_ls))/size(G,2)/max(vecnorm(G));
    lambdas = 10.^linspace(log10(lam_min), log10(lam_max),50);
end

if and(length(lambdas)==1,all(lambdas<0))
    lam_max = max(max(abs(G'*b),[],2)./vecnorm(G).^2');
    lam_min = min(vecnorm(G*W_ls))/size(G,2)/max(vecnorm(G));
    lambdas = 10.^linspace(log10(lam_min), log10(lam_max),-lambdas);
end

W = zeros(m,num_eq);
for l=1:length(lambdas)
    lambda = lambdas(l);
    for k=1:num_eq
        if isempty(M)
            [W(:,k),~,~] = sparsifyDynamics_qp(G,b(:,k),lambda,gamma,[],A{k},c{k},find(excl_inds{k}),const_tol,max_its,disp_opt,maxits);
        else
            [W(:,k),~,~] = sparsifyDynamics_qp(G,b(:,k),lambda,gamma,M(:,k),A{k},c{k},find(excl_inds{k}),const_tol,max_its,disp_opt,maxits);
            W(:,k) = W(:,k)./M(:,k);
        end
    end
    proj_cost = [proj_cost 2*alpha*norm(G*(W-W_ls))/GW_ls];
    overfit_cost = [overfit_cost 2*(1-alpha)*length(find(W~=0))/length(find(W_ls~=0))];
    lossvals = [lossvals proj_cost(end) + overfit_cost(end)];
end

l = find(lossvals == min(lossvals),1);
lambda = lambdas(l);
its_all = zeros(num_eq,1);

resid = b*0;
for k=1:num_eq
    if isempty(M)
        [W(:,k),its,thrs_EL] = sparsifyDynamics_qp(G,b(:,k),lambda,gamma,[],A{k},c{k},find(excl_inds{k}),const_tol,max_its,disp_opt,maxits);
        resid(:,k) = (b(:,k) - G*W(:,k))/norm(b(:,k)); 
    else
        [W(:,k),its,thrs_EL] = sparsifyDynamics_qp(G,b(:,k),lambda,gamma,M(:,k),A{k},c{k},find(excl_inds{k}),const_tol,max_its,disp_opt,maxits);
        resid(:,k) = (b(:,k) - G*(W(:,k)./M(:,k)))/norm(b(:,k)); 
    end
    its_all(k) = its;
end
lossvals = [lossvals;lambdas; [[lossvals(1:l);lambdas(1:l)] zeros(2,length(lambdas)-l)]; proj_cost; overfit_cost];
end


function [Xi,its,thrs_EL] = sparsifyDynamics_qp(Theta,dXdt,lambda,gamma,M,A,b,excl_inds,const_tol,max_its,disp_opt,max_its_stls)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz
%
% modified by Daniel A. Messenger, 2020 to prevent return of zero vector
% and include regularization
%
% compute Sparse regression: sequential least squares

    options = optimoptions('quadprog','Display',disp_opt,'ConstraintTolerance',const_tol,'MaxIterations',max_its);
    n = min(size(dXdt));
    nn = size(Theta,2);

    dXdt_conj = Theta'*dXdt;
    Theta_conj = Theta'*Theta;

    if  gamma ~= 0
        Theta_conj = Theta_conj+gamma^2*eye(nn);
    end

    Xi = quadprog(Theta_conj,-dXdt_conj,A,b,[],[],[],[],[],options);  % initial guess: Least-squares
    if isempty(M)
        thrs_EL = [];
    else
        Xi = M.*Xi;
        bnds = norm(dXdt)./vecnorm(Theta)'.*M; 
        LBs = lambda*max(1,bnds);
        UBs = 1/lambda*min(1,bnds);
        thrs_EL = [LBs bnds UBs];
    end

    smallinds = 0*Xi;
    its = 0;
    while its < max_its_stls
        if ~isempty(M)
            smallinds_new = or(abs(Xi)<LBs,abs(Xi)>UBs);
%             smallinds_new(excl_inds) = 0;
            if or(length(find(smallinds_new))==nn,all(smallinds_new(:)==smallinds(:)))
                return
            else
                smallinds = smallinds_new;
                Xi(smallinds)=0;
                for ind=1:n
                    biginds = ~smallinds(:,ind);
                    quadprog(Theta_conj(biginds,biginds),-dXdt_conj(biginds,ind),A(:,biginds(1:min(size(A,2),end))),b,[],[],[],[],[],options);
                    Xi(biginds,ind) = M(biginds).*quadprog(Theta_conj(biginds,biginds),-dXdt_conj(biginds,ind),A(:,biginds(1:min(size(A,2),end))),b,[],[],[],[],[],options);
                end
            end
        else
            smallinds_new = (abs(Xi)<lambda);
            smallinds_new(excl_inds) = 0;
            if or(all(smallinds_new(:)==smallinds(:)),length(find(smallinds_new))==length(Xi))
                its = j;
                return
            else
                smallinds = smallinds_new;
                Xi(smallinds)=0;
                for ind = 1:n        
                    biginds = ~smallinds(:,ind);
                    Xi(biginds,ind) = quadprog(Theta_conj(biginds,biginds),-dXdt_conj(biginds,ind),A(:,biginds),b,[],[],[],[],[],options);
                end
            end
        end
        its=its+1;
    end
end


function [Xi,its,thrs_EL] = sparsifyDynamics(Theta,dXdt,lambda,gamma,M,maxits)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz
%
% modified by Daniel A. Messenger, 2020 to prevent return of zero vector
% and include regularization
%
% compute Sparse regression: sequential least squares

n = min(size(dXdt));
nn = size(Theta,2);

if ~exist('maxits','var')
    maxits= nn;
end

if  gamma ~= 0
    Theta_reg = [Theta;gamma*norm(Theta)*eye(nn)];
    dXdt_reg = [dXdt;zeros(nn,n)];
else
    Theta_reg = Theta;
    dXdt_reg = dXdt;
end

Xi = Theta_reg \ dXdt_reg;  % initial guess: Least-squares
if ~isempty(M)
    Xi = M.*Xi;
end

if isempty(M)
    thrs_EL = [];
else
    bnds = norm(dXdt_reg)./vecnorm(Theta_reg)'.*M; 
    LBs = lambda*max(1,bnds);
    UBs = 1/lambda*min(1,bnds);
    thrs_EL = [LBs bnds UBs];
end

smallinds = 0*Xi;
its = 0;
while its < maxits
    if ~isempty(M)
        smallinds_new = or(abs(Xi)<LBs,abs(Xi)>UBs);
        if or(length(find(smallinds_new))==nn,all(smallinds_new(:)==smallinds(:)))
            return
        else
            smallinds = smallinds_new;
            if  gamma ~= 0
                Theta_reg = [Theta;gamma*norm(Theta(:,~smallinds))*eye(nn)];
            end
            Xi(smallinds)=0;    
            for ind=1:n
                Xi(~smallinds,ind) = M(~smallinds).*(Theta_reg(:,~smallinds)\dXdt_reg(:,ind));
            end
        end
    else
        smallinds_new = (abs(Xi)<lambda);
        if or(length(find(smallinds_new))==nn,all(smallinds_new(:)==smallinds(:)))
            return
        else
            smallinds = smallinds_new;
            if  gamma ~= 0
                Theta_reg = [Theta;gamma*norm(Theta(:,~smallinds))*eye(nn)];
            end
            Xi(smallinds)=0;
            for ind = 1:n        
                biginds = ~smallinds(:,ind);
                Xi(biginds,ind) = Theta_reg(:,biginds)\dXdt_reg(:,ind);
            end
        end
    end
    its = its + 1;
end
end

function [w,num_its] = dougrach(A,b,gamma,lambda,mu,maxits,tol,alph,M)

normz = vecnorm(A,inf);
A = A./normz;
[m,n] = size(A);
w = (A \ b)*0+randn(size(A,2),1);

num_its=1;
check=2*tol;

if isempty(M)
    M = w*0+1;
end

while and(num_its < maxits, check>tol)
	wprox = max(abs(w)-gamma*lambda,0).*sign(w);
	wprox = 2*wprox-w;
	wstar = ([sqrt(gamma)*A;eye(n)]) \ [sqrt(gamma)*b; wprox];
	wstar = (1-mu/2)*w+mu/2*(2*wstar-wprox);
% 	wstar = max(abs(wstar)-gamma*lambda,0).*sign(wstar);
	wstar(M.*abs(wstar)./normz'<alph)=0;
	check = norm(wstar-w)/norm(w);
	w=wstar;
	num_its = num_its+1;
end 

for i=1:size(b,2)
inds = w(:,i)~=0;
w(inds,i) = (A(:,inds) \ b(:,i))./normz(inds)';
% smallinds = abs(w)<1/lambda;
% w(smallinds) = 0;
end
end

function [wsparse,Wcell,res,Q,R] = qr_sparse_reg(G,b,M,lambda,gamma,tol)
    [Q,R] = qr(G,0);
    wsparse = zeros(size(G,2),size(b,2));
    Wcell = repmat({R*0},size(b,2),1);
    for nn=1:size(b,2)
        w = Q \ b(:,nn);
        [~,inds] = sort(abs(w),'descend');
        W = [];
        Wbool = w*0;
        res = [];
        restemp = 1;
        i = 1;
        check = tol+1;
        while and(check>tol,i<=size(G,2))
            W = [W w*0];
            res = [res restemp];
            proj = Q(:,inds(i))'*G(:,1:inds(i))./vecnorm(G(:,1:inds(i))).^2;
%             proj = R(inds(i),1:inds(i))./vecnorm(G(:,1:inds(i))).^2;
            [~,a] = max(abs(proj));
            Wbool(:,i) = Wbool(:,end);
            Wbool(a,i) = Wbool(a,i) + 1;
            W(Wbool(:,i)~=0,i) = G(:,Wbool(:,i)~=0) \ b(:,nn);
%             W(Wbool(:,i)~=0,i) = [G(:,Wbool(:,i)~=0);gamma*norm(G(:,Wbool(:,i)~=0))*eye(length(find(Wbool(:,i)~=0)))] \ [b(:,nn);zeros(length(find(Wbool(:,i)~=0)),1)];
            W(Wbool(:,i)~=0,i) = R(:,Wbool(:,i)~=0) \ w;
            restemp = norm(G*W(:,i)-b(:,nn))/norm(b(:,nn));
            if i>=2
                if ~isequal(find(Wbool(:,end)~=0),find(Wbool(:,end-1)~=0))
                    check = res(end-1)-res(end);
                else
                    check = tol+1;
                end
            end
            i = i+1;
        end
        istar = findchangepts(res,'Statistic','linear');
        if isempty(istar)
            istar = size(W,2);
        end
        wsparse_nn = W(:,istar);
        [wsparse_nn(wsparse_nn~=0),~] = sparsifyDynamics(G(:,wsparse_nn~=0),b(:,nn),lambda,gamma,M(wsparse_nn~=0,nn),inf);
        wsparse(:,nn) = wsparse_nn;
        Wcell{nn} = W;
    end
end
% 
%             proj = R(inds(i),1:inds(i))./vecnorm(G(:,1:inds(i))).^2;
%             [~,a] = max(abs(proj));
%             Wbool(:,i) = Wbool(:,end);
%             Wbool(a,i) = Wbool(a,i) + 1;
%             W(Wbool(:,i)~=0,i) = R(:,Wbool(:,i)~=0) \ w;
