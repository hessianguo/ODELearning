%% load data

if any(exps>length(Xscell))
    disp('exps num does not exist,using available exps') 
    exps = unique(min(exps,length(Xscell)));
end
if ~exist('true_nz_weight_tags','var')
    true_nz_weight_tags = [];
end

Xscell(exps)=cellfun(@(x) x(1:NN,:,:),Xscell(exps),'uni',false);
vars={'Xscell',Xscell,'t',t,'numx',numx,'exps',exps,'subsampN',subsampN,...
    'numsdv',numsdv,'coarsen_data',coarsen_data,'custdom',custdom,'Xsnz',Xsnz,...
    'Shift',scoord,'true_nz_weights',true_nz_weight_tags};
tic;
rng('shuffle');
vartot = {'U_exact','xs','lhs','true_nz_weights','pde_num','sigma_NR','noise_dist','noise_alg','Xscell','t',...
    'bw','numx','exps','subsampN','numsdv','coarsen_data','toggle_2ndorder','custdom','Xsnz','Shift'};
varchar = {};
for j=1:length(vars)
    if ischar(vars{j})
        varchar{end+1} = vars{j};
    end
end
varfoo=find(~ismember(vartot,varchar));
for j=1:length(varfoo)
    assignin('base',vartot{varfoo(j)},[]);
end
[U_obs,xs_obs,n,dims,dim,snr,sigma,noise,true_nz_weights,lhs,dx,dt] = load_pde_data_fcn(vars{:});
ET_load_data = toc;
fprintf(1,'ET_load_data = %4.4f \n',ET_load_data);

%% set library

tic,
drifttags = get_drifttags(driftpolys,drifttrigs,diffpolys,difftrigs,dim,crossdrift);
[tags_pde_0,lib_list_0,~,lhs_ind_0,max_dx,max_dt,polys,customf,customconv]  = get_lib_tags(n,dim,lhs,max_dx,max_dt,polys,trigs,use_cross_dx,use_all_dt,custom_remove,custom_add,drifttags,convargs,true_nz_weights);
ET_build_lib = toc;
fprintf(1,'ET_build_lib = %4.4f \n',ET_build_lib);

%% set discretization

tic,
[corners,sig_est] = findcornerpts(U_obs{1},xs_obs);
if isequal(mxs,1)
    k_x = floor(mean(cellfun(@(x)x(end),corners(1:dim-1))));
end
if isequal(mts,1)
    k_t = corners{dim}(end);
end
mx_max=floor(min((cellfun(@(x)length(x),xs_obs(1:end-1))-1)/2)./1);
mx_min=floor(min((cellfun(@(x)length(x),xs_obs(1:end-1))-1)/2)./10);
mt_max=floor(((length(xs_obs{end})-1)/2)./1);
mt_min=floor(((length(xs_obs{end})-1)/2)./10);
[mxs,mts] = ndgrid(mxs,mts);
ms = [mxs(:) mts(:)];
Cfs = {}; sub_indscell = {}; ps = [];
for i=1:size(ms,1)
    m_x = ms(i,1);
    m_t = ms(i,2);
    [phifun,m_x,p_x] = get_phi_handle(min(dims(1:end-1)),'phi_class',phi_class{1},'tau',tau_x,'tauhat',tauhat_x,'m',m_x,'p',p_x,'k',k_x,'maxd',max_dx);
    m_x=max(min(m_x,mx_max),mx_min);
    Cfs_x = phi_weights(phifun,m_x,max_dx);
    [phifun,m_t,p_t] = get_phi_handle(dims(end),'phi_class',phi_class{2},'tau',tau_t,'tauhat',tauhat_t,'m',m_t,'p',p_t,'k',k_t,'maxd',max_dt);
    m_t=max(min(m_t,mt_max),mt_min);
    Cfs_t = phi_weights(phifun,m_t,max_dt);
    Cfs = [Cfs,{{Cfs_x,Cfs_t}}];
    [sub_inds,ss] = get_subinds(xs_obs,m_x,m_t,sm_x,sm_t);
    sub_indscell = [sub_indscell,{sub_inds}];
    ms(i,1) = m_x;
    ms(i,2) = m_t;
    ps = [ps;[p_x p_t]];
end

% Set scales
[scales,M_full_0] = get_scales(U_obs,scales,polys,ps,ms,max_dx,max_dt,lib_list_0,xs_obs,customf,customconv,phi_class);

ET_setdisc = toc;
fprintf(1,'tf support = %4.4f',ms);
fprintf(1,'tf degree = %4.4f',ps);
fprintf(1,'\n ET_setdisc = %4.4f \n',ET_setdisc);

%% Build Library

tic;
[Thetacell,libtree] = get_lib_columns_multiphi(n,lib_list_0,U_obs,Cfs,ms,sub_indscell,dim,scales,xs_obs,customf,customconv);
if ~isempty(true_nz_weights)
    if isequal(class(true_nz_weights{1}),'double')
        axi_0 = tags2axi(true_nz_weights,lib_list_0);
    else
        axi_0 = [];
    end
else
    axi_0 = [];
end

Theta_pdx = [];
for j=1:length(Thetacell)
    Theta_pdx = [Theta_pdx;Thetacell{j}];
end

[Theta_pdx,tags_pde,lib_list,lhs_ind,M_full,axi] = combterms(Theta_pdx,tags_pde_0,...
    lib_list_0,lhs_ind_0,M_full_0,polys,customconv,axi_0,toggle_comb,dim,driftpolys);

num_eq = length(lhs_ind);
[K,m] = size(Theta_pdx);
G = Theta_pdx(:,~ismember(1:m,lhs_ind));
b = zeros(K,num_eq);
M = [];
for k=1:num_eq
    b(:,k) = Theta_pdx(:,lhs_ind(k));
    if ~isempty(M_full)
        M = [M M_full(~ismember(1:m,lhs_ind))/M_full(lhs_ind(k))];
    end
end
tags_pde_G = tags_pde(~ismember(1:length(tags_pde),lhs_ind));
lib_list_G = lib_list(~ismember(1:size(lib_list,1),lhs_ind),:);
if ~isempty(axi)
    axi = axi(~ismember(1:length(tags_pde),lhs_ind));
end
if ~isempty(true_nz_weights)
    if isequal(class(true_nz_weights{1}),'cell')
        axi = str2axi(true_nz_weights,tags_pde_G);
    end
end

% remove rows corresponding to low density, scale rows by approx. variance 
inds_keep=trimrows(trim_tags,trim_fcn,inds_keep_fcn,n,U_obs,Cfs,ms,sub_indscell,dim,scales,xs_obs,G);
if and(covtol>0,covtol<1)
    cov = [];
    for i=1:length(Cfs)
        m_x = ms(i,1);
        m_t = ms(i,2);
        Cfs_x = Cfs{i}{1};
        Cfs_t = Cfs{i}{2};
        sub_inds = sub_indscell{i};
        cov = [cov;get_cov(U_obs,Cfs_x,Cfs_t,sub_inds,m_x,dx,scales,dim,covtol)];
    end
else
    cov = ones(size(G,1),1);
end
ET_build_Gb = toc;
fprintf(1,'ET_build_Gb = %4.4f \n',ET_build_Gb);

%% solve sparse regression problem
tic;
gamma = (cond(G)>gamma_tol)*min(10.^(-(16+min(log10(lambda)))/2),0.1/norm(G));
excl_inds={ismember(tags_pde_G,excl_tags)};
if ~all(cellfun(@(x)all(x==0),excl_inds))
    Aineq={-double(excl_inds{1})};
    if ~isempty(M)
        bineq={-excl_tols./M(excl_inds{1},:)}; 
    else
        bineq={-excl_tols}; 
    end
else
    Aineq=repmat({[]},size(b,2));
    bineq=repmat({[]},size(b,2));
    excl_inds=repmat({[]},size(b,2));
end
vars = {'meth',meth,'keeprows',inds_keep,'cov',cov,'lambda',lambda,'gamma',gamma,...
    'M',M,'maxits',maxits,'sparsity_scale',sparsity_scale,'alpha',alpha,'excl_inds',excl_inds,...
    'Aineq',Aineq,'bineq',bineq,'tol',tol};
[W,resid,its_all,lossvals,thrs_EL,lambda_hat,G_fin,b_fin] = solve_sparsereg(G,b,vars{:});
ET_solve_sparse_reg = toc;
fprintf(1,'ET_solve_sparse_reg = %4.4f \n',ET_solve_sparse_reg);