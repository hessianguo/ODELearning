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

