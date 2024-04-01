%% solve sparse regression problem
tic;
gamma = (cond(G)>gamma_tol)*min(10.^(-(16+min(log10(lambda)))/2),0.1/norm(G));
if exist('excl_tags','var')
    excl_inds={ismember(tags_pde_G,excl_tags)};
else
    excl_inds={};
end
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