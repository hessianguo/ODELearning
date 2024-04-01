
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
fprintf(1,'ET_setdisc = %4.4f \n',ET_setdisc);
