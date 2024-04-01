function covfull = get_cov(Ucell,Cfs_x,Cfs_t,sub_inds,m_x,dx,scales,dim,tol)
    n=length(Ucell);
    covfull = [];
    for nn=1:n
        U = Ucell{nn};
        if isempty(scales)
            scales = ones(n+dim,1)';
        end
        phi_x = Cfs_x(1,:).^2;
    %     phi_x = phi_x/norm(phi_x);
        phi_gradx2 = ((m_x*dx*scales(n+1))^(-1)*Cfs_x(2,:)).^2;
    %     phi_gradx2 = phi_gradx2/norm(phi_gradx2);
        phi_t = Cfs_t(1,:).^2;    
    %     phi_t = phi_t/norm(phi_t);
        cols_x = [{phi_gradx2},repmat({phi_x},1,dim-2)];
        cov = zeros(prod(cellfun(@(x) length(x),sub_inds)),1);
        for i=1:dim-1
            cols = [circshift(cols_x,i-1),{phi_t}];        
            cov = cov + reshape(convNDfft(U*scales(n),cols,sub_inds,1),[],1);
        end
        covfull = [covfull sqrt(abs(cov))];
    end
    covfull = covfull/max(covfull);
    covfull = max(covfull,tol);
    covfull = covfull*mean(1./covfull);
end
