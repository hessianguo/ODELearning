function [sub_inds,ss] = get_subinds(xs_obs,m_x,m_t,sm_x,sm_t)
    dims = cellfun(@(x)length(x),xs_obs);
    dim = length(xs_obs);
    s_x = max(floor(m_x/sm_x),1);max(floor(length(xs_obs{1})/25),1);
    s_t = max(floor(m_t/sm_t),1);max(floor(length(xs_obs{end})/25),1);

    sub_inds = cell(1,dim);
    ss = [repmat(s_x,1,dim-1) s_t];
    mm = [repmat(m_x,1,dim-1) m_t];
    for j=1:dim
        N = dims(j);
        m = mm(j);
        s = ss(j);
        sub_inds{j} = 1:s:N-2*m;
    end
end