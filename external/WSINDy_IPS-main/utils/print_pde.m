
function str = print_pde(w,tags_pde,lhs,digs)
    if ~exist('digs','var')
        digs=6;
    end
    nz_inds = find(w);
    if ~isempty(nz_inds)
        str = [lhs,' = ',num2str(w(nz_inds(1)),digs), tags_pde{nz_inds(1)}];
        for k=2:length(nz_inds)
            str = [str,' + ',num2str(w(nz_inds(k)),digs),tags_pde{nz_inds(k)}];
        end
    else
        str = '';
    end
end
