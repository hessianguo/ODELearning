function [tags,lib_list,pdx_list] = build_fcn_lib_tags(n,dim,max_dx,max_dt,polys,trigs,use_cross_dx,use_all_dt)
tags = [];
for p = 1:length(polys)
    [num,den] = numden(sym(polys(p)));
    tags = [tags;partitionNk(eval(num),n)/eval(den)];
end
for k=1:length(trigs)
    trig_inds = [-trigs(k)*1i*eye(n);trigs(k)*1i*eye(n)];
    tags = [tags; trig_inds];
end

if and(use_cross_dx,use_all_dt)
    inds_grid = cell(dim,1);
    for d=1:dim-1
        inds_grid{d} = 1:max_dx+1;
    end
    inds_grid{dim} = 1:max_dt+1;
    [inds_grid{:}] = ndgrid(inds_grid{:});

    pdxs = (0:max(max_dx,max_dt))';
    pdx_list = [];
    for d=1:dim-1
        pdx_list = [pdx_list pdxs(inds_grid{d},:)];
    end
    pdx_list = [pdx_list pdxs(inds_grid{end},:)];
elseif and(use_cross_dx,~use_all_dt)
    inds_grid = cell(dim-1,1);
    for d=1:dim-1
        inds_grid{d} = 1:max_dx+1;
    end
    [inds_grid{:}] = ndgrid(inds_grid{:});

    pdxs = (0:max_dx)';
    pdx_list = [];
    for d=1:dim-1
        pdx_list = [pdx_list pdxs(inds_grid{d},:)];
    end
    pdx_list = [zeros(1,dim);[pdx_list [max_dt;zeros(size(pdx_list,1)-1,1)]]];
elseif and(~use_cross_dx,use_all_dt)    
    pdx_list = zeros(1,dim);
    for k=1:max_dt                                                        
        pdx_list = [pdx_list;[zeros(1,dim-1) k]];
    end
    for k=1:dim-1
        for j=1:max_dx
            pdx_list = [pdx_list;[zeros(1,k-1) j zeros(1,dim-k)]];
        end
    end
elseif and(~use_cross_dx,~use_all_dt)
    pdx_list = zeros(1,dim);
    pdx_list = [pdx_list;[zeros(1,dim-1) max_dt]];
    for k=1:dim-1
        for j=1:max_dx
            pdx_list = [pdx_list;[zeros(1,k-1) j zeros(1,dim-k)]];
        end
    end
end
if any(ismember(polys,0))
    lib_list = zeros((size(tags,1)-1)*size(pdx_list,1),size(tags,2)+size(pdx_list,2));    
    for i=2:size(tags,1)
        for j=1:size(pdx_list,1)
            lib_list(1+(i-2)*size(pdx_list,1)+j,:) = [tags(i,:) pdx_list(j,:)];
        end
    end
else
    [a,b] = ndgrid(1:size(tags,1),1:size(pdx_list,1));
    lib_list = [tags(a,:) pdx_list(b,:)];
end
end

