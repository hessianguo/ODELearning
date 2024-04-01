function inds_keep=trimrows(trim_tags,trim_fcn,inds_keep_fcn,n,U_obs,Cfs,ms,sub_indscell,dim,scales,xs_obs,G)
    if ~isempty(trim_tags)
        [colcell,~] = get_lib_columns_multiphi(n,trim_tags,U_obs,Cfs,ms,sub_indscell,dim,scales,xs_obs,{},{});
        col = [];
        for j=1:size(colcell{1},2)
            coltemp=[];
            for i=1:length(colcell)
                coltemp=[coltemp;colcell{i}(:,j)]; 
            end
            col = [col coltemp];
        end
        for j=1:length(trim_fcn)
            col=trim_fcn{j}(col);
        end
        inds_keep = inds_keep_fcn(col);
    else
        inds_keep = true(size(G,1),1);
    end
end