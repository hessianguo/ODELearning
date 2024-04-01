function customftags = get_drifttags(driftpolys,drifttrigs,diffpolys,difftrigs,dim,crossdrift)
    upol=1;
    customftags={};
    diff1=[eye(dim-1) zeros(dim-1,1)];
    drifttags= [];
    for p = 1:length(driftpolys)
        [num,den] = numden(sym(driftpolys(p)));
        drifttags = [drifttags;partitionNk(eval(num),dim-1)/eval(den)];
    end    
    if dim-1==1
        abmat  = drifttrigs(:);
    elseif dim-1==2
        [a,b] = ndgrid(drifttrigs);
        abmat = [a(:) b(:)];
        abmat=abmat(~all(abmat==[0 0],2),:);
    end
    drifttags = [drifttags; abmat*1i];
    drifttags=[drifttags zeros(size(drifttags,1),1)];
    for j=1:size(diff1,1)
        customftags{end+1} = {upol,drifttags,diff1(j,:)};
    end
    
    diff2=[2*eye(dim-1) zeros(dim-1,1)];
    if ~exist('crossdrift','var')
        crossdrift=0;
    end
    if crossdrift
        if dim==3
            diff2=[diff2;[1 1 0]];
        end
    end
    difftags=[];
    for p = 1:length(diffpolys)
        [num,den] = numden(sym(diffpolys(p)));
        difftags = [difftags;partitionNk(eval(num),dim-1)/eval(den)];
    end
    if dim-1==1
        abmat  = difftrigs(:);
    elseif dim-1==2
        [a,b] = ndgrid(difftrigs);
        abmat = [a(:) b(:)];
%         abmat=abmat(~all(abmat==[0 0],2),:);
    end
    difftags = [difftags; abmat*1i];    
    difftags=[difftags zeros(size(difftags,1),1)];
    for j=1:size(diff2,1)
        customftags{end+1} = {upol,difftags,diff2(j,:)};
    end
end