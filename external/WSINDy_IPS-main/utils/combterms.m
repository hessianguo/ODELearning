function [Theta_pdx,tags_pde,lib_list,lhs_ind,M_full,axi] = combterms(Theta_pdx,tags_pde,lib_list,lhs_ind,M_full,polys,customconv,axi,toggle_comb,dim,driftpolys,drifttrigs)

    combcell = {};
    if and(dim==3,toggle_comb==1)
        tagends = cellfun(@(x) x(strfind(x,'_{')+1:end), tags_pde, 'Uni', 0);
        inds=find(ismember(tagends,{'{xx}'}));
        for i=1:length(inds)
            str=tags_pde{inds(i)}(1:end-5);
            combcell = [combcell,{{{[str,'_{xx}'],[str,'_{yy}']},[str,'_{lap}'],[1 1]}}];
        end        
%         for i=1:length(polys)
%             combcell = [combcell,{{{['u^{',num2str(polys(i)),'}_{xx}'],['u^{',num2str(polys(i)),'}_{yy}']},['u^{',num2str(polys(i)),'}_{lap}'],[1 1]}}];
% %             combcell = [combcell,{{{['u^{',num2str(polys(i)),'}_{xxxx}'],['u^{',num2str(polys(i)),'}_{xxyy}'],['u^{',num2str(polys(i)),'}_{yyyy}']},['u^{',num2str(polys(i)),'}_{bilap}']}}];
%         end
        convpair = reshape(tags_pde(size(lib_list,1)-length(customconv)+1:end)',2,[])';
        for i=1:size(convpair,1)
            combcell = [combcell,{{{convpair{i,1},convpair{i,2}},erase(convpair{i,1},'x'),[1 1]}}];
        end
        customfpolys=[];
        for p = 1:length(driftpolys)
            [num,den] = numden(sym(driftpolys(p)+1));
            customfpolys = [customfpolys;partitionNk(eval(num),dim-1)/eval(den)];
        end
        coords='xyt';
        utag=num2str(1);
        fstr=@(tags,dx) ['[x.^',num2str(tags(1)),'.*y.^',num2str(tags(2)),'.*t.^0]u^',utag,')_{',dx,'}'];
        fstr2=@(tags) ['(grad[x.^',num2str(tags(1)),'.*y.^',num2str(tags(2)),'.*t.^0]u^',utag,')_{div}'];
        for j=1:size(customfpolys,1)
            tags=customfpolys(j,:);
            combpair={};
            for d=1:dim-1
                tagmin=tags;
                if tags(d)~=0
                    tagmin(d)=tagmin(d)-1;
                    combpair{end+1}=fstr(tagmin,coords(d));
                end
            end
            combcell{end+1}={combpair,fstr2(tags),tags(tags~=0)};
        end
    end
    
    [~,J] = size(Theta_pdx);
    c = length(combcell);
    inds_all = [];
    newcols = [];
    newtags = {};
    newscales = [];
    newaxi = [];
    newlibs = [];
    for j=1:c
        inds = find(ismember(tags_pde,combcell{j}{1}));
%         if length(inds)==length(combcell{j}{1})
        if ~isempty(inds)
            newtags = [newtags,{combcell{j}{2}}];
            newcols = [newcols sum(Theta_pdx(:,inds).*combcell{j}{3},2)];
            if ~isempty(M_full)
                newscales = [newscales;M_full(inds(1))];
            end
            inds_all = unique([inds_all(:);inds(:)]);
            if ~isempty(axi)
                newaxi = [newaxi;axi(inds(1),:)];
            end
            newlibs = [newlibs;NaN*ones(1,size(lib_list,2))];
        end
    end
    inds_retain = find(~ismember(1:J,inds_all));
    for i=1:length(lhs_ind)
        lhs_ind(i) = lhs_ind(i) - sum(inds_all<lhs_ind(i));
    end
    Theta_pdx = [Theta_pdx(:,inds_retain) newcols];
    if ~isempty(M_full)
        M_full = [M_full(inds_retain);newscales];
    end
    tags_pde = [tags_pde(inds_retain),newtags];
    lib_list = [lib_list(inds_retain,:);newlibs];
    if ~isempty(axi)
        axi = [axi(inds_retain,:);newaxi];
    end
end


