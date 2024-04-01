
function [tags_pde,lib_list,pdx_list,lhs_ind,max_dx,max_dt,polys,customf,customconv] = get_lib_tags(n,dim,lhs,max_dx,max_dt,polys,trigs,use_cross_dx,use_all_dt,custom_remove,custom_add,drifttags,convargs,true_nz_weight_tags)

[~,lib_list,pdx_list] = build_fcn_lib_tags(n,dim,max_dx,max_dt,polys,trigs,use_cross_dx,use_all_dt);
if and(~isempty(lib_list),~use_all_dt)
    lib_list = lib_list(or(~lib_list(:,end)>0,ismember(lib_list,lhs,'rows')),:);
end

custom_add = [custom_add;lhs];
for j=1:length(true_nz_weight_tags)
    if isequal(class(true_nz_weight_tags),'double')
        custom_add = [custom_add;true_nz_weight_tags{j}(:,1:end-1)];
    end
end
lib_list = unique([lib_list;custom_add],'rows');

inds = [];
for i=1:length(custom_remove)
    if isequal(class(custom_remove{i}),'function_handle')
        inds = unique(custom_remove{i}(lib_list,lhs));
        lib_list = lib_list(~ismember(1:size(lib_list,1),inds),:);
    elseif isequal(class(custom_remove{i}),'double')
        lib_list = lib_list(~ismember(lib_list,custom_remove{i},'rows'),:);
    end
end
[tags_pde,lib_list] = build_str_tags(lib_list,dim,n);


customf = {};
if dim==2
    coords='xt';
elseif dim==3
    coords='xyt';
end
for i=1:length(drifttags)
    tags_dx=drifttags{i}{3};
    str_pdx = '';
    for j=1:length(tags_dx)
        str_pdx=[str_pdx,repelem(coords(j),tags_dx(j))];
    end
    for j=1:size(drifttags{i}{2},1)
        tags = drifttags{i}{2}(j,:);
        str_temp = '';
        for k=1:length(tags)
            if isreal(tags(k))
                str_temp=[str_temp,coords(k),'.^',num2str(tags(k)),'.*'];
            else
                symb = imag(tags(k));
                if symb>0
                    str_temp = [str_temp,'cos(',num2str(abs(symb)),'*',coords(k),').*'];
                else
                    str_temp = [str_temp,'sin(',num2str(abs(symb)),'*',coords(k),').*'];
                end
            end
        end
        str_temp=['[',str_temp(1:end-2),']'];
        customftemp = {{drifttags{i}{1},drifttags{i}{2}(j,:),drifttags{i}{3},['(',str_temp,'u^',num2str(drifttags{i}{1}),')_{',str_pdx,'}']}};
        customf = [customf,customftemp];
    end
end
for i=1:length(customf)
    tags_pde{end+1} = customf{i}{end};
    lib_list(end+1,:) = NaN;
end


customconv={};
for i=1:length(convargs)
    customconv = [customconv,convtags(convargs{i}{:})];
end
for i=1:length(customconv)
    tags_pde{end+1} = customconv{i}{end};
    lib_list(end+1,:) = NaN;
end

max_dx = max([reshape(lib_list(:,n+1:end-1),[],1);...
    reshape(cellfun(@(x)max(x{3}(1:dim-1)),customf),[],1);...
    reshape(cellfun(@(x)max(x{3}(1:dim-1)),customconv),[],1)]);

max_dt = max([reshape(lib_list(:,end),[],1);...
    reshape(cellfun(@(x)max(x{3}(dim)),customf),[],1);...
    reshape(cellfun(@(x)max(x{3}(dim)),customconv),[],1)]);

polys = unique(reshape(real(lib_list(:,1:n)),[],1));
polys = polys(~isnan(polys));

lhs_ind=zeros(size(lhs,1),1);
try
    for k=1:size(lhs,1)
        lhs_ind(k) = find(ismember(lib_list,lhs(k,:),'rows'));
    end
catch
    disp('ERROR: LHS not computed')
    return
end

end
