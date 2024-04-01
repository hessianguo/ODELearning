
function c = liblisttree(lib_list,n)

    c = {};
    indout = 1:size(lib_list,1);
    L = size(lib_list,1);

    while sum(cellfun(@(x)length(x),c))<L
        rootind = indout(1);
        indout = indout(indout~=rootind);
        r1 = lib_list(rootind,1:n);
        branch = rootind;
        for j=indout
            r2 = lib_list(j,1:n);
            if linkrows(r1,r2)
                branch = [branch j];
                indout = indout(indout~=j);
                r1=r2;
            end
        end
        c = [c,{branch}];
    end
end    


function bool = linkrows(r1,r2)
    bool = all(r1<=r2) && isreal(r1) && isreal(r2) && all(~isnan(r1)) && all(~isnan(r2));
end
