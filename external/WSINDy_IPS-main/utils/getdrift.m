function [driftx,sigf,funcell] = getdrift(W,customf,tags_pde_G,dim)
    tagends = cellfun(@(x) x(strfind(x,'{'):strfind(x,'}')),tags_pde_G, 'Uni', 0);
    uni=unique(tagends);

    funcell = cell(length(uni),1);
    for i=1:length(uni)
        if dim==2
            funcell{i} = {@(x,t)0*x+0*t, uni{i}};
        elseif dim==3
            funcell{i} = {@(x,y,t)0*x+0*y+0*t, uni{i}};
        end
        inds = find(ismember(tagends,uni(i)));
        for j=1:length(inds)
            tagsf = tags_pde_G{inds(j)};
            coeff= W(ismember(tags_pde_G,{tagsf}));
            if and(coeff~=0,contains(tagsf,'u^1'))
                if dim==2
                    funcell{i}{1} = @(x,t) funcell{i}{1}(x,t)+coeff*eval(tagsf(strfind(tagsf,'[')+1:strfind(tagsf,']')-1));
                elseif dim==3
                    funcell{i}{1} = @(x,y,t) funcell{i}{1}(x,y,t)+coeff*eval(tagsf(strfind(tagsf,'[')+1:strfind(tagsf,']')-1));
                end
            end
        end
        if isequal(uni{i},'{xx}')
            ii=find(ismember(tags_pde_G,{'u^{1}_{xx}'}));
            if ~isempty(ii)
                funcell{i}{1} = @(x,t) funcell{i}{1}(x,t) + W(ii);
            end
            sigf = funcell{i}{1};
        elseif isequal(uni{i},'{lap}')
            ii=find(ismember(tags_pde_G,{'u^{1}_{lap}'}));
            if ~isempty(ii)
                funcell{i}{1} = @(x,y,t) funcell{i}{1}(x,y,t) + W(ii);
            end
            sigf = funcell{i}{1};
        elseif isequal(uni{i},'{x}')
            ii=find(ismember(tags_pde_G,{'u^{1}_{x}'}));
            if ~isempty(ii)
                if dim==2
                    funcell{i}{1} = @(x,t) funcell{i}{1}(x,t) + W(ii);
                elseif dim==3
                    funcell{i}{1} = @(x,y,t) funcell{i}{1}(x,y,t) + W(ii);
                end
            end
            driftx = funcell{i}{1};
        elseif isequal(uni{i},'{y}')
            ii=find(ismember(tags_pde_G,{'u^{1}_{y}'}));
            if ~isempty(ii)
                funcell{i}{1} = @(x,y,t) funcell{i}{1}(x,y,t) + W(ii);
            end
            drifty = funcell{i}{1};
        end
    end
    if ~exist('sigf','var')
        sigf=@(x,t) 0*x;
    end
    if dim==2
        if ~exist('driftx','var')
            driftx=@(x,t) 0*x;
        end
    elseif dim==3
        if ~exist('driftx','var')
            driftx=@(x,t) 0*x;
        end
        if ~exist('drifty','var')
            drifty=@(x,t) 0*x;
        end
    end        
end
    
                
                
%             
% 
% function [driftf,sigf,drift,sig] = getdrift(W,customf,tags_pde_G)
% 
%     drift = {};
%     sig = {};
%     for i=1:length(customf)
%         tagsf = customf{i}{end};
%         coeff= W(ismember(tags_pde_G,{tagsf}));
%         if coeff~=0
%             if contains(tagsf,'u^{1}')
%                 if isequal(tagsf(end-3:end),'{xx}')
%                     sig{end+1} = @(x) coeff*prod(x.^(reshape(eval(tagsf(strfind(tagsf,'['):strfind(tagsf,']')))),2);
%                 elseif isequal(tagsf(end-2:end),'{x}')
%                     drift{end+1} = @(x) coeff*prod(x.^(eval(tagsf(strfind(tagsf,'['):strfind(tagsf,']')))),2);
%                 end
%             end
%         end
%     end
%     
%     sigf = @(x) 0*x+W(ismember(tags_pde_G,{'u^{1}_{xx}'}));
%     for i=1:length(sig)
%         sigf = @(x) sigf(x)+sig{i}(x);
%     end
%     sigf = @(x) sqrt(2*max(sigf(x),0));
%     driftf = @(x) 0*x+W(ismember(tags_pde_G,{'u^{1}_{x}'}));
%     for i=1:length(drift)
%         driftf = @(x) driftf(x)+drift{i}(x);
%     end
% end
%     
%                 
%                 
%             