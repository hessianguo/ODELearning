function [scales,M_scale] = get_scales(U_obs,toggle_scale,polys,ps,ms,max_dx,max_dt,lib_list,xs,customf,customconv,phi_class)
    m_x = min(ms(:,1));
    m_t = min(ms(:,2));
    p_x = max(ps(:,1));
    p_t = max(ps(:,2));
    dx = xs{1}(2)-xs{1}(1);
    dt = xs{end}(2)-xs{end}(1);
    dim = length(xs);
    dims = cellfun(@(x)length(x),xs);
    if or(isequal(toggle_scale,0),isempty(toggle_scale))
        scales = [];
        M_scale = [];
    else
        n = length(U_obs);
        if length(toggle_scale) == 1
            if toggle_scale > 0
                scale_u = zeros(1,n);
                if toggle_scale == 1
                    scale_u_fcn = @(v) norm(v(:)/norm(v(:),1)^(1/max(polys)),max(polys))^(max(polys)/(max(polys)-1));
                elseif toggle_scale == 2
                    scale_u_fcn = @(v) min(max(norm(v(:)/norm(v(:),2)^(1/max(polys)),2*max(polys))^(max(polys)/max(max(polys)-1,1)),eps),1/eps);
                elseif toggle_scale == Inf
                    scale_u_fcn = @(v) norm(v,inf)/(10^(1/max(polys)));
                end
                for k=1:n 
                    scale_u(k) = scale_u_fcn(U_obs{k}(:));
                end
            else
                scale_u = [];
            end
        else
            scale_u = toggle_scale;
        end

        if length(scale_u)~=size(lib_list,2)
            if isequal(phi_class{1},1)
               scale_x = (prod(p_x-(0:floor(max_dx/2)-1))/prod(1:ceil(max_dx/2))*prod(1:max_dx))^(1/max_dx) / (m_x*dx);
            elseif isequal(phi_class{1},2)
               scale_x = 1/(dx*p_x)^2;
            else
                scale_x=1;
            end
            if isequal(phi_class{2},1)
                scale_t = (prod(p_t-(0:floor(max_dt/2)-1))/prod(1:ceil(max_dt/2))*prod(1:max_dt))^(1/max_dt) / (m_t*dt);   % enforce unit inf norm                
            elseif isequal(phi_class{2},2)
                scale_t = 1/(dt*p_t)^2;
            else
                scale_t=1;
            end
            scales = [scale_u repmat(scale_x,1,dim-1) scale_t];
        else
             scales = scale_u;
        end
        M_scale = scales.^(-lib_list(1:end-length(customf)-length(customconv),:));
        M_scale(imag(M_scale)~=0)=1;
        for j=1:length(customf)
            M_scale(end+1,:) = scales.^(-[customf{j}{1} customf{j}{3}+customf{j}{2}.*isreal(customf{j}{2})]);
        end
        for j=1:length(customconv)
            M_scale(end+1,:) = scales.^(-[customconv{j}{1}+customconv{j}{4} customconv{j}{3}]);
        end
        M_scale = prod(M_scale,2);
    end
end

