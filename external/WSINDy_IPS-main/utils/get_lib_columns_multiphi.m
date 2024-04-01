function [Theta_pdx,libtree] = get_lib_columns_multiphi(n,lib_list,U_obs,Cfs,ms,sub_inds,dim,scales,xs,customf,customconv)

    dx = mean(cellfun(@(x)mean(diff(x)),xs(1:end-1)));
    dt = mean(diff(xs{end}));
    Ns = size(U_obs{1});
    if isempty(scales)
        scales = ones(n+dim,1);
    end

    LK = length(Cfs);
    Cfs_ffts = repmat(cell(dim,1),1,LK);
    for ll=1:LK
        Cfs_x = Cfs{ll}{1};
        Cfs_t = Cfs{ll}{2};
        m_x = ms(ll,1);
        m_t = ms(ll,2);
        [mm,nn] = size(Cfs_x);
        for k=1:dim-1
            Cfs_ffts{ll}{k} = [zeros(mm,Ns(k)-nn) (m_x*dx*scales(n+k)).^(-(0:mm-1)').*Cfs_x];
            Cfs_ffts{ll}{k} = fft(Cfs_ffts{ll}{k},[],2);
        end
        [mm,nn] = size(Cfs_t);
        Cfs_ffts{ll}{dim} = [zeros(mm,Ns(dim)-nn) (m_t*dt*scales(n+dim)).^(-(0:mm-1)').*Cfs_t];
        Cfs_ffts{ll}{dim} = fft(Cfs_ffts{ll}{dim},[],2);
    end
    
    libtree = liblisttree(lib_list(1:end-length(customf)-length(customconv),:),n);

    Theta_pdx = {};
    for ll=1:LK
        Theta_pdx{ll} = zeros(prod(cellfun(@(x)length(x),sub_inds{ll})),size(lib_list,1));
    end

    for i=1:length(libtree)
        fcn = ones(size(U_obs{1}));
        tags_root = zeros(1,n);
        indout = libtree{i};
        while ~isempty(indout)
            tags = lib_list(indout(1),1:n);
            sametags = indout(ismember(lib_list(indout,1:n),tags,'rows'));
            for k=1:n
                if isreal(tags(k))
                    fcn = fcn.*(U_obs{k} / scales(k)).^(tags(k)-tags_root(k));
                else
                    if imag(tags(k))<0
                        fcn = sin(abs(imag(tags(k)))*U_obs{k});
                    else
                        fcn = cos(abs(imag(tags(k)))*U_obs{k});
                    end
                end
            end
            for ind = sametags
                for ll=1:LK
                    test_conv_cell = {};
                    for k=1:dim
                        test_conv_cell{k} = Cfs_ffts{ll}{k}(lib_list(ind,n+k)+1,:);
                    end
                    fcn_conv = convNDfft(fcn,test_conv_cell,sub_inds{ll},2);
                    Theta_pdx{ll}(:,ind) = fcn_conv(:);
                end
            end
            indout = indout(~ismember(indout,sametags));
            tags_root = tags;     
        end
    end
    
% ind =1;
% while ind<=size(lib_list,1)-length(customf)-length(customconv)
%     tags = lib_list(ind,1:n);
%     fcn = ones(size(U_obs{1}));
%     for k=1:n
%         if isreal(tags(k))
%             fcn = fcn.*((U_obs{k} / scales(k)).^tags(k));
%         else
%             if imag(tags(k))<0
%                 fcn = sin(abs(imag(tags(k)))*U_obs{k});
%             else
%                 fcn = cos(abs(imag(tags(k)))*U_obs{k});
%             end
%         end
%     end
%     while all(lib_list(ind,1:n) == tags)
%         test_conv_cell = {};
%         for k=1:dim
%             test_conv_cell{k} = Cfs_ffts{k}(lib_list(ind,n+k)+1,:);
%         end
%         fcn_conv = convNDfft(fcn,test_conv_cell,sub_inds,2);
%         Theta_pdx(:,ind) = fcn_conv(:);
%         ind = ind+1;
%         if ind > size(lib_list,1)
%             break
%         end
%     end
% end

    for i=1:length(xs)
        L = length(xs{i});
        empty_inds = repmat({1},1,length(xs));
        empty_inds{i} = L;
        xs{i} = reshape(xs{i},empty_inds{:});
    end

    Theta_customf = {};
    for ll=1:LK
        Theta_customf{ll} = zeros(size(Theta_pdx{ll},1),length(customf));
    end
    for ind =1:length(customf)
        u_tags = customf{ind}{1};
        fcn = ones(size(U_obs{1}));
        for k=1:n
            if isreal(u_tags(k))
                fcn = fcn.*(U_obs{k} / scales(k)).^(u_tags(k));
            else
                if imag(u_tags(k))<0
                    fcn = sin(abs(imag(u_tags(k)))*U_obs{k});
                else
                    fcn = cos(abs(imag(u_tags(k)))*U_obs{k});
                end
            end
        end
        x_tags = customf{ind}{2};
        for k=1:dim
            if isreal(x_tags(k))
                fcn = fcn.*((xs{k} / scales(n+k)).^x_tags(k));
            else
                if imag(x_tags(k))<0
                    fcn = fcn.*sin(abs(imag(x_tags(k)))*xs{k});
                else
                    fcn = fcn.*cos(abs(imag(x_tags(k)))*xs{k});
                end
            end
        end

        psi_tags = customf{ind}{3};
        for ll=1:LK
            test_conv_cell = {};
            for k=1:dim
                test_conv_cell{k} = Cfs_ffts{ll}{k}(psi_tags(k)+1,:);
            end
            fcn_conv = convNDfft(fcn,test_conv_cell,sub_inds{ll},2);
            Theta_customf{ll}(:,ind) = fcn_conv(:);
        end
    end
    for ll=1:LK
        Theta_pdx{ll}(:,1+size(lib_list,1)-length(customf)-length(customconv):size(lib_list,1)-length(customconv)) = Theta_customf{ll};
    end
    
    Theta_conv = {};
    for ll=1:LK
        Theta_conv{ll} = zeros(size(Theta_pdx{ll},1),length(customconv));
    end
    for ind=1:length(customconv)
        u_tags = customconv{ind}{1};
        K = customconv{ind}{2};
        psi_tags = customconv{ind}{3};
        v_tags = customconv{ind}{4};
        svdtol = customconv{ind}{5};
        fcn = (U_obs{1} / scales(1)).^u_tags(1);
        for k=2:n
            if u_tags(k)~=0
                fcn = fcn.*((U_obs{k} / scales(k)).^u_tags(k));
            end
        end
        fcn = convNDfftsketch(fcn,xs,K,svdtol);
        for k=1:n
            if v_tags(k)~=0
                fcn = fcn.*((U_obs{k} / scales(k)).^v_tags(k));
            end
        end
        for ll=1:LK
            test_conv_cell = {};
            for k=1:dim
                test_conv_cell{k} = Cfs_ffts{ll}{k}(psi_tags(k)+1,:);
            end
            fcn_conv = convNDfft(fcn,test_conv_cell,sub_inds{ll},2);
            Theta_conv{ll}(:,ind) = fcn_conv(:);
        end
    end
    for ll=1:LK
        Theta_pdx{ll}(:,1+size(lib_list,1)-length(customconv):end) = Theta_conv{ll};
    end

end
