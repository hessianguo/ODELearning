
function [KconvX,Kmat] = convNDfftsketch(X,xs,K,svdtol)
    if isempty(svdtol)
        svdtol = 1e-4;%sqrt(eps('double'));
    end
    Ns = size(X);
    dim = length(Ns);
    dx = mean(diff(xs{1}));
    xK = xs(1:dim-1);
    for k=1:dim-1
        xK{k} = xs{k}(:) - min(xs{k});
        inds = repmat({1},1,dim); inds{k} = 2*Ns(k)-1;
        xK{k} = reshape([-flipud(xK{k}(2:end));xK{k}],inds{:});
    end
    [xK{:}] = ndgrid(xK{:});
    Kmat = K(xK{:});
%     Kmat = Kmat*(1/mean(reshape(abs(Kmat),[],1)));
    if dim==2
        KconvX = conv2(X,Kmat*dx,'same');
    elseif dim==3
        l = size(Kmat);
        try 
            [Uk,Sk,Vk] = svdsketch(Kmat,svdtol);
        catch
            [Uk,Sk,Vk] = svdsketch(Kmat);
        end
%         trunc = min(size(Sk,1),4*1.8e9/(size(X,3)*prod(l)*log(prod(l))));
%         norm(Uk(:,1:trunc)*Sk(1:trunc,1:trunc)*Vk(:,1:trunc)'-Kmat)/norm(Kmat)
        trunc = find(vecnorm(diag(Sk) - triu(repmat(diag(Sk),1,size(Sk,1))))/norm(diag(Sk))<svdtol,1);
        KconvX = 0*X;
        for i=1:trunc
%            KconvX = KconvX + Sk(i,i)*fastconv({Uk(:,i)*dx,Vk(:,i)'*dx,[]},X);
            KconvX = KconvX + Sk(i,i)*permute(convn(permute(convn(X,Uk(:,i)*dx,'same'),[2 1 3]),Vk(:,i)*dx,'same'),[2 1 3]);
        end
    end
end
