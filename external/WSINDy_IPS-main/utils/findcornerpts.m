function [corners,sig_est] = findcornerpts(U_obs,xs)
    dims = size(U_obs);
    dim = length(dims);
    corners = cell(dim,1);
    for d=1:dim
        if dim ==1
            shift = [1 2];
        else
            shift = circshift(1:dim,1-d);
        end
        dim_perm = dims(shift);
        x = xs{d}(:);
        L = length(x);
        wn = ((0:L-1)-floor(L/2))'*(2*pi)/range(x);
        xx = wn(1:ceil(end/2));
        NN = length(xx);
        Ufft = abs(fftshift(fft(permute(U_obs,shift))));       
        if dim>2
            Ufft = reshape(Ufft,[L,prod(dim_perm(2:end))]);
        end
        Ufft = mean(Ufft,2);
        Ufft = cumsum(Ufft);            
        Ufft = Ufft(1:ceil(L/2),:);
        errs = zeros(NN,1);
        for k=1:NN
           subinds1 = 1:k;
           subinds2 = k:NN;
           Ufft_av1 = Ufft(subinds1);
           Ufft_av2 = Ufft(subinds2);
           m1 = range(Ufft_av1)/range(xx(subinds1));
           m2 = range(Ufft_av2)/range(xx(subinds2));
           L1 = min(Ufft_av1)+m1*(xx(subinds1)-xx(1));
           L2 = max(Ufft_av2)+m2*(xx(subinds2)-xx(end));
           errs(k) = sqrt(sum(((L1-Ufft_av1)./Ufft_av1).^2) + sum(((L2-Ufft_av2)./Ufft_av2).^2)); % relative l2 
        end
        [~,tstarind] = min(errs);
        tstar = -xx(tstarind);
        corners{d} = [tstar max(NN-tstarind+1,1)];
    end
    Ufft = abs(fftshift(fft(permute(U_obs,shift))));
    sig_est = sqrt(mean(reshape(Ufft([1:tstarind-3 2*NN-tstarind+2:end],:),[],1).^2)/L);
end
