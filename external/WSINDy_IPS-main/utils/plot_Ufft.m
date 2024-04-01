if toggle_plot_fft>0
    figure; 
    coords=1:dim;
    for j=1:dim-1
        coords=[coords;circshift(coords(end,:),-1)];
    end
    for j=1:dim
        coordsj = coords(j,:);
        dd=coordsj(1);
        Ufft = abs(fft(permute(U_obs{1},coordsj)));
        Ufft = reshape(Ufft,size(Ufft,1),[]);
        Ufft = mean(Ufft(floor(end/2):end,:),2);
        L = length(Ufft)-1;
        ks = -L:L;
        Ufft = [Ufft; flipud(Ufft(1:end-1))]/max(Ufft);
        subplot(dim,1,j)
            semilogy(ks,Ufft)
            hold on
            if dd<dim
                Cfs_ffts = fft([zeros(1,length(xs_obs{dd})-2*ms(1,1)-1) Cfs{1}{1}(1,:)]);
            else
                Cfs_ffts = fft([zeros(1,length(xs_obs{dd})-2*ms(1,2)-1) Cfs{1}{2}(1,:)]);
            end
            Cfs_ffts=abs(Cfs_ffts(floor(end/2):end));
            Cfs_ffts=[Cfs_ffts fliplr(Cfs_ffts(1:end-1))];
            Cfs_ffts=Cfs_ffts/max(Cfs_ffts);
            semilogy(ks,Cfs_ffts)
            if exist('corners','var')
                k = corners{dd}(2);
                semilogy([-k k],Ufft(L+1-k)*[1 1],'o','markersize',12)
            end    
            hold off     
            ylim([min(Ufft)*0.1 max(Ufft)])
            legend({'$\mathcal{F}(U_d)$','$\mathcal{F}(\phi_d)$','$k_d^*$'},'interpreter','latex','fontsize',14)
            title(['coord',num2str(dd)])
    end
    xlabel('k')
end