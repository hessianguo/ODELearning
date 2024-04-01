figind=0;

%% plot basis fcn

if toggle_plot_basis_fcn
    pdx=[1 3];
    figind = figind +1;
    figure(figind); 
    [x1,t1]=meshgrid(xs_obs{1}(1:2*ms(1,1)+1),xs_obs{end}(1:2*ms(1,2)+1));
    mesh(x1,t1,Cfs_t(pdx(1),:)'*Cfs_x(pdx(2),:))
    xlabel('x'); ylabel('t')
    title('\partial^\alpha \psi')
    drawnow
end

%% plot data

if toggle_plot_sol~=0
    figind = figind +1;
    figure(figind); 
    colormap(turbo(50))
    if dim==2
        surf(xs_obs{1},xs_obs{2},U_obs{min(end,toggle_plot_sol)}', 'EdgeColor','none')
        view([0 90])       
        zlabel('$u$','interpreter','latex','fontsize',14)
        set(gca, 'TickLabelInterpreter','latex','fontsize',14)
        xlabel('$x$','interpreter','latex','fontsize',14)
        ylabel('$t$','interpreter','latex','fontsize',14)
        legend('$U(x,t)$','interpreter','latex','fontsize',14)
        xlim([xs_obs{1}(1) xs_obs{1}(end)])
        ylim([xs_obs{2}(1) xs_obs{2}(end)])
        colorbar
    elseif dim==3
        plotgap=1;
        for j=1:plotgap:floor(length(xs_obs{3}))
            for i=1:length(toggle_plot_sol)
                subplot(length(toggle_plot_sol),1,i)
                surf(1:length(xs_obs{1}),1:length(xs_obs{2}),squeeze(U_obs{min(end,toggle_plot_sol(i))}(:,:,j))', 'EdgeColor','none')
                xlabel('$x$','interpreter','latex','fontsize',14)
                ylabel('$y$','interpreter','latex','fontsize',14)            
                view([0 90])
                legend('$U(x,t)$','interpreter','latex','fontsize',14)
                title(num2str(xs_obs{3}(j)))
                colorbar
            end
            drawnow
        end
    end
end

%% plot loss fcn (MSTLS)

if size(lossvals,2)>1 && toggle_plot_loss
    figind = figind +1;
    figure(figind); 
    loglog(lossvals(2,:),lossvals(1,:),'o-')
    xlabel('$\lambda$','interpreter','latex','fontsize',14)
    ylabel('$\mathcal{L}$','interpreter','latex','fontsize',14)
    set(gca, 'TickLabelInterpreter','latex','fontsize',14)
    xtick =10.^linspace(log10(lossvals(2,1)),log10(lossvals(2,end)),4);
    xticks(xtick)
    xticklb = num2str(log10(xtick)');
    xticklabels(strcat('$10^{',xticklb(:,1:min(4,end)),'}$'))
    legend('MSTLS Loss')
end

%% plot drift and diffusion

if toggle_plot_drift
    if exist('sigf','var')
        figind = figind +1;
        figure(figind);     
        if dim==2
            plot(xs_obs{1},sigf(xs_obs{1}',0),'r')
            legend({'$\hat{\sigma}\hat{\sigma}^T$'},'interpreter','latex','fontsize',14)
            if exist('sigx_true','var')
                if ~isempty(sigx_true)
                    hold on
                    plot(xs_obs{1},sigx_true(xs_obs{1}),'b--')
                    hold off
                    title(['diffusivity rel. err.=', num2str(norm(sigf(xs_obs{1},0)-sigx_true(xs_obs{1}))/norm(sigx_true(xs_obs{1})))])
                    legend({'$\hat{\sigma}\hat{\sigma}^T$','$\sigma \sigma^T$'},'interpreter','latex','fontsize',14)
                end
            elseif exist('nu','var')
                fprintf('\nnu rel. err. %0.4f',abs(nu_learned-nu)/nu)
            end
        elseif dim==3
            [xx,yy] = meshgrid(xs_obs{1},xs_obs{2});
            surf(xs_obs{1},xs_obs{2},sigf(xx,yy,0), 'EdgeColor','none')
            view([0 90])       
            colorbar
            title('\sigma(x)')
        end
    end
    if exist('driftf','var')
        figind = figind +1;
        figure(figind);     
        if dim==2
            plot(xs_obs{1},driftf(xs_obs{1}',0),'r')
            title('V')
        elseif dim==3            
            Vlearned = @(x,y,t) 0*x+0*y+0*t;
            for ll=1:length(drift)
                if isequal(drift{ll}{2},'{div}')
                    Vlearned = @(x,y,t) Vlearned(x,y,t)+ drift{ll}{1}(x,y,t);
                elseif isequal(drift{ll}{2},'{x}')
                    Vlearned = @(x,y,t) Vlearned(x,y,t)+ drift{ll}{1}(x,y,t).*x;
                elseif isequal(drift{ll}{2},'{y}')
                    Vlearned = @(x,y,t) Vlearned(x,y,t)+ drift{ll}{1}(x,y,t).*y;
                end
            end
            Vlearnedplot = Vlearned(xx,yy,1);
            surf(xs_obs{1},xs_obs{2},Vlearnedplot, 'EdgeColor','none')
            view([0 90])       
            colorbar
            title('V')
        end
    end
end

%% plot interaction function (Histograms)

if toggle_plot_IPforce
    figind = figind +1;
    figure(figind); 
    xR = max(cellfun(@(x) range(x),xs_obs(1:end-1)));
    x = linspace(-xR,xR,2*length(xs_obs{1}));
    if dim -1 ==  1
        plot(x,-f_learnedcell{1}(x,0),'linewidth',2)
        legend('learned \nabla{K}')
        if exist('f_true','var')
            hold on
            plot(x,f_true(abs(x)).*x./max(abs(x),eps),'linewidth',2)
            legend('learned  \nabla{K}','true  \nabla{K}')
            title(['interaction force rel. err.=',num2str(norm(-f_learnedcell{1}(x,0)-f_true(abs(x)).*x./max(abs(x),eps),2)/norm(f_true(abs(x)).*x./max(abs(x),eps),2))])
        end
    elseif dim -1 ==2
        if ~exist('f_true','var')
            f_true=[];
        end
        [xx,yy] = meshgrid(x);
        subplot(2,1+~isempty(f_true),1)
            flearned_mat = -f_learnedcell{1}(xx,yy,0).*xx./max(hypot(xx,yy),eps)-f_learnedcell{2}(xx,yy,0).*yy./max(hypot(xx,yy),eps);
            imagesc(flearned_mat)
            colorbar
            axis equal
            title('\nabla{K} learned')
        subplot(2,1+~isempty(f_true),2+~isempty(f_true):2+2*(~isempty(f_true)))
            plot(x,flearned_mat(ceil(end/2),:),'r')
            xlim([0 max(x)])
        if ~isempty(f_true)
            subplot(2,2,2)
                ftrue_mat = f_true(hypot(xx,yy));
                imagesc(ftrue_mat) 
                colorbar
                axis equal
                title('\nabla{K} true')
            subplot(2,2,3:4)
                hold on
                plot(x,ftrue_mat(ceil(end/2),:),'b')
                xlim([0 max(x)])
                hold off
                legend({'\nabla{K} learned','\nabla{K} true'})
                title(['\nabla{K} err=',num2str(norm(flearned_mat(:)-ftrue_mat(:),2)/max(norm(ftrue_mat(:),2),eps))])
        end
    end
end

%% plot data fft

if ~isempty(toggle_plot_fft)
    figind = figind +1;
    figure(figind); 
    dd=toggle_plot_fft{1}(1);
    Ufft = abs(fft(permute(U_obs{toggle_plot_fft{2}},toggle_plot_fft{1})));
    Ufft = reshape(Ufft,size(Ufft,1),[]);
    Ufft = mean(Ufft(floor(end/2):end,:),2);
    L = length(Ufft)-1;
    ks = -L:L;
    Ufft = [Ufft; flipud(Ufft(2:end))];
    semilogy(ks,Ufft)
    hold on
    if dd<dim
        Cfs_ffts = fft([zeros(1,length(xs_obs{dd})-2*ms(toggle_plot_fft{3},1)-1) Cfs{toggle_plot_fft{3}}{1}(1,:)]);
    else
        Cfs_ffts = fft([zeros(1,length(xs_obs{dd})-2*ms(toggle_plot_fft{3},2)-1) Cfs{toggle_plot_fft{3}}{2}(1,:)]);
    end
    Cfs_ffts=abs(Cfs_ffts(floor(end/2):end));
    Cfs_ffts=[Cfs_ffts fliplr(Cfs_ffts(2:end))];
    Cfs_ffts=Cfs_ffts*max(Ufft)/max(Cfs_ffts);
    semilogy(ks,Cfs_ffts)
    if exist('corners','var')
        k = corners{dd}(2);
        semilogy([-k k],Ufft(L+1-k)*[1 1],'o','markersize',12)
    end    
   hold off     
   ylim([min(Ufft)*0.1 max(Ufft)])
   legend({'$\mathcal{F}(U_d)$','$\mathcal{F}(\phi_d)$','$k_d^*$'},'interpreter','latex','fontsize',14)
end