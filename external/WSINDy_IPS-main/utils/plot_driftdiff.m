%% plot drift and diffusion

if toggle_plot_drift
    if exist('sigf','var')
        figure;     
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
        figure;     
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
