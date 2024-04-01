if plotgap>0
    figure;
    colormap(turbo(50))
    if dim==2
        for j=1:plotgap:floor(length(xs_obs{end}))
            plot(xs_obs{1},U_obs{1}(:,j)')
            set(gca, 'TickLabelInterpreter','latex','fontsize',14)
            xlabel('$x$','interpreter','latex','fontsize',14)
            ylabel('$U$','interpreter','latex','fontsize',14)
            xlim([xs_obs{1}(1) xs_obs{1}(end)])
            ylim([0 max(U_obs{1}(:))])
            legend('$U(x,t)$','interpreter','latex','fontsize',14)
            drawnow
        end
        surf(xs_obs{1},xs_obs{2},U_obs{1}', 'EdgeColor','none')
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
        for j=1:plotgap:floor(length(xs_obs{end}))
            surf(1:length(xs_obs{1}),1:length(xs_obs{2}),squeeze(U_obs{1}(:,:,j))', 'EdgeColor','none')
            xlabel('$x$','interpreter','latex','fontsize',14)
            ylabel('$y$','interpreter','latex','fontsize',14)            
            view([0 90])
            legend('$U(x,t)$','interpreter','latex','fontsize',14)
            title(num2str(xs_obs{3}(j)))
            colorbar
            drawnow
        end
    end
end