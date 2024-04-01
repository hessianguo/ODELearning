%% plot loss fcn (MSTLS)

if size(lossvals,2)>1 && toggle_plot_loss
    figure; 
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
