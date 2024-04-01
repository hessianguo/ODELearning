%% plot interaction function (Histograms)

if toggle_plot_IPforce
    figure; 
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

