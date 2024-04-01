if any(exps>length(Xscell))
    disp('exps num does not exist,using available exps') 
    exps = unique(min(exps,length(Xscell)));
end
if ~exist('true_nz_weight_tags','var')
    true_nz_weight_tags = [];
end

Xscell(exps)=cellfun(@(x) x(randperm(end,NN),:,:),Xscell(exps),'uni',false);
vars={'Xscell',Xscell,'t',t,'numx',numx,'exps',exps,...
    'numsdv',numsdv,'coarsen_data',coarsen_data,'custdom',custdom,'Xsnz',Xsnz,...
    'Shift',scoord,'true_nz_weights',true_nz_weight_tags};
tic;
rng('shuffle');
vartot = {'U_exact','xs','lhs','true_nz_weights','pde_num','sigma_NR','noise_dist','noise_alg','Xscell','t',...
    'bw','numx','exps','subsampN','numsdv','coarsen_data','toggle_2ndorder','custdom','Xsnz','Shift'};
varchar = {};
for j=1:length(vars)
    if ischar(vars{j})
        varchar{end+1} = vars{j};
    end
end
varfoo=find(~ismember(vartot,varchar));
for j=1:length(varfoo)
    assignin('base',vartot{varfoo(j)},[]);
end
[U_obs,xs_obs,n,dims,dim,snr,sigma,noise,true_nz_weights,lhs,dx,dt,Ntot] = load_pde_data_fcn(vars{:});
ET_load_data = toc;
fprintf(1,'ET_load_data = %4.4f \n',ET_load_data);
