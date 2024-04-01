function [U_obs,xs_obs,n,dims,dim,snr,sigma,noise,true_nz_weights,lhs,dx,dt,Ntot] = load_pde_data_fcn(varargin)

defaultUexact = [];
defaultxs = [];
defaultlhs = [];
defaulttrue_nz_weights = {};
defaultpde_num = Inf;
defaultsigma_NR = 0;
defaultnoise_dist = 0;
defaultnoise_alg = 0;
defaultXscell = [];
default_t = [];
defaulttoggle_2ndorder = 0;
defaultbw = 0;
defaultnumx = 128;
defaultexps = 1;
defaultsubsampN = 1;
defaultnumsdv = 4;
defaultcoarsen_data = [];
defaultavg_data = [];
defaultcustdom = [];
defaultXsnz = [0 1];
defaultShift = [];

inp = inputParser;
addParameter(inp,'pde_num',defaultpde_num);
addParameter(inp,'sigma_NR',defaultsigma_NR);
addParameter(inp,'noise_dist',defaultnoise_dist);
addParameter(inp,'noise_alg',defaultnoise_alg);
addParameter(inp,'Xscell',defaultXscell);
addParameter(inp,'U_exact',defaultUexact);
addParameter(inp,'xs',defaultxs);
addParameter(inp,'lhs',defaultlhs);
addParameter(inp,'t',default_t);
addParameter(inp,'true_nz_weights',defaulttrue_nz_weights);
addParameter(inp,'toggle_2ndorder',defaulttoggle_2ndorder);
addParameter(inp,'bw',defaultbw);
addParameter(inp,'numx',defaultnumx);
addParameter(inp,'custdom',defaultcustdom);
addParameter(inp,'exps',defaultexps);
addParameter(inp,'subsampN',defaultsubsampN);
addParameter(inp,'numsdv',defaultnumsdv);
addParameter(inp,'coarsen_data',defaultcoarsen_data);
addParameter(inp,'avg_data',defaultavg_data);
addParameter(inp,'Xsnz',defaultXsnz);
addParameter(inp,'Shift',defaultShift);
parse(inp,varargin{:});  

pde_num = inp.Results.pde_num;
sigma_NR = inp.Results.sigma_NR;
noise_dist = inp.Results.noise_dist;
noise_alg = inp.Results.noise_alg;
Xscell = inp.Results.Xscell;
U_exact = inp.Results.U_exact;
xs = inp.Results.xs;
lhs = inp.Results.lhs;
true_nz_weights = inp.Results.true_nz_weights;
t = inp.Results.t;
toggle_2ndorder = inp.Results.toggle_2ndorder;
bw = inp.Results.bw;
numx = inp.Results.numx;
exps = inp.Results.exps;
subsampN = inp.Results.subsampN;
numsdv = inp.Results.numsdv;
coarsen_data = inp.Results.coarsen_data;
avg_data = inp.Results.avg_data;
custdom = inp.Results.custdom;
Xsnz = inp.Results.Xsnz;
Shift = inp.Results.Shift;

%% load data: U_obs, xs_obs, true_nz_weights, lhs

pde_names = {'burgers.mat','KdV.mat','KS.mat','NLS.mat',...
    'Sine_Gordon.mat','full_or_old/rxn_diff_old.mat','Nav_Stokes.mat','porous.mat',...
    'AC_temp','qanr1Dcont'};

if isempty(U_exact)
    if and(pde_num<=length(pde_names),pde_num>0)
        load(['~/Desktop/data/WSINDy_PDE/datasets/',pde_names{pde_num}],'U_exact','xs','lhs','true_nz_weights')
    elseif and(exist('Xscell','var'),pde_num==inf)
        if ~toggle_2ndorder
            [U_exact,xs,lhs,~,dx,dt,Ntot] = hist1storder(Xscell,t,'numx',numx,'exps',exps,'bw',bw,'subsamp',subsampN,'numsdv',numsdv,'custdom',custdom,'noise',Xsnz);
        else
            if ~exist('Vscell','var')
                Vscell = [];
            end
            [U_exact,xs,lhs,~,dx,dt] = hist2ndorder(Xscell,t,'numx',numx,'exps',exps,'bw',bw,'subsamp',subsampN,'numsdv',numsdv,'Vscell',Vscell);
        end 
    end
end

U_obs = U_exact;
xs_obs = xs;

if isempty(lhs)
    lhs= [1 zeros(1,length(xs_obs)-1) 1];
end

%% coarsen data: rewrite coarsened versions to U_obs, xs_obs

n = length(U_obs);
dims = size(U_obs{1});
dim = length(dims);

if ~isempty(coarsen_data)
    dim = length(size(U_obs{1}));
    inds = cell(1,dim);
    for j=1:dim
        inds{j} = 1+floor(coarsen_data(j,1)*dims(j)):coarsen_data(j,2):ceil(coarsen_data(j,3)*dims(j));
        xs_obs{j} = xs{j}(inds{j});
    end
    for j=1:n
        U_obs{j} = U_obs{j}(inds{:});
    end
end

for j=1:min(length(Shift),dim)
    xs_obs{j} = xs_obs{j}+Shift(j);
end

%% add noise

rng_seed = rng().Seed; 
rng(rng_seed);
[U_obs,noise,snr,sigma] = gen_noise(U_obs,sigma_NR,noise_dist,noise_alg,rng_seed,0);

%% smooth data
 
% U_obs{1} = wdenoise2(U_obs{1});%,'DenoisingMethod','UniversalThreshold','ThresholdRule','hard');
% avg_data = zeros(length(size(U_obs{1})),1); %[3 3 0];
if ~isempty(avg_data)
    [U_obs,xs_obs] = conv_movavg(U_obs,xs_obs,avg_data);
end

dims = size(U_obs{1});
dx = xs_obs{1}(2)-xs_obs{1}(1); dt = xs_obs{end}(2)-xs_obs{end}(1);

end