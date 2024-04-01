%% particle data - Xscell, t

exps = 1:4;
subsampN = 1;

numx = 100 + 1;
numsdv = 4;
custdom = [];
coarsen_data = [[0 1 1];[0 1 1];[0 1 0.5]];
Xsnz = [0 1];
scoord = 0;

NN = size(Xscell{1},1);
dim = size(Xscell{1},2)+1;

%% Set Library

% local autonomous operators
max_dx = 0;
max_dt = 0;
polys = [];
trigs = [];
use_all_dt = 0;
use_cross_dx = 0;
custom_add = [];
custom_remove = {@(mat,lhs) find(all([mat(:,2)==0 mat(:,3)==0 ~ismember(mat,lhs,'rows')],2))};
toggle_comb = 1;

% local non-autonomous operators
driftpolys=[]; drifttrigs=[0:5];
diffpolys=[]; difftrigs=[0:5]; crossdrift=0;

% non-local operators
convargs = {{'dimx',dim-1,'utagin',1,'utagout',1,'psitags',1,'Mon',[0:5],'Sing',[-1:0.5:0],'Exp',[],'Singeps',[0.01],'svdtol',[1e-4]}};

%% Set Discretization

phi_class = {1,1};
sm_x = 6;
sm_t = 6;

%%% uncomment to set test fcn params manually
p_x = 5; mxs = 24;
tau_x = []; k_x = []; tauhat_x = []; 
p_t = 7; mts = 12;
tau_t = []; k_t = []; tauhat_t = [];

% %%% set test fcn params using cornerpoint
% tauhat_x = 0.5; tau_x = 10^-6;
% p_x = []; mxs = 1;
% tauhat_t = 0.3; tau_t = 10^-3;
% p_t = []; mts = 1;

%%% rescale coordinates and/or use approx. variance for improved conditioning
scales = 2;
covtol = 0;

%%% trim rows with low particle density
trim_tags = [1 zeros(1,dim-1) 0]; % column corresponding to particle density
trim_fcn = {@(col) max(col/max(abs(col)),eps)}; % map density values to (0,1]
inds_keep_fcn = @(col) log10(col)>-2; % keep rows that have at least 1% of max density

%% MSTLSQP

lambda = 10.^(linspace(-4, 0, 100));
gamma_tol = Inf;
alpha = 0.5;
maxits = Inf;
sparsity_scale = 0;
excl_tags={'u^{1}_{lap}','u^{1}_{xx}','([x.^0.*t.^0]u^1)_{xx}','([x.^0.*y.^0.*t.^0]u^1)_{lap}'};
excl_tols=0;
tol = 10^-8; 
maxQPits = 100; 
dispQP = 'off';
meth = 'STLSQP';

%% run alg

wsindy_IPS_script;

%% Display 
% close all;

print_loc = 1;
toggle_plot_basis_fcn = 0;
toggle_plot_sol = 0;
toggle_plot_loss = 1;
toggle_plot_drift = 0;
toggle_plot_IPforce = 1;
toggle_plot_fft = {[3 1 2],1,1};

get_results;
wsindy_IPS_display;