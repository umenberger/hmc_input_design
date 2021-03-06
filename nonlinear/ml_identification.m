
%%

em_model = model;

% Dimensions
em_model.nx = 1;
em_model.ny = 1;
em_model.nw = 1;
em_model.nv = 1;
em_model.nu = 1;

% note: parameters are variances
em_model.processnoise.sample = @(x,t,theta) sqrt(theta(6))*randn(size(x));
em_model.processnoise.eval = @(x,t,theta) -.5*x.^2/theta(6);
em_model.measurementnoise.eval = @(x,y,t,theta) -.5*(theta(3)*x + theta(4)*x.^2 + theta(5)*x.^3 - y).^2/theta(7);
em_model.measurementnoise.sample = @(x,t,theta) sqrt(theta(7))*randn(size(x));

em_model.initialcondition.sample = @(x,theta) sqrt(theta(8))*randn(size(x));

em_model.theta = theta;

%%

num_realizations = 10;

nths = 100; % 600

ths = linspace(-1.5,1.0,nths);

num_particles = 5e3; % 10e3

data_psts_nom = zeros(num_realizations,nths,num_trials);
data_psts_sgn = zeros(num_realizations,nths,num_trials);
data_psts_lo = zeros(num_realizations,nths,num_trials);

plot_posts = 0;

plot_em = 0;

if plot_posts
    rpf = figure;
    plot(theta(param_index),0,'ko')
    hold on
end

em_estimates = zeros(2,num_realizations,num_trials);

for ti = 1:num_trials
    
    for ri = 1:num_realizations    
       
% generate appropriate disturbances, to be used for all tests  
    w = 1*mvnrnd(0,1*theta(6),T)';
    v = 1*mvnrnd(0,1*theta(7),T)';
    
   
    
%% EM for sgn    
% -------------------------------------------------------------------------
%     fprintf('\nSign:\n')
    
%     u_tmp = us_nom(:,:,ti);
    u_tmp = u_sgn;
    
    [y,x] = sim_flex_nl(model,theta,x1,u_tmp,w,v);  
    
%     stop
    if plot_posts
    if (~isnan(sum(y))) && (max(abs(y)) < 1e3) 
    
    n = length(ths);
    ls = zeros(1,n);
    lls = zeros(1,n);

    fprintf('Running particle filter...')
    clear ops_pf
    ops_pf.num_particles = num_particles;
    ops_pf.model = model;
    ops_pf.u = u_tmp;
    ops_pf.y = y;

    theta_tmp = theta;

    for i = 1:n

        theta_tmp(param_index) = ths(i);
        ops_pf.theta = theta_tmp; 

        [particles,weights,xf,loglik,ancestors] = pf_flex_nl(ops_pf);

        lls(i) = (loglik);
        ls(i) = exp(loglik);

    end
    fprintf('done\n')
    sl = trapz(ths,ls);
    ls_tmp = ls/sl; 
    
    else
        ls_tmp = nan(1,nths);
        fprintf('unstable\n')
    end
    
    data_psts_sgn(ri,:,ti) = ls_tmp;    
    
    end
    
% run em
    clear ops_em
    ops_em.em_model = em_model;
    ops_em.u = u_tmp;
    ops_em.y = y;
    ops_em.parameter = 0*theta(1);
    
    res_em_sgn = quick_em(ops_em);
    
    
%% posteriors for lo    
% -------------------------------------------------------------------------

%     fprintf('\nLocal:\n')
    
%     u_tmp = us_nom(:,:,ti);
    u_tmp = 1*u_lo;
    
    [y,x] = sim_flex_nl(model,theta,x1,u_tmp,w,v);  
    
    if plot_posts
    if (~isnan(sum(y))) && (max(abs(y)) < 1e3)
    
    n = length(ths);
    ls = zeros(1,n);
    lls = zeros(1,n);

    fprintf('Running particle filter...')
    clear ops_pf
    ops_pf.num_particles = num_particles;
    ops_pf.model = model;
    ops_pf.u = u_tmp;
    ops_pf.y = y;

    theta_tmp = theta;

    for i = 1:n

        theta_tmp(param_index) = ths(i);
        ops_pf.theta = theta_tmp; 

        [particles,weights,xf,loglik,ancestors] = pf_flex_nl(ops_pf);

        lls(i) = (loglik);
        ls(i) = exp(loglik);

    end
    fprintf('done\n')
    sl = trapz(ths,ls);
    ls_tmp = ls/sl; 
    
    else
        ls_tmp = nan(1,nths);
        fprintf('unstable\n')
    end    
    
    data_psts_lo(ri,:,ti) = ls_tmp;     
    end
    
% run em
    clear ops_em
    ops_em.em_model = em_model;
    ops_em.u = u_tmp;
    ops_em.y = y;
    ops_em.parameter = 0*theta(1);
    
    res_em_lo = quick_em(ops_em);    
    
    
%% figure 
% -------------------------------------------------------------------------    

    if plot_posts

    lw = 1.5;
    figure(rpf)
%     plot(ths,data_psts_nom(ri,:,ti),'linewidth',lw)
    plot(ths,data_psts_sgn(ri,:,ti),'linewidth',lw)
    plot(ths,data_psts_lo(ri,:,ti),'linewidth',lw)

    if hessian_opt  
        plot(ths,ls_hop,'linewidth',lw)
        legend('true','nominal','sign of nominal','locally optimized','hessian optimization')
    else
        legend('true','sign of nominal','locally optimized')
    end

    grid on
    xlabel('parameter')
    title('posteriors from pf')    
    
    end
    
%%

    if plot_em
    figure
    plot(1:length(res_em_sgn.params),res_em_sgn.params)
    hold on
    plot(1:length(res_em_lo.params),res_em_lo.params)
    plot([1 length(res_em_lo.params)],theta(1)*[1 1])
    end
    
%%
% average the last...20 iterates
    em_estimates(1,ri,ti) = mean(res_em_sgn.params(end-19:end));
    em_estimates(2,ri,ti) = mean(res_em_lo.params(end-19:end));

    end

end


%% bar chart

e_dp = em_estimates(1,:,1) - theta(param_index);
nobins = 20;

figure
hg_dp = histogram(e_dp,nobins);
be_dp = hg_dp.BinEdges;
v_dp = hg_dp.Values;
tmp = diff(be_dp);
d_dp = tmp(1)/2;

e_hmc = em_estimates(2,:,1) - theta(param_index);

hg_hmc = histogram(e_hmc, be_dp);
be_hmc = hg_hmc.BinEdges;
v_hmc = hg_hmc.Values;

lw = 3;
trnsp = 0.5;
fs = 14;

figure
subplot(1,2,1)
plot(nan,nan,'s','color',col_dp,'MarkerFaceColor',col_dp)
hold on
plot(nan,nan,'s','color',col_hmc,'MarkerFaceColor',[col_hmc])

plot([be_dp(1:end-1);be_dp(1:end-1)],[zeros(nobins,1)';v_dp],'color',col_dp,'linewidth',lw)
hold on
plot([be_dp(1:end-1)+d_dp;be_dp(1:end-1)+d_dp],[zeros(nobins,1)';v_hmc],'color',[col_hmc,trnsp],'linewidth',lw)
grid on

tmp_leg = {'pseudo-random','optimized'};

xlabel('parameter estimate error','interpreter','latex','fontsize',fs)
ylabel('count','interpreter','latex','fontsize',fs)
% grid on
% % title('T_2 relaxation posteriors')   
legend(tmp_leg,'interpreter','latex','fontsize',fs)
% set(gca,'fontsize',fs)
% set(gca,'TickLabelInterpreter','latex')
legend boxoff







































