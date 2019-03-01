%%

clear variables
close all
clc

u_min = -1;
u_max = 1;

num_trials = 1;

num_resamples = 4;

num_gd_steps = 50;

hessian_opt = 0;
num_hop_steps = 1e4;

costs_hop = zeros(1,num_hop_steps,num_trials);



%% nominal input

clear theta

% -------------------------------------------------------------------------

% theta(1) = -0.5; theta(2) = 1; theta(3) = 1; theta(4) = 1; theta(5) = 0.1;
% theta(6) = 0.1; theta(7) = 0.1; theta(8) = 0.01;
% 
% % Note: (8/2/19) - we observe a lot of instability with th(6) = 0.1;
% % theta(6) = 
% 
% f = @(x,u,th) th(1)*(x.*(x-1).*(x+1)) + th(2)*u; 
% g = @(x,u,th) th(3)*x + th(4)*x.^2 + th(5)*x.^3;
% 
% clear model
% model.f = f;
% model.g = g;
% 
% model.dfdx = @(x,u,th) th(1)*(3*x.^2-1);
% model.dgdx = @(x,u,th) th(3) + 2*th(4)*x + 3*th(5)*x.^2;
% model.dfdth = @(x,u,th) (x.*(x-1).*(x+1));

% -------------------------------------------------------------------------

% theta(1) = -0.5; theta(2) = 1; theta(3) = 1; theta(4) = 1; theta(5) = 0.1;
% theta(6) = 0.1; theta(7) = 0.1; theta(8) = 0.01;
% 
% % Note: (8/2/19) - we observe a lot of instability with th(6) = 0.1;
% % theta(6) = 
% 
% f = @(x,u,th) th(1)*(x.*(x-1).*(x+1))./(1+x.^2) + th(2)*u; 
% g = @(x,u,th) th(3)*x + th(4)*x.^2 + th(5)*x.^3;
% 
% clear model
% model.f = f;
% model.g = g;
% 
% model.dfdx = @(x,u,th) th(1)*((1+x.^2).*(3*x.^2-1) - (x.^3-x).*(1+x.^2))./((1+x.^2).^2);
% model.dgdx = @(x,u,th) th(3) + 2*th(4)*x + 3*th(5)*x.^2;
% model.dfdth = @(x,u,th)((1+x.^2).*(3*x.^2-1) - (x.^3-x).*(1+x.^2))./((1+x.^2).^2);

% -------------------------------------------------------------------------


param_index = 1;

nx = 1;
ny = 1;
nu = 1;

T = 30;
x1 = 0;

model.x1 = x1;
model.nx = nx;
model.nu = nu;
model.ny = ny;

%% posteriors on the fly

post_on_fly = 0;

ths_on_fly = linspace(-2,1,500);

posteriors_on_fly = zeros(1,length(ths_on_fly),num_resamples);

if post_on_fly
    fig_fly = figure;
    plot(theta(param_index),0,'ko')
    hold on
    xlabel('parameter')
    grid on
    leg_fly = {'true'};
    title('posteriors on the fly')    
end


%%

input_storage_intervals = [1:10,20:20:num_resamples];

us_inc_store = zeros(nu,T,length(input_storage_intervals));

inc_st_index = 1;


%%

us_nom = zeros(nu,T,num_trials);
us_sgn = zeros(nu,T,num_trials);
us_lo = zeros(nu,T,num_trials);

clear pdfs_nom pdfs_sgn pdfs_lo

norms_us_lo = zeros(num_resamples,num_trials);



%%


for trial_index = 1:num_trials


u = 0.1*mvnrnd(0,1,T)';

% u = sign(u);
u(u>u_max) = u_max;
u(u<u_min) = u_min;

w = 0*mvnrnd(0,theta(6),T)';
v = 0*mvnrnd(0,theta(7),T)';

[y,x] = sim_flex_nl(model,theta,x1,u,w,v);

figure
subplot(3,1,1)
plot(1:T,y)
title('output')

subplot(3,1,2)
plot(1:T,x)
title('state')

subplot(3,1,3)
plot(1:T,u)
title('input')


%%

clear ops_hmc

ops_hmc.theta = theta;
ops_hmc.param_index = param_index;
ops_hmc.iters = 1e3;
ops_hmc.inertia = 1*eye(T*nx+1);
% ops_hmc.stepsize = 0.5e-2;
% ops_hmc.simlength = 100;
ops_hmc.stepsize = 0.5e-2;
ops_hmc.simlength = 50;
ops_hmc.model = model;

% for stable model
% ops_hmc.stepsize = 0.1e-2;
% ops_hmc.simlength = 75;

num_samples = 40;


%% quick check

% ops = ops_hmc;
% 
% % th_tmp = theta;
% % th_tmp(param_index) = -0.45;
% 
% ops.u = u;
% ops.theta = theta;
% 
% res = likelihood_hessian(ops)
% 
% stop

%%

figure(100)
plot(theta(param_index),0,'ko')
hold on
xlabel('parameter')
grid on
leg_hmc = {'true'};
title('posteriors (via hmc)')

% figure(200)
% plot(1:T,u)
% hold on
% xlabel('time')
% grid on
% leg_inputs = {'nominal'};
% title('inputs')


%% another input

u_sgn = u_max*sign(u);

[y_sgn,x_sgn] = sim_flex_nl(model,theta,x1,u_sgn,w,v);

% hmc
q_init = [x_sgn,theta(param_index)]';

ops = ops_hmc;

ops.y = y_sgn;
ops.u = u_sgn;
ops.q_init = q_init;

fprintf('Running HMC...')
t_hmc_s = tic;
res_hmc = hmc_flex_nl(ops);
fprintf('done. %.2f sec\n',toc(t_hmc_s))

fprintf('acceptance rate: %.2f\n',100*sum(res_hmc.acpt)/ops.iters)  

% 
noBurnInIterations = 500;
noBins = 50;

ops = [];
ops.markovChain = res_hmc.q(T*nx+1,1:1:end);
ops.noBurnInIterations = noBurnInIterations;
ops.noBins = noBins;
ops.normalization = 'pdf';
ops.title = ['param ' num2str(1)] ;
ops.subplots = 4;

res_plot = process_mcmc(ops);
close(res_plot.fighandle);

lw = 1.5;

figure(100)
tmp = diff(res_plot.binEdges);
d = tmp(1)/2;
plot(res_plot.binEdges(1:end-1)+d,res_plot.binCount,'k','linewidth',lw)
leg_hmc{end+1} = ['sign'];
legend(leg_hmc)

ths_sgn = res_plot.binEdges(1:end-1)+d;
pdf_sgn = res_plot.binCount;

%% iteratively improve nominal input via hmc

x_tmp = x;
y_tmp = y;
u_tmp = u;

u_lo = u_tmp;

% x_tmp = x_sgn;
% y_tmp = y_sgn;
% u_tmp = u_sgn;

% num_resamples = 1;

%%

for lin_index = 1:num_resamples
    
    t_start_iter = tic;
    
%% run hmc

    q_init = [x_tmp,theta(param_index)]';
    
    ops = ops_hmc;

    ops.y = y_tmp;
    ops.u = u_tmp;
    ops.q_init = q_init;

    fprintf('Running HMC...')
    t_hmc_s = tic;
    res_hmc = hmc_flex_nl(ops);
    fprintf('done. %.2f sec\n',toc(t_hmc_s))

    fprintf('acceptance rate: %.2f\n',100*sum(res_hmc.acpt)/ops.iters)  
    
% 

    noBurnInIterations = 500;
    noBins = 50;

    ops = [];
    ops.markovChain = res_hmc.q(T*nx+1,1:1:end);
    ops.noBurnInIterations = noBurnInIterations;
    ops.noBins = noBins;
    ops.normalization = 'pdf';
    ops.title = ['param ' num2str(1)] ;
    ops.subplots = 4;

    res_plot = process_mcmc(ops);
    close(res_plot.fighandle);
    
    if lin_index == 1
        lw = 1.5;
    else
        lw = 0.8;
    end

%     figure(100)
%     tmp = diff(res_plot.binEdges);
%     d = tmp(1)/2;
%     plot(res_plot.binEdges(1:end-1)+d,res_plot.binCount,'linewidth',lw)
%     leg_hmc{end+1} = ['iteration ' num2str(lin_index-1)];
%     legend(leg_hmc)
        
%     if lin_index > 1
%         
%         figure(101)
% %         figure
%         
%         plot(theta(param_index),0,'ko')
%         hold on
%         plot(ths_sgn,pdf_sgn,'k')
%         plot(ths_tmp,pdf_tmp)
%         plot(res_plot.binEdges(1:end-1)+d,res_plot.binCount)
%         xlabel('parameter')
%         grid on
%         legend('true','sign',['iter ' num2str(lin_index-2)],['iter ' num2str(lin_index-1)]);
%         title('posteriors (via hmc)')  
%         hold off
%     
%     
%     end
    
    ths_tmp = res_plot.binEdges(1:end-1)+d;
    pdf_tmp = res_plot.binCount;    
    

%% local optimization

% samples
    ofst = 500;
    del = 10;

    q_samples = res_hmc.q(:,ofst:del:ofst+(num_samples-1)*del);
    p_samples = res_hmc.p(:,ofst:del:ofst+(num_samples-1)*del); 

    clear ops

    ops = ops_hmc;

    th_wrong = theta;
%     th_wrong(1) = -0.8;
%     ops.theta = th_wrong;

    ops.q_samples = q_samples;
    ops.p_samples = p_samples;

    % ops.y = ys; % fix output

    ops

    gdss = 10;
    
    u_gd = u_tmp;
    u_gd_old = u_gd;
    
    f_old = inf;

    for i = 1:num_gd_steps

    % compute gradient
        [f_new,df_new] = nmpc_hmc_flex_nl_multicost(u_gd,ops);
        
        if i == 1
            f_old = f_new;
            df_old = df_new;
        end
        
        fprintf('\tcost at start of iter %d: %.5e, f_old %.5e\n',i,f_new*size(q_samples,2),f_old*size(q_samples,2))
%        fprintf('\t\t u(1) = %.4f, u_old(1) = %.4f\n',u_gd(1),u_gd_old(1))
        
        if f_new < f_old
% then the previous stepsize was fine
    % update input
            u_gd_old = u_gd;
            f_old = f_new; 
            df_old = df_new;
        else
            if i > 1
            u_gd = u_gd_old;
            gdss = gdss/2;       
            fprintf('changing stepsize: %.5e\n',gdss)
            end
            
            if gdss < 1e-4
                break;
            end
            
        end
        
        u_gd = u_gd - gdss*df_old'; 
        
    % project onto constraints
        u_gd(u_gd>u_max) = u_max;
        u_gd(u_gd<u_min) = u_min;

    end

    u_tmp = u_gd;

    figure(200)
    plot(1:T,u)
    hold on
    plot(1:T,u_tmp)
    xlabel('time')
    grid on
    legend('nominal',['iteration ' num2str(lin_index)])
    title('inputs')
    hold off

    if lin_index > 1
    figure(201)
    plot(1:T,u_lo)
    hold on
    plot(1:T,u_tmp)
    legend(['iter ' num2str(lin_index-1)],['iter ' num2str(lin_index)])
    grid on
    title('inputs')
    hold off
    end
    
    
%     if lin_index == num_resamples
%     figure(202)
%     plot(1:T,u_lo)
%     hold on
%     plot(1:T,u_tmp)
%     legend(['iter ' num2str(lin_index-1)],['iter ' num2str(lin_index)])
%     grid on
%     title('inputs')
%     hold off
%     end    
    
    norms_us_lo(lin_index,trial_index) = norm(u_tmp - u_lo);


    % simualte
    [y_tmp,x_tmp] = sim_flex_nl(model,theta,x1,u_tmp,w,v);   
    
    u_lo = u_tmp;
    x_lo = x_tmp;
    y_lo = y_tmp;    
    
    
    fprintf('\niteration time: %.2f\n',toc(t_start_iter))
    
    
    if sum(lin_index == input_storage_intervals)
        
        us_inc_store(:,:,inc_st_index) = u_lo;
        
%         tmp.ths = ths_lo;
%         tmp.pdf = pdf_lo;
%         pdfs_lo{trial_index} = tmp;
        
        inc_st_index = inc_st_index + 1;
        
    end    
    
    
    pause(0.1)

%% posterior as we go

    

    if post_on_fly

        num_particles = 10e3;

        n = length(ths_on_fly);
        ls = zeros(1,n);
        lls = zeros(1,n);

        fprintf('Running particle filter...')
        clear ops_pf
        ops_pf.num_particles = num_particles;
        ops_pf.model = model;
        ops_pf.u = u_tmp;
        ops_pf.y = y_tmp;

        theta_tmp = theta;

        for i = 1:n

            theta_tmp(param_index) = ths_on_fly(i);
            ops_pf.theta = theta_tmp; 

            [particles,weights,xf,loglik,ancestors] = pf_flex_nl(ops_pf);

            lls(i) = (loglik);
            ls(i) = exp(loglik);

        end
        fprintf('done\n')
        sl = trapz(ths_on_fly,ls);
        posteriors_on_fly(:,:,lin_index) = ls/sl;

        figure(fig_fly)
        plot(ths_on_fly,posteriors_on_fly(:,:,lin_index))
        leg_fly{end+1} = ['iter ' num2str(lin_index)];
        legend(leg_fly)
       
    end
    
    
    
    
end


%% optimize approximate hessian with gradient descent
    
if hessian_opt

    clear ops

    ops = ops_hmc;

    th_wrong = theta;
    th_wrong(1) = -0.8;
    ops.theta = th_wrong;

    ops

    gdss = 0.1;
    
    u_gd = u;
    u_gd_old = u_gd;
    
    f_old = inf;

    i = 1;

    stop_opt = 0;

    while ~stop_opt

        ops.u = u_gd;

        res = likelihood_hessian(ops);
        f_new = res.hessian;
        df_new = res.d_hessian_fd;
        
        if i == 1
            f_old = f_new;
            df_old = df_new;
        end
        
        fprintf('\tcost at start of iter %d: %.5e, f_old %.5e\n',i,f_new,f_old)
%        fprintf('\t\t u(1) = %.4f, u_old(1) = %.4f\n',u_gd(1),u_gd_old(1))
        
        if f_new <= f_old
% then the previous stepsize was fine
    % update input
    
            if (i > 1) && (f_old - f_new < 1e-3)
                stop_opt = 1;
            end

            costs_hop(1,i,trial_index) = f_new;    
    
            u_gd_old = u_gd;
            f_old = f_new; 
            df_old = df_new;
            i = i + 1;
        else
            if i > 1
            u_gd = u_gd_old;
            gdss = gdss/2;       
            fprintf('changing stepsize: %.5e\n',gdss)
            end
            
%             if gdss < 1e-4
%                 break;
%             end
            
        end
        
        u_gd = u_gd - gdss*df_old'; 
        
    % project onto constraints
        u_gd(u_gd>u_max) = u_max;
        u_gd(u_gd<u_min) = u_min;
        
        
        if i == num_hop_steps
            stop_opt = 1;
        end        

    end

    u_hess = u_gd;

%     us_hop(:,:,trial_index) = u_hess;

    % simualte
    [y_hess,x_hess] = sim_flex_nl(model,theta,x1,u_hess,w,v);   

    u_hop = u_hess;
    x_hop = x_hess;
    y_hop = y_hess; 

end



%% posteriors from hmc

tmp_num_iters = 10e3;

ops = ops_hmc;

ops.iters = tmp_num_iters;
ops.y = y;
ops.u = u;
ops.q_init = [x,theta(param_index)]';

fprintf('Running HMC...')
t_hmc_s = tic;
res_hmc = hmc_flex_nl(ops);
fprintf('done. %.2f sec\n',toc(t_hmc_s))

fprintf('acceptance rate: %.2f\n',100*sum(res_hmc.acpt)/ops.iters)  

% 

noBurnInIterations = 500;
noBins = 50;

ops = [];
ops.markovChain = res_hmc.q(T*nx+1,1:1:end);
ops.noBurnInIterations = noBurnInIterations;
ops.noBins = noBins;
ops.normalization = 'pdf';
ops.title = ['nominal'] ;
ops.subplots = 4;

res_plot = process_mcmc(ops);
% close(res_plot.fighandle);

tmp = diff(res_plot.binEdges);
d = tmp(1)/2;

ths_nom = res_plot.binEdges(1:end-1)+d;
pdf_nom = res_plot.binCount;



ops = ops_hmc;

ops.iters = tmp_num_iters;
ops.y = y_sgn;
ops.u = u_sgn;
ops.q_init = [x_sgn,theta(param_index)]';

fprintf('Running HMC...')
t_hmc_s = tic;
res_hmc = hmc_flex_nl(ops);
fprintf('done. %.2f sec\n',toc(t_hmc_s))

fprintf('acceptance rate: %.2f\n',100*sum(res_hmc.acpt)/ops.iters)  

% 

noBurnInIterations = 500;
noBins = 50;

ops = [];
ops.markovChain = res_hmc.q(T*nx+1,1:1:end);
ops.noBurnInIterations = noBurnInIterations;
ops.noBins = noBins;
ops.normalization = 'pdf';
ops.title = ['sign'] ;
ops.subplots = 4;

res_plot = process_mcmc(ops);
% close(res_plot.fighandle);

tmp = diff(res_plot.binEdges);
d = tmp(1)/2;

ths_sgn = res_plot.binEdges(1:end-1)+d;
pdf_sgn = res_plot.binCount;



ops = ops_hmc;

ops.iters = tmp_num_iters;
ops.y = y_lo;
ops.u = u_lo;
ops.q_init = [x,theta(param_index)]';

fprintf('Running HMC...')
t_hmc_s = tic;
res_hmc = hmc_flex_nl(ops);
fprintf('done. %.2f sec\n',toc(t_hmc_s))

fprintf('acceptance rate: %.2f\n',100*sum(res_hmc.acpt)/ops.iters)  

% 

noBurnInIterations = 500;
noBins = 50;

ops = [];
ops.markovChain = res_hmc.q(T*nx+1,1:1:end);
ops.noBurnInIterations = noBurnInIterations;
ops.noBins = noBins;
ops.normalization = 'pdf';
ops.title = ['optimized'] ;
ops.subplots = 4;

res_plot = process_mcmc(ops);
% close(res_plot.fighandle);

tmp = diff(res_plot.binEdges);
d = tmp(1)/2;

ths_lo = res_plot.binEdges(1:end-1)+d;
pdf_lo = res_plot.binCount;


%%
lw = 1.5;
figure
plot(theta(param_index),0,'ko')
hold on
plot(ths_nom,pdf_nom,'linewidth',lw)
plot(ths_sgn,pdf_sgn,'linewidth',lw)
plot(ths_lo,pdf_lo,'linewidth',lw)
grid on
xlabel('parameter')
legend('true','nominal','sign of nominal','locally optimized')
title('posteriors from hmc')


%% compute posteriors with particle filter

% ths = linspace(0.9*theta(param_index),1.1*theta(param_index),500);
ths = linspace(-2.5,1.5,600);

num_particles = 10e3;

n = length(ths);
ls = zeros(1,n);
lls = zeros(1,n);

fprintf('Running particle filter...')
clear ops_pf
ops_pf.num_particles = num_particles;
ops_pf.model = model;
ops_pf.u = u;
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
ls_nom = ls/sl;


fprintf('Running particle filter...')
clear ops_pf
ops_pf.num_particles = num_particles;
ops_pf.model = model;
ops_pf.u = u_sgn;
ops_pf.y = y_sgn;

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
ls_sgn = ls/sl;


fprintf('Running particle filter...')
clear ops_pf
ops_pf.num_particles = num_particles;
ops_pf.model = model;
ops_pf.u = u_lo;
ops_pf.y = y_lo;
% ops_pf.u = u_lo_sgn;
% ops_pf.y = y_lo_sgn;

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
ls_lo = ls/sl;


if hessian_opt

fprintf('Running particle filter...')
clear ops_pf
ops_pf.num_particles = num_particles;
ops_pf.model = model;
ops_pf.u = u_hop;
ops_pf.y = y_hop;

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
ls_hop = ls/sl;

end

%

lw = 1.5;
figure
plot(theta(param_index),0,'ko')
hold on
plot(ths,ls_nom,'linewidth',lw)
plot(ths,ls_sgn,'linewidth',lw)
plot(ths,ls_lo,'linewidth',lw)

if hessian_opt  
    plot(ths,ls_hop,'linewidth',lw)
    legend('true','nominal','sign of nominal','locally optimized','hessian optimization')
else
    legend('true','nominal','sign of nominal','locally optimized')
end
    
grid on
xlabel('parameter')
title('posteriors from pf')


%%

figure
plot(1:T,u,'linewidth',4)
hold on
plot(1:T,u_sgn,'k','linewidth',3.5)
plot(1:T,u_lo,'r','linewidth',1.5)
set(gca,'ylim',[-1.1,1.5])
% set(gca,'xlim',[50 100])
xlabel('time')
title('inputs')
grid on
legend('nominal','sign of nominal',['iter ' num2str(lin_index)])


%%

figure
subplot(3,1,1)
plot(1:T,y,'linewidth',lw)
hold on
plot(1:T,y_sgn,'linewidth',lw)
plot(1:T,y_lo,'linewidth',lw)
title('output')
grid on

subplot(3,1,2)
plot(1:T,x,'linewidth',lw)
hold on
plot(1:T,x_sgn,'linewidth',lw)
plot(1:T,x_lo,'linewidth',lw)
title('state')
grid on

subplot(3,1,3)
plot(1:T,u,'linewidth',lw)
hold on
plot(1:T,u_sgn,'linewidth',lw)
plot(1:T,u_lo,'linewidth',lw)
title('input')
grid on



%% store some stuff

us_nom(:,:,trial_index) = u;
us_sgn(:,:,trial_index) = u_sgn;
us_lo(:,:,trial_index) = u_lo;

clear tmp
tmp.ths = ths_nom;
tmp.pdf = pdf_nom;
pdfs_nom{trial_index} = tmp;

tmp.ths = ths_sgn;
tmp.pdf = pdf_sgn;
pdfs_sgn{trial_index} = tmp;

tmp.ths = ths_lo;
tmp.pdf = pdf_lo;
pdfs_lo{trial_index} = tmp;


end

stop


%%

s_date = datestr(clock,'ddmmmyyyy_HHMM');
s_name = ['nonlin_mac_' s_date '.mat'];

save(s_name,'-v7.3')


%%

u_lo_sgn = sign(u_lo);
[y_lo_sgn,x_lo_sgn] = sim_flex_nl(model,theta,x1,u_lo_sgn,w,v); 


%% check norms

figure
plot(norms_us_lo)
set(gca,'yscale','log')
ylabel('|u(i) - u(i+1)|')
xlabel('iteration')
grid on


%%

length(input_storage_intervals)

figure

for i = 1:10
    
    subplot(5,2,i)
    if i == 1
        plot(u)
    else
        plot(us_inc_store(:,:,i-1))
    end
    hold on
    plot(us_inc_store(:,:,i))
    
    title(['iter ' num2str(i-1) ', and iter ' num2str(i)])
     
end


%%

figure

for i = 11:20
    
    subplot(5,2,i-10)
    plot(us_inc_store(:,:,i-1))
    hold on
    plot(us_inc_store(:,:,i))
    
    title(['iter ' num2str(input_storage_intervals(i-1)) ', and iter ' num2str(input_storage_intervals(i))])
     
end


%%

figure

for i = 21:29
    
    subplot(5,2,i-20)
    plot(us_inc_store(:,:,i-1))
    hold on
    plot(us_inc_store(:,:,i))
    
    title(['iter ' num2str(input_storage_intervals(i-1)) ', and iter ' num2str(input_storage_intervals(i))])
     
end


%%

figure
plot(u,'k','linewidth',1.5)
hold on
tmp_leg = {'initial'};
for i = [10:5:60]
    plot(us_inc_store(:,:,i))
    tmp_leg{end+1} = ['iter ' num2str(input_storage_intervals(i))]; 
end
legend(tmp_leg)
grid on

%%

figure;
plot(theta(param_index),0,'ko')
hold on
xlabel('parameter')
grid on
leg_tmp = {'true'};
title('posteriors on the fly')  

for i = [1,100,150]

    plot(ths_on_fly,posteriors_on_fly(:,:,i))
    leg_tmp{end+1} = ['iter ' num2str(i)];

end

legend(leg_tmp)


