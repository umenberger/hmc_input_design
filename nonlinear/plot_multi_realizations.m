%%

num_realizations = 1;

nths = 100; % 600

ths = linspace(-1.5,1.0,nths);

num_particles = 5e3; % 10e3

data_psts_nom = zeros(num_realizations,nths,num_trials);
data_psts_sgn = zeros(num_realizations,nths,num_trials);
data_psts_lo = zeros(num_realizations,nths,num_trials);

rpf = figure;
plot(theta(param_index),0,'ko')
hold on

for ti = 1:num_trials
    
    for ri = 1:num_realizations    
       
% generate appropriate disturbances, to be used for all tests  
    w = 1*mvnrnd(0,1*theta(6),T)';
    v = 1*mvnrnd(0,1*theta(7),T)';
    
    
%     w_max = 0.1;
%     w_min = -0.1;
% 
%     w(w>w_max) = w_max;
%     w(w<w_min) = w_min;    
%     
    

% posteriors for nominal    
% -------------------------------------------------------------------------
    fprintf('\nNominal:\n')


%     u_tmp = us_nom(:,:,ti);
    u_tmp = u;
    
    [y,x] = sim_flex_nl(model,theta,x1,u_tmp,w,v);  
    
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
    
    data_psts_nom(ri,:,ti) = ls_tmp;
    
    
%% posteriors for sgn    
% -------------------------------------------------------------------------
    fprintf('\nSign:\n')


%     w = 1*mvnrnd(0,0.1,T)';
%     
%     w_max = 0.2;
%     w_min = -0.2;
% 
%     w(w>w_max) = w_max;
%     w(u<w_min) = w_min;     
    

%     u_tmp = us_nom(:,:,ti);
    u_tmp = 0.8/0.8*u_sgn;
    
    [y,x] = sim_flex_nl(model,theta,x1,u_tmp,w,v);  
    y
%     stop
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
    
    
    
    
%% posteriors for lo    
% -------------------------------------------------------------------------

    fprintf('\nLocal:\n')
    
%     u_tmp = us_nom(:,:,ti);
    u_tmp = 1*u_lo;
    
    [y,x] = sim_flex_nl(model,theta,x1,u_tmp,w,v);  
    
%     stop
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
    
    
%% figure 
% -------------------------------------------------------------------------    
    lw = 1.5;
    figure(rpf)
    plot(ths,data_psts_nom(ri,:,ti),'linewidth',lw)
    plot(ths,data_psts_sgn(ri,:,ti),'linewidth',lw)
    plot(ths,data_psts_lo(ri,:,ti),'linewidth',lw)

    if hessian_opt  
        plot(ths,ls_hop,'linewidth',lw)
        legend('true','nominal','sign of nominal','locally optimized','hessian optimization')
    else
        legend('true','nominal','sign of nominal','locally optimized')
    end

    grid on
    xlabel('parameter')
    title('posteriors from pf')    


    end

end













































