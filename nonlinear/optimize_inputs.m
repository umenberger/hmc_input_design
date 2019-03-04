%%

%% experiment parameters

% input amplitude constraints
u_min = -1;
u_max = 1;

num_trials = 1;

num_resamples = 2; % number of times Hamiltonian states are resampled with hmc

num_gd_steps = 30; % number of gradient descent steps per set of hmc samples

%% nominal input

clear theta

% model parameters (true)
theta(1) = -0.5; theta(2) = 1; theta(3) = 1; theta(4) = 1; theta(5) = 0.1;
theta(6) = 0.01; theta(7) = 0.01; theta(8) = 0.01;
 
f = @(x,u,th) th(1)*(x.*(x-1).*(x+1))./(1+x.^2) + th(2)*u; 
g = @(x,u,th) th(3)*x + th(4)*x.^2 + th(5)*x.^3;

clear model
model.f = f;
model.g = g;

model.dfdx = @(x,u,th) th(1)*((1+x.^2).*(3*x.^2-1) - (x.^3-x).*(1+x.^2))./((1+x.^2).^2);
model.dgdx = @(x,u,th) th(3) + 2*th(4)*x + 3*th(5)*x.^2;
model.dfdth = @(x,u,th)(x.*(x-1).*(x+1))./(1+x.^2);

param_index = 1; % parameter to identify 

nx = 1;
ny = 1;
nu = 1;

T = 30; % length of input
x1 = 0;

model.x1 = x1;
model.nx = nx;
model.nu = nu;
model.ny = ny;

%% compute posteriors during optimization progress

post_on_fly = 1; % on/off

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



%% experimental trials


for trial_index = 1:num_trials

    
%% nominal input

    u = 0.1*mvnrnd(0,1,T)';

    u(u>u_max) = u_max;
    u(u<u_min) = u_min;

    w = 0*mvnrnd(0,theta(6),T)';
    v = 0*mvnrnd(0,theta(7),T)';

    [y,x] = sim_flex_nl(model,theta,x1,u,w,v);


%% hmc parameters

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

    num_samples = 40;

%%

    figure(100)
    plot(theta(param_index),0,'ko')
    hold on
    xlabel('parameter')
    grid on
    leg_hmc = {'true'};
    title('posteriors (via hmc)')


%% input saturated to constraints

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

        ths_tmp = res_plot.binEdges(1:end-1)+d;
        pdf_tmp = res_plot.binCount;    


%% gradient descent

% samples from hmc
        ofst = 500;
        del = 10;

        q_samples = res_hmc.q(:,ofst:del:ofst+(num_samples-1)*del);
        p_samples = res_hmc.p(:,ofst:del:ofst+(num_samples-1)*del); 

        clear ops

        ops = ops_hmc;

        ops.q_samples = q_samples;
        ops.p_samples = p_samples;

        gdss = 10; % initial stepsize

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

            if f_new < f_old % then the previous stepsize was fine
    
                u_gd_old = u_gd; % update input
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

        norms_us_lo(lin_index,trial_index) = norm(u_tmp - u_lo);

        % simulate
        [y_tmp,x_tmp] = sim_flex_nl(model,theta,x1,u_tmp,w,v);   

        u_lo = u_tmp;
        x_lo = x_tmp;
        y_lo = y_tmp;    

        fprintf('\niteration time: %.2f\n',toc(t_start_iter))

        if sum(lin_index == input_storage_intervals)

            us_inc_store(:,:,inc_st_index) = u_lo;

            inc_st_index = inc_st_index + 1;

        end    


        pause(0.1)

    %% posterior as we go

        if post_on_fly

            num_particles = 2e3;

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


end

