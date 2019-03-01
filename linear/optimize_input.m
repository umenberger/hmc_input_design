%%

% code to:
% - randomly sample an initial input
% - run the proposed input design algorithm to locally optimize the input

%% experiment parameters

num_trials = 1;

num_resamples = 1; % number of times to resample using HMC

num_gd_steps = 2; % number of gradient descent steps to perform

compute_posteriors = 1;


%% system

% True system:
A = [0.5, 0.2; -0.2,0.7];
B = [0;1];
C = [1,0];
D = 0.01;
G = eye(2);
Sw = 1e-2*eye(2);
Sv = 1e-2;
S1 = 1e0*eye(2);

T = 50; % input length

[ny,nu] = size(D);
[nx,nw] = size(G);

% Input magnitude limits
u_min = -1;
u_max = 1;

w0 = 0*mvnrnd(zeros(nw,1),Sw,T)';
v0 = 0*mvnrnd(zeros(ny,1),Sv,T)';

theta_true = packTheta(A,B,C,D,eye(nx),G,Sw,Sv,zeros(nx,1),S1);

%% parameter selection

% Select which parameters you want from [A,B;C,D]
A_ = zeros(2);
A_(2,1:2) = 1; % select A(2,1) and A(2,2)
B_ = zeros(2,1);
C_ = zeros(1,2);
D_ = 0;

ps = [A_,B_;C_,D_];

param_select = logical(ps(:));

Gamma = [A,B;C,D];
% Gamma(param_select)

%% options for computing posterior

% posterior computation
n = 80;
nth = sum(param_select);
ths = zeros(nth,n);
ths(1,:) = linspace(-1.2,1.4,n);
ths(2,:) = linspace(-0.1,1.1,n);

clear ops_posterior
ops_posterior.theta_range = ths;
ops_posterior.theta = theta_true;

% posteriors
posteriors_nom = zeros(nth,n,num_trials);
posteriors_sgn = zeros(nth,n,num_trials);
posteriors_lo = zeros(nth,n,num_trials);


%% hmc parameters

clear ops_hmc

ops_hmc.theta           = theta_true;
ops_hmc.iters           = 600;
ops_hmc.inertia         = eye(sum(param_select));
ops_hmc.stepsize        = 1e-2;
ops_hmc.simlength       = 30;
ops_hmc.param_select    = param_select;
ops_hmc.q_init          = Gamma(param_select);
ops_hmc.q_nom           = Gamma(param_select);

num_samples = 40;
ofst = 100;
del = 10;

%%

us_nom = zeros(nu,T,num_trials);
us_sgn = zeros(nu,T,num_trials);
us_lo = zeros(nu,T,num_trials);

clear pdfs_nom pdfs_sgn pdfs_lo

norms_us_lo = zeros(num_resamples,num_trials);


%% begin trials


for trial_index = 1:num_trials

    
    fprintf('\nBeginning trial %d\n',trial_index)
    fprintf('-----------------------------------\n')

    t_trial_start = tic;    
    

%% nominal input

    u = 0.1*mvnrnd(zeros(nu,1),eye(nu),T)';
    
    us_nom(:,:,trial_index) = u;
    
    u(u>1) = u_max;
    u(u<-1) = u_min;
    
    [y,x] = sim_lgss(theta_true,zeros(nx,1),u,w0,v0);
    
    if compute_posteriors
        ops = ops_posterior;
        ops.u = u;
        ops.y = y;

        posteriors_nom(:,:,trial_index) = hack_2param_posterior_lgss(ops);
    end
    
    
%% sign of inputs

    u_sgn = sign(u);    
    us_sgn(:,:,trial_index) = u_sgn;

    [y_sgn,x_sgn] = sim_lgss(theta_true,zeros(nx,1),u_sgn,w0,v0);

    if compute_posteriors
        ops = ops_posterior;
        ops.u = u_sgn;
        ops.y = y_sgn;

        posteriors_sgn(:,:,trial_index) = hack_2param_posterior_lgss(ops);
    end    
    

%% iteratively improve nominal input

    x_tmp = x;
    y_tmp = y;
    u_tmp = u;

    u_lo = u_tmp;

%% sample from hmc

    for lin_index = 1:num_resamples

        t_start_iter = tic;

    %% run hmc

        q_init = Gamma(param_select);

        ops = ops_hmc;

        ops.y           = y_tmp;
        ops.u           = u_tmp;
        ops.q_init      = q_init;

        fprintf('Running HMC...')

        res_hmc = hmc_lgss_gamma(ops);      

        fprintf('done.\n')   


    %% local optimization

    % *extremely* simple implementation of projected gradient descent 

    % samples
        q_samples = res_hmc.q(:,ofst:del:ofst+(num_samples-1)*del);
        p_samples = res_hmc.p(:,ofst:del:ofst+(num_samples-1)*del); 

        clear ops

        ops = ops_hmc;

        ops.q_samples = q_samples;
        ops.p_samples = p_samples;

        gdss = 5; % initial stepsize

        u_gd = u_tmp;
        u_gd_old = u_gd;

        f_old = inf;

        for i = 1:num_gd_steps

            [f_new,df] = hmc_lgss_multicost(u_gd,ops); % compute gradient

            fprintf('\tcost at start of iter %d: %.5e, f_old %.5e\n',i,f_new*size(q_samples,2),f_old*size(q_samples,2))

            if f_new < f_old % then the previous stepsize was fine

                u_gd_old = u_gd; % update input
                f_old = f_new;
                u_gd = u_gd - gdss*df'; % take a new descent step             
            else
                u_gd = u_gd_old;
                gdss = gdss/2;       
                fprintf('changing stepsize: %.5e\n',gdss)            
            end

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
        [y_tmp,x_tmp] = sim_lgss(theta_true,zeros(nx,1),u_tmp,w0,v0);

        u_lo = u_tmp;
        x_lo = x_tmp;
        y_lo = y_tmp;    

        fprintf('\niteration time: %.2f\n',toc(t_start_iter))

        pause(0.01)

    end

    if compute_posteriors
        ops = ops_posterior;
        ops.u = u_lo;
        ops.y = y_lo;
        posteriors_lo(:,:,trial_index) = hack_2param_posterior_lgss(ops);
    end 


end



