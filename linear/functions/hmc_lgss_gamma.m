function res = hmc_lgss_gamma(ops)
% parameters not hardcoded.

%% Process inputs

theta       = ops.theta; % include initial A value here
noiters     = ops.iters;
M           = ops.inertia;
eps         = ops.stepsize;
L           = ops.simlength;
% mu_pr = ops.priorMean;
% sg_pr = ops.priorVariance;
ys          = ops.y;
us          = ops.u;

param_select= ops.param_select;

psel = find(param_select);

q_init      = ops.q_init;

%% Run

Minv = inv(M);

theta_tmp = theta;
Gamma_tr = [theta.A,theta.B;theta.C,theta.D];

[nx,nu]  = size(theta_tmp.B);
[ny,T]   = size(ys);

nq = length(q_init);

qs = zeros(nq,noiters); % stored positions
ps = zeros(nq,noiters); % stored positions
as = zeros(1,noiters); % stored accept probs
acpt = zeros(1,noiters); % accepted?

q = q_init;

for i = 1:noiters
   
% We will store first,
    qs(:,i) = q;
    
% Step 1: sample momentum variable
% --------------------------------

    p = mvnrnd(zeros(nq,1),Minv)'; 
    
    ps(:,i) = p;
    
% Step 2: simulate Hamiltonian dynamics
% -------------------------------------

% Simulate for L timesteps

    p_ = p;
    q_ = q;
          
    theta_tmp = update_theta(q_);
    dU = -dll_lgss_gamma(ys,us,theta_tmp,param_select);
    U_c = -robust_loglik(ys,us,theta_tmp);
%     [U_c,dll] = dll_fd(theta_tmp);
%     dU = -dll;

    
    for t = 1:L
        
        p_ = p_ - (eps/2)*dU; % Initial half step        
        
        q_ = q_ + eps*(M\p_); % full position step 
        
        theta_tmp = update_theta(q_);
        dU = -dll_lgss_gamma(ys,us,theta_tmp,param_select); % compute new gradient  
   
        p_ = p_ - (eps/2)*dU; % another half step
                   
    end
    
    p_ = -p_;
    

% Step 2: accept/reject
% ---------------------

    if 1
   
    U_p = -robust_loglik(ys,us,theta_tmp);
    
    K_c = 0.5*(p'/M)*p;
    
    K_p = 0.5*(p_'/M)*p_;
    
    a = exp(U_c - U_p + K_c - K_p);
    
    else
        
    a = -1; % reject
    
    end
    
   
    as(i) = a;
    
    u = rand(1);
    
    if u <= a
        q = q_;
        p = p_;
        acpt(i) = 1;
    end
      
    
end

%% Return some results

res.q = qs;
res.p = ps;
res.acpt = acpt;



%%

    function th = update_theta(q)
        
        tmp = Gamma_tr;
        tmp(param_select) = q;
        th = theta;
        th.A = tmp(1:nx,1:nx);
        th.B = tmp(1:nx,nx+1:end);
        th.C = tmp(nx+1:end,1:nx);
        th.D = tmp(nx+1:end,nx+1:end);
        
    end


%% 
    function [ll0,dll] = dll_fd(th)
        
        
        del = 1e-4;

        th_tmp = th;

        ll0 = robust_loglik(ys,us,th);

        dll = zeros(nq,1);      

        for ii = 1:length(psel)
            
            jj = psel(ii);

            tmp = [th.A,th.B;th.C,th.D];
            tmp(jj) = tmp(jj) + del;
            A_ = tmp(1:nx,1:nx);
            B_ = tmp(1:nx,nx+1:end);
            C_ = tmp(nx+1:end,1:nx);
            D_ = tmp(nx+1:end,nx+1:end);

            th_tmp.A = A_;
            th_tmp.B = B_;
            th_tmp.C = C_;
            th_tmp.D = D_;

            dll(ii) = (robust_loglik(ys,us,th_tmp) - ll0)/del;

        end        
        
        
        
    end

        







end

