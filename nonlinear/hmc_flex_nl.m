function res = hmc_flex_nl(ops)
% hacky for now because parameters are hardcoded

%% Process inputs

theta       = ops.theta; % include initial A value here
paramindex  = ops.param_index; 
noiters     = ops.iters;
M           = ops.inertia;
eps         = ops.stepsize;
L           = ops.simlength;
% mu_pr = ops.priorMean;
% sg_pr = ops.priorVariance;
ys          = ops.y;
us          = ops.u;

q_init      = ops.q_init;


f = ops.model.f;
g = ops.model.g;
dfdx = ops.model.dfdx;
dgdx = ops.model.dgdx;
dfdth = ops.model.dfdth;

% tha = theta(1);
% thb = theta(2);
% thc = theta(3);
% thd = theta(4);
% 
% Sw = theta(5);
% Sv = theta(6);
% S1 = theta(7);

%% Run

Minv = inv(M);

T = size(ys,2);
nx = 1;
ny = 1;

nq = length(q_init);

nth = nq - T*nx;

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

    p = mvnrnd(zeros(nq,1),Minv)'; % Is (1/m) really the variance?
    
    ps(:,i) = p;
    
% Step 2: simulate Hamiltonian dynamics
% -------------------------------------

% Simulate for L timesteps

    p_ = p;
    q_ = q;
          

    dU = (-dll_fullstate(q_));
    
% check with finite differences
%     dU_fd = zeros(nq,1);
%     ftmp0 = ll_fullstate(q_);
%     del = 1e-4;
%     for j = 1:nq
%         qtmp = q_;
%         qtmp(j) = qtmp(j) + del;
%         dU_fd(j) = -(ll_fullstate(qtmp) - ftmp0)/del;
%     end
% 
%     [dU,dU_fd,dU-dU_fd]
    
    
    for t = 1:L
        
        p_ = p_ - (eps/2)*dU; % Initial half step        
        
        q_ = q_ + eps*(M\p_); % full position step 
        
        dU = (-dll_fullstate(q_)); % compute new gradient
        
%         dU
   
        p_ = p_ - (eps/2)*dU; % another half step
                   
    end
    
    p_ = -p_;
    

% Step 2: accept/reject
% ---------------------

%     if q_ < 1
    if 1

    U_c = -ll_fullstate(q);
    
    U_p = -ll_fullstate(q_);    
    
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


    function ll = ll_fullstate(q)
        
        
        qx = q(1:T*nx);
        qth = q(T*nx+1:T*nx+nth);
       
        qx = reshape(qx,nx,T);
        
       th_tmp = theta;
       th_tmp(paramindex) = qth;
       
%        ee_y = ys - (th_tmp(4)*qx + th_tmp(5)*qx.^2 + th_tmp(10)*qx.^3 + th_tmp(9)*us);
       ee_y = ys - g(qx,us,th_tmp);
       
%        ee_x = qx(:,2:T) - (th_tmp(1)*qx(:,1:T-1) + th_tmp(2)*(qx(:,1:T-1))./(1+qx(:,1:T-1).^2) + th_tmp(3)*us(:,1:T-1));  
       ee_x = qx(:,2:T) - f(qx(:,1:T-1),us(:,1:T-1),th_tmp);
        
        ll = -(sum((ee_y.^2)/th_tmp(7)) + sum((ee_x.^2)/th_tmp(6)) + (qx(:,1)^2)/th_tmp(8)); 
%         ll = -(sum((ee_y.^2)/th_tmp(7)));
        
    end




    function dll = dll_fullstate(q)
        
% In this case:
% [dll/dx; dll/db]

% slow and naive at first       
       qx = q(1:T*nx);
       qth = q(T*nx+1:T*nx+nth);
       
       qx = reshape(qx,nx,T);
       
       th_tmp = theta;
       th_tmp(paramindex) = qth;
       
%        ee_y = ys - (th_tmp(4)*qx + th_tmp(5)*qx.^2 + th_tmp(10)*qx.^3 + th_tmp(9)*us);
        ee_y = ys - g(qx,us,th_tmp);
%        ee_x = qx(:,2:T) - (th_tmp(1)*qx(:,1:T-1) + th_tmp(2)*(qx(:,1:T-1))./(1+qx(:,1:T-1).^2) + th_tmp(3)*us(:,1:T-1));
        ee_x = qx(:,2:T) - f(qx(:,1:T-1),us(:,1:T-1),th_tmp);
       
% grad w.r.t. x    
%     gwrtx = 2*(th_tmp(4) + 2*th_tmp(5)*qx + 3*th_tmp(10)*qx.^2).*ee_y/th_tmp(7) - 2*[qx(:,1)/th_tmp(8),ee_x/th_tmp(6)] + ...
%             2*[(th_tmp(1) + th_tmp(2)*(1 - qx(:,1:T-1).^2)./((1+qx(:,1:T-1).^2).^2)).*ee_x,0]/th_tmp(6);

    gwrtx = 2*(dgdx(qx,us,th_tmp)).*ee_y/th_tmp(7) - 2*[qx(:,1)/th_tmp(8),ee_x/th_tmp(6)] + ...
            2*[(dfdx(qx(:,1:T-1),us(:,1:T-1),th_tmp)).*ee_x,0]/th_tmp(6);

        
        

% grad w.r.t. th
%     gwrtth = 0;

    gwrtth = sum(2*(dfdth(qx(:,1:T-1),us(:,1:T-1),th_tmp)).*ee_x/th_tmp(6));
    
%     if paramindex == 1
%         gwrtth = sum(2*qx(:,1:T-1).*ee_x/th_tmp(6));
%     elseif paramindex == 2
%         gwrtth = sum(2*(qx(:,1:T-1)./(1+qx(:,1:T-1).^2)).*ee_x/th_tmp(6));
%     end    
    
    dll = [gwrtx(:); gwrtth(:)];
    
    end
        



end

