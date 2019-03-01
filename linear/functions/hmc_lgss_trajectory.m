function [f,df,q,p] = hmc_lgss_trajectory(ops,q_init,p_init)

% compute cost and gradient for HMC at a particular value of u, given
% initial conditions

%% process inputs

theta       = ops.theta; % include initial A value here
M           = ops.inertia;
eps         = ops.stepsize;
L           = ops.simlength;

u           = ops.u;

param_select= ops.param_select;

psel = find(param_select);

q_nom      = ops.q_nom;

if isfield(ops,'return_gradient')
    ret_grad = ops.return_gradient;
else
    ret_grad = 0;
end

Minv = inv(M);

theta_tmp = theta;
Gamma_tr = [theta.A,theta.B;theta.C,theta.D];

[nx,nu]  = size(theta_tmp.B);
[T]   = size(u,2);
ny = size(theta_tmp.C,1);
nw = size(theta_tmp.G,1);

x0 = zeros(nx,1);
w0 = zeros(nw,T);
v0 = zeros(ny,T);


nq = length(q_init);

del = 1e-5; % increment for finite difference approximation

% if we supply an output, use it; otherwise, generate an approximate output
if isfield(ops,'y')
    y = ops.y;
    compute_approx_y = 0;
else
    compute_approx_y = 1;
end



%% do the calculation
    
% compute the cost given the input:
    [f,q,p] = cost(u);
        
% compute a finite difference approximation of the gradient of the cost w.r.t.
% the input u
    if ret_grad
        tnu = length(u(:));
        df_fd = zeros(tnu,1);

        for i = 1:tnu

            utmp = u;
            utmp(i) = utmp(i) + del;

            df_fd(i) = (cost(utmp) - f)/del;

        end
        df = df_fd;
    else     
        df = nan;
    end    


    
%%

    function [c,qs,ps] = cost(u_)
        
        if compute_approx_y
%             y_ = sim_usual_nl(theta,0,u_,zeros(1,T),zeros(1,T));
            [y_,~] = sim_lgss(theta,x0,u_,w0,v0);
        else
            y_ = y;
        end
        
        ps = zeros(nq,L);
        qs = zeros(nq,L);
        
        
        theta_tmp = update_theta(q_init);
        dU = -dll_lgss_gamma(y_,u_,theta_tmp,param_select);        
%         dU = (-dll_fullstate(q_init,u_,y_));
        
        ps(:,1) = p_init - (eps/2)*dU; % Initial half step 

        qs(:,1) = q_init + eps*ps(:,1); % full position step

        for t = 2:L

            theta_tmp = update_theta(qs(:,t-1));
            dU = -dll_lgss_gamma(y_,u_,theta_tmp,param_select); 
%             dU = (-dll_fullstate(qs(:,t-1),u_,y_));
            
            ps(:,t) = ps(:,t-1) - (eps)*(dU); % full momentum step

            qs(:,t) = qs(:,t-1) + eps*ps(:,t); % full position step

        end    

        c = (qs(:,L) - q_nom)'*(qs(:,L) - q_nom);        

    end



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





end