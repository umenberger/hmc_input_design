function [cst,df,q,p] = nmpc_hmc_flex_nl_cost(ops,q_init,p_init)


% compute cost and gradient for HMC at a particular value of u, given
% initial conditions

%% process inputs

theta   = ops.theta; % include initial A value here
M           = ops.inertia;
eps         = ops.stepsize;
L           = ops.simlength;

u           = ops.u;
paramindex  = ops.param_index; % include initial A value here

q_nom = theta(paramindex);

model = ops.model;

f = ops.model.f;
g = ops.model.g;
dfdx = ops.model.dfdx;
dgdx = ops.model.dgdx;
dfdth = ops.model.dfdth;

if isfield(ops,'return_gradient')
    ret_grad = ops.return_gradient;
else
    ret_grad = 0;
end

Minv = inv(M);

nx = 1;
ny = 1;
T = size(u,2);

nq = length(q_init);

nth = nq - T*nx;

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
    [cst,q,p] = cost(u);
        
% compute a finite difference approximation of the gradient of the cost w.r.t.
% the input u
    if ret_grad
        tnu = length(u(:));
        df_fd = zeros(tnu,1);

        for i = 1:tnu

            utmp = u;
            utmp(i) = utmp(i) + del;

            df_fd(i) = (cost(utmp) - cst)/del;

        end
        df = df_fd;
    else     
        df = nan;
    end    


    
%%

    function [c,qs,ps] = cost(u_)
        
        if compute_approx_y
%             y_ = sim_usual_nl(theta,0,u_,zeros(1,T),zeros(1,T));
            y_ = sim_flex_nl(model,theta,0,u_,zeros(1,T),zeros(1,T));
        else
            y_ = y;
        end
        
        ps = zeros(nq,L);
        qs = zeros(nq,L);
        
        dU = (-dll_fullstate(q_init,u_,y_));
        
        ps(:,1) = p_init - (eps/2)*dU; % Initial half step 

        qs(:,1) = q_init + eps*ps(:,1); % full position step

        for t = 2:L

            dU = (-dll_fullstate(qs(:,t-1),u_,y_));
            
            ps(:,t) = ps(:,t-1) - (eps)*(dU); % full momentum step

            qs(:,t) = qs(:,t-1) + eps*ps(:,t); % full position step

        end    

    % add cost (maybe it's only necessary to have a terminal cost?)
        c = (qs(T*nx+1:end,L) - q_nom)'*(qs(T*nx+1:end,L) - q_nom);        

    end


%%


    function dll = dll_fullstate(q,u,y)
        
% In this case:
% [dll/dx; dll/db]

% slow and naive at first       
       qx = q(1:T*nx);
       qth = q(T*nx+1:T*nx+nth);
       
       qx = reshape(qx,nx,T);
       
       th_tmp = theta;
       th_tmp(paramindex) = qth;
       
%        ee_y = ys - (th_tmp(4)*qx + th_tmp(5)*qx.^2 + th_tmp(10)*qx.^3 + th_tmp(9)*us);
        ee_y = y - g(qx,u,th_tmp);
%        ee_x = qx(:,2:T) - (th_tmp(1)*qx(:,1:T-1) + th_tmp(2)*(qx(:,1:T-1))./(1+qx(:,1:T-1).^2) + th_tmp(3)*us(:,1:T-1));
        ee_x = qx(:,2:T) - f(qx(:,1:T-1),u(:,1:T-1),th_tmp);
       
% grad w.r.t. x    
%     gwrtx = 2*(th_tmp(4) + 2*th_tmp(5)*qx + 3*th_tmp(10)*qx.^2).*ee_y/th_tmp(7) - 2*[qx(:,1)/th_tmp(8),ee_x/th_tmp(6)] + ...
%             2*[(th_tmp(1) + th_tmp(2)*(1 - qx(:,1:T-1).^2)./((1+qx(:,1:T-1).^2).^2)).*ee_x,0]/th_tmp(6);

    gwrtx = 2*(dgdx(qx,u,th_tmp)).*ee_y/th_tmp(7) - 2*[qx(:,1)/th_tmp(8),ee_x/th_tmp(6)] + ...
            2*[(dfdx(qx(:,1:T-1),u(:,1:T-1),th_tmp)).*ee_x,0]/th_tmp(6);

        
       

% grad w.r.t. th
%     gwrtth = 0;

    gwrtth = sum(2*(dfdth(qx(:,1:T-1),u(:,1:T-1),th_tmp)).*ee_x/th_tmp(6));
    
%     if paramindex == 1
%         gwrtth = sum(2*qx(:,1:T-1).*ee_x/th_tmp(6));
%     elseif paramindex == 2
%         gwrtth = sum(2*(qx(:,1:T-1)./(1+qx(:,1:T-1).^2)).*ee_x/th_tmp(6));
%     end    
    
    dll = [gwrtx(:); gwrtth(:)];
    
    end




%% finite difference approx

    function fd = dUdq_fd(q,u,y)
        
        fd = zeros(nq);
        
        dUdq_0 = -dll_fullstate(q,u,y);
        
%         del = 1e-4;
        
        for ii = 1:nq
            
            tmpq = q;
            tmpq(ii) = tmpq(ii) + del;
            fd(:,ii) = (-dll_fullstate(tmpq,u,y) - dUdq_0)/del;
        
        end
        
    end



    function fd = dUdu_fd(q,u,y)
        
        fd = zeros(nq,T);
        
        dUdu_0 = -dll_fullstate(q,u,y);
        
%         del = 1e-4;
        
        tmpu = u(:);
        
        for ii = 1:(T*nu)
            
            tmpu_ = tmpu;
            tmpu_(ii) = tmpu_(ii) + del;
            u_ = reshape(tmpu_,nu,T);
            fd(:,ii) = (-dll_fullstate(q,u_,y) - dUdu_0)/del;
        
        end
        
    end


    
        
        

%%

    function ll = ll_fullstate(q,u,y)
        
        
        qx = q(1:T*nx);
        qth = q(T*nx+1:T*nx+nth);
       
        qx = reshape(qx,nx,T);
       
% compute equation errors
        Atmp = theta_tmp.A;
        Atmp(2,1) = qth(1);
        Atmp(2,2) = qth(2);

        Btmp = theta_tmp.B;
%         Btmp(1) = qth(3);

        Ctmp = theta_tmp.C;
        Dtmp = theta_tmp.D;

        Swtmp = theta_tmp.Q;
        Svtmp = theta_tmp.R;

        ee_eta = y - Ctmp*qx;
        ee_eps = qx(:,2:T) - Atmp*qx(:,1:T-1) - Btmp*u(:,1:T-1);  
        
        ll = -(ee_eta(:)'*(kron(eye(T),inv(Svtmp)))*ee_eta(:) + ee_eps(:)'*(kron(eye(T-1),inv(Swtmp)))*ee_eps(:));
        ll = ll - (qx(:,1)-theta_tmp.m1)'*(theta_tmp.P1\(qx(:,1)-theta_tmp.m1));
        
      
        
    end




 









end