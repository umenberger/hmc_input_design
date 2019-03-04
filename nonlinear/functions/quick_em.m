function res = quick_em(ops)

    N = 10;
    M = 100;
    
    em_model = ops.em_model;
    
    param_est = ops.parameter;
    
    theta = em_model.theta;

    y = ops.y;
    u = ops.u;
    
    T = size(y,2);
    
    max_iters = 150;
    
    params = zeros(1,max_iters);
    
    for iter = 1:max_iters
       
%% E step: smoother
% set parameter
    theta(1) = param_est;
    em_model.theta = theta;
    
% run smoother    
    [x_bwd,ll] = ffbsi(y, u, em_model, N, M);    

%% M step: least squares
    n_ = 0;
    d_ = 0;
    
    for i = 1:N
        x_ = x_bwd(:,:,i);
        r_ = em_model.dfdth(x_(:,1:T-1),u(:,1:T-1),theta);
        b_ = x_(:,2:T) - theta(2)*u(:,1:T-1);
        
        n_ = n_ + sum(b_.*r_);
        d_ = d_ + sum(r_.^2);
        
    end

    param_est = n_/d_;
    params(iter) = param_est;
 
    end

    res.params = params;

end