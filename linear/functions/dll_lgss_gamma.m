function df = dll_lgss_gamma(y,u,theta,param_select)

% compute the gradient of the loglik w.r.t. to specified parameters in
% Gamma = [A,B;C,D]

    [A,B,C,D,~,G,Sw,Sv,mu,S1] = unpackTheta(theta);
    
    Pi = blkdiag(Sw,Sv);

    Gamma = [A,B;C,D];
    
    Phi_22 = y*y';
    Psi_22 = y*u';
    Sigma_22 = u*u';    
    
    T = size(y,2);

%% run smoother

    [xf,Pf,xs,Ps,cM] = em_jsd(y,u,A,B,C,D,G,Sw,Sv,mu,S1);
     
% Phi
    Phi_11 = [xs(:,2:T),xf(:,T+1)]*[xs(:,2:T),xf(:,T+1)]' + sum(Ps(:,:,2:T),3) + Pf(:,:,T+1);     
    Phi_12 = [xs(:,2:T),xf(:,T+1)]*y';       
       
    Phi = [Phi_11, Phi_12; Phi_12', Phi_22]/T;
    
% (Psi)
    Psi_11 = [xs(:,2:T),xf(:,T+1)]*xs' + sum(cM,3);
    Psi_12 = [xs(:,2:T),xf(:,T+1)]*u';
    Psi_21 = y*xs';
      
    Psi = [Psi_11, Psi_12; Psi_21, Psi_22]/T;
    
% Sigma
    Sigma_11 = xs*xs' + sum(Ps,3);
    Sigma_12 = xs*u';
    
    Sigma = [Sigma_11, Sigma_12; Sigma_12', Sigma_22]/T;
    
    
%% compute gradient
    
    df = Pi\Psi + (Pi')\Psi - (Pi')\(Gamma*Sigma') - Pi\(Gamma*Sigma);
    
    df = 0.5*T*df(param_select);































    