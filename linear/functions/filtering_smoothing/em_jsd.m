function [xf,P,xs,Ps,G] = em_jsd(y,u,A,B,C,D,N,Q,R,m1,P1)

% Slightly scrappy function to perform all the smoothing for an 
% application of the EM algorithm to LGSS models with states as latent
% variables
[nx,nu] = size(B);
[ny,T] = size(y);


%% Kalman Filtering

% There will be some excessive comments because its hard to keep track of 
% everything.
epsilon = zeros(ny,T);
Lambda = zeros(ny,ny,T);

% Not the predictor form gain
K = zeros(nx,ny,T);

% xf(:,t) = xf_{t|t-1}, i.e. predicted mean
xf = zeros(nx,T+1);
xf(:,1) = m1;

% P(:,:,t) is P_{t|t-1}, i.e. predicted covariance
P = zeros(nx,nx,T+1);
P(:,:,1) = P1;

% P_(:,:,t) is P_{t|t}, i.e. corrected covariance
P_ = zeros(nx,nx,T);

for t = 1:T
    
% Innovation at time t    
    epsilon(:,t) = y(:,t) - (C*xf(:,t) + D*u(:,t));
% Johan's lambda at time t    
    Lambda(:,:,t) = C*P(:,:,t)*C' + R;
% Standard Kalman gain (i.e. not Johan's Kalman predictor gain) at t
%  - it is missing the leading the A.
    K(:,:,t) = (P(:,:,t)*C')/Lambda(:,:,t);
% Corrected covariance P_{t|t}
    P_(:,:,t) = P(:,:,t) - K(:,:,t)*C*P(:,:,t);
% Filtered x_{t+1|t} - note the A...
    xf(:,t+1) = A*xf(:,t) + A*K(:,:,t)*epsilon(:,t) + B*u(:,t);
% Predicted covariance P_{t+1|t} - note the A...   
    P(:,:,t+1) = A*P(:,:,t)*(A-A*K(:,:,t)*C)' + N*Q*N';
end

%% Smoothing

% Smoothed x
xs = zeros(nx,T);

% Smoothed P
Ps = zeros(nx,nx,T);

% r_t and M_t
r = zeros(nx,1);
M = zeros(nx);

for t = T:-1:1
   
% Note the A...    
    L = A-A*K(:,:,t)*C;
% Obtaining r_{t-1} from r_t    
    r = C'*(Lambda(:,:,t)\epsilon(:,t)) + L'*r;
% Obtaining M_{t-1} from M_t
    M = C'*(Lambda(:,:,t)\C) + L'*M*L; 
    
% Smoothed quantities    
    xs(:,t) = xf(:,t) + P(:,:,t)*r;
    Ps(:,:,t) = P(:,:,t) - P(:,:,t)*M*P(:,:,t);
end
    
% Until 26/8/2016 this is what we used; where did this even come from?
% --------------------------------------------------------------------

% Cross covariance between states.
% This is a bit of a mess, mostly because the paper does not include all of
% the terms that you need and the indexes are also quite tricky to keep
% track of!
% Basically, G contains T matrices that range from M_{T+1|T} to 
% M_{2|T}, which are all the matrices that you need for the EM algorithm.

G = zeros(nx,nx,T);
G(:,:,T) = A*P_(:,:,T); % M_{T+1|T}
G(:,:,T-1) = (eye(nx) - K(:,:,T)*C)*A*P_(:,:,T-1); % M_{T|T}

for j = T-2:-1:1
  t = j+1; % Index correction...  
% J_t
    J = P_(:,:,t)*A'/P(:,:,t+1);
% J_{t-1}
    J_ = P_(:,:,t-1)*A'/P(:,:,t);    
 
% G_{t|T}    
    G(:,:,j) = P_(:,:,t)*J_' + J*(G(:,:,j+1)-A*P_(:,:,t))*J_';
    
end

% Reproducing what's in the Gibson/Ninness paper
% ----------------------------------------------

% G = zeros(nx,nx,T);
% J = zeros(nx,nx,T);
% 
% for t = 1:T
%     J(:,:,t) = P_(:,:,t)*A'/P(:,:,t+1);
% end
% 
% G(:,:,T) = (eye(nx) - K(:,:,T)*C)*A*P_(:,:,T-1);
% 
% for t = T-1:-1:2
%    
%     G(:,:,t) = P_(:,:,t)*J(:,:,t-1)' + J(:,:,t)*(G(:,:,t+1)-A*P_(:,:,t))*J(:,:,t-1)';
%     
% end







end

