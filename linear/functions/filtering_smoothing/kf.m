function [xp,Pp,K,Lambda,epsilon] = kf(y,u,theta)
%KF  Kalman filter
%
%  [xp,Pp] = KF(y,u,theta) Computes the state predictions and predictive 
%  covariances
%  [~,~,K,Lambda,epsilon] = KF(y,u,theta) also returns the predictive
%  Kalman gain, the innovation covariance and the innovations.
% 
%  The state space model is given by
%   x(t+1) = A x(t) + B u(t) + G w(t),
%    y(t)  = C x(t) + D u(t) + v(t),
%  where
%   w(t) ~ N(0,Q),
%   v(t) ~ N(0,R),
%   x(1) ~ N(m1,P1).
%
%  Let
%   nx = states dimension
%   ny = output dimension
%   nu = input dimension
%   nw = process noise dimension
%
%  Inputs:
%         y  - (ny * T)
%         u  - (nu * T)
%   theta    - struct with system parameters
%        .A  - (nx * nx) matrix
%        .B  - (nx * nu) matrix
%        .C  - (ny * nx) matrix
%        .D  - (ny * nu) matrix
%        .G  - (nx * nw) matrix
%        .Q  - (nw * nw) positive definite covariance matrix
%        .R  - (ny * ny) positive definite covariance matrix
%        .m1 - (nx * 1)  vector
%        .P1 - (nx * nx) positive definite covariance matrix
%
%  Outputs:
%   xp      - (nx * T+1)        predicted state estimate
%   Pp      - (nx * nx * T+1)   predicted estimate covariance
%   K       - (nx * ny * T)     predictive Kalman gain
%   Lambda  - (ny * ny * T)     innovation covariance
%   epsilon - (ny * T)          innovation
%
% By: Johan Wågberg, Uppsala 2015

theta = check_input_dimensions(theta,u,y);
[A,B,C,D,~,G,Q,R,m1,P1] = unpackTheta(theta);

[ny,T] = size(y);
nx = size(A,1);

xp = zeros(nx,T+1);
xp(:,1) = m1;
Pp = zeros(nx,nx,T+1);
Pp(:,:,1) = P1;
if nargout > 2
    epsilon = zeros(ny,T);
    Lambda = zeros(ny,ny,T);
    K = zeros(nx,ny,T);
    for t = 1:T
        epsilon(:,t) = y(:,t) - (C*xp(:,t) + D*u(:,t));
        Lambda(:,:,t) = C*Pp(:,:,t)*C' + R;
        K(:,:,t) =(A*Pp(:,:,t)*C')/Lambda(:,:,t);
        xp(:,t+1) = A*xp(:,t) + K(:,:,t)*epsilon(:,t) + B*u(:,t);
        Pp(:,:,t+1) = A*Pp(:,:,t)*(A-K(:,:,t)*C)' + G*Q*G';
    end
else
    for t = 1:T
        epsilon = y(:,t) - (C*xp(:,t) + D*u(:,t));
        Lambda = C*Pp(:,:,t)*C' + R;
        K =(A*Pp(:,:,t)*C')/Lambda;
        xp(:,t+1) = A*xp(:,t) + K*epsilon + B*u(:,t);
        Pp(:,:,t+1) = A*Pp(:,:,t)*(A-K*C)' + G*Q*G';
    end
end