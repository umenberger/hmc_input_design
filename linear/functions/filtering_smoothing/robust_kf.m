function [xp,Pp,K,Lambda,epsilon] = robust_kf(y,u,theta)
%ROBUST_KF Robust Kalman filter
%
%  [xp,Pp] = ROBUST_KF(y,u,theta) Computes the state predictions and
%  predictive covariances
%  [~,~,K,Lambda,epsilon] = ROBUST_KF(y,u,theta) also returns the
%  predictive Kalman gain, the innovation covariance and the innovations.
% 
%  For both the predictive covariance and the innovation covariance, a
%  square root is returned such that if P is the true covariance and V is
%  the returned value, then P = V'*V.
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
%   Pp      - (nx * nx * T+1)   square root predicted estimate covariance
%   K       - (nx * ny * T)     predictive Kalman gain
%   Lambda  - (ny * ny * T)     square root innovation covariance
%   epsilon - (ny * T)          innovation
%
% By: Johan Wågberg, Uppsala 2015
theta = check_input_dimensions(theta,u,y);
[A,B,C,D,~,G,Q,R,m1,P1] = unpackTheta(theta);

[ny,T] = size(y);
nx = size(A,1);
nw = size(Q,1);

Pc = chol(P1);

try
Qc = chol(Q);
catch
    Q = Q - 1.1*eye(size(Q))*min(eig(Q));
    Qc = chol(Q);
end
Rc = chol(R);

xp = zeros(nx,T+1);
xp(:,1) = m1;
Pp = zeros(nx,nx,T+1);
Pp(:,:,1) = Pc;
if nargout > 2
    epsilon = zeros(ny,T);
    Lambda = zeros(ny,ny,T);
    K = zeros(nx,ny,T);
    for t = 1:T
        epsilon(:,t) = y(:,t) - (C*xp(:,t) + D*u(:,t));
        H = [Pp(:,:,t)*C'  Pp(:,:,t)*A' ;
            Rc             zeros(ny,nx) ;
            zeros(nw,ny)   Qc*G'        ];
        Hs = qr(H);
        K(:,:,t) = (triu(Hs(1:ny,1:ny))\Hs(1:ny,ny+1:ny+nx))';
        Lambda(:,:,t) = triu(Hs(1:ny,1:ny));
        xp(:,t+1) = A*xp(:,t) + K(:,:,t)*epsilon(:,t) + B*u(:,t);
        Pp(:,:,t+1) = triu(Hs(ny+1:ny+nx,ny+1:ny+nx));
    end
    
else
    for t = 1:T
        epsilon = y(:,t) - (C*xp(:,t) + D*u(:,t));
        H = [Pp(:,:,t)*C'    Pp(:,:,t)*A'   ;
            Rc             zeros(ny,nx)  ;
            zeros(nw,ny)   Qc*G'         ];
        Hs = qr(H);
        K = (triu(Hs(1:ny,1:ny))\Hs(1:ny,ny+1:ny+nx))';
        xp(:,t+1) = A*xp(:,t) + K*epsilon + B*u(:,t);
        Pp(:,:,t+1) = triu(Hs(ny+1:ny+nx,ny+1:ny+nx));
    end
end