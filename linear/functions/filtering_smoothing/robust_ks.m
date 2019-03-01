function [xs,Ps,K,N,r] = robust_ks(y,u,theta)
%ROBUST_KS Robust Kalman smoother for the states
%
%  Computes the smoothed distribution of the disturbances in a linear
%  Gaussian state space model by the method given in 1 implemented in
%  square root form.
%
%  [xs,Ps] = ROBUST_KS(y,u,theta) Compute the expected value and variance
%  of the states x given measurements y.
%  [~,~,~,~,K,M,r] = ROBUST_KS(y,u,theta) also returns the predictive
%  Kalman gain and two quantities from the backwards recursion.
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
%   y   - (ny * T)
%   u   - (nu * T)
%  theta   - struct with system parameters
%       .A  - (nx * nx) matrix
%       .B  - (nx * nu) matrix
%       .C  - (ny * nx) matrix
%       .D  - (ny * nu) matrix
%       .G  - (nx * nw) matrix
%       .Q  - (nw * nw) positive definite covariance matrix
%       .R  - (ny * ny) positive definite covariance matrix
%       .m1 - (nx * 1)  vector
%       .P1 - (nx * nx) positive definite covariance matrix
%
%  Outputs:
%   xs      - (nx * T+1)      expected value of x given y
%   Ps      - (nx * nx * T+1) variance of x given y
%   K       - (nx * ny * T)   predictive Kalman gain
%   N       - (nx * nx * T+1) square root quantity from backwards recursion
%   r       - (ny * T+1)      quantity from backwards recursion
%
% References:
%   1. Durbin J, Koopman SJ. Time series analysis by state space methods.
%      Oxford Univ. Press; 01/01/2012.
%
% By: Johan Wågberg, Uppsala 2015

theta = check_input_dimensions(theta,u,y);
A = theta.A;
C = theta.C;
nx = size(A,1);
T = size(y,2);

[xs,Ps,K,Lambda,epsilon] = robust_kf(y,u,theta);

if nargout > 2
    N = zeros(nx,nx,T+1);
    r = zeros(nx,T+1);
    Ps(:,:,T+1) = Ps(:,:,T+1)'*Ps(:,:,T+1);
    for t = T:-1:1
        L = A-K(:,:,t)*C;
        r(:,t) = C'*(Lambda(:,:,t)\(Lambda(:,:,t)'\epsilon(:,t))) + L'*r(:,t+1);
        Ms = triu(qr([Lambda(:,:,t)'\C;N(:,:,t+1)*L]));
        N(:,:,t) = Ms(1:nx,1:nx);
        
        xs(:,t) = xs(:,t) + Ps(:,:,t)'*Ps(:,:,t)*r(:,t);
        Ps(:,:,t) = (Ps(:,:,t)'*Ps(:,:,t)) - (N(:,:,t)*Ps(:,:,t)'*Ps(:,:,t))'*(N(:,:,t)*Ps(:,:,t)'*Ps(:,:,t));
    end
else
    N = zeros(nx);
    r = zeros(nx,1);
    Ps(:,:,T+1) = Ps(:,:,T+1)'*Ps(:,:,T+1);
    for t = T:-1:1
        L = A-K(:,:,t)*C;
        r = C'*(Lambda(:,:,t)\(Lambda(:,:,t)'\epsilon(:,t))) + L'*r;
        Ms = triu(qr([Lambda(:,:,t)'\C;N*L]));
        N = Ms(1:nx,1:nx);
        
        xs(:,t) = xs(:,t) + Ps(:,:,t)'*Ps(:,:,t)*r;
        Ps(:,:,t) = (Ps(:,:,t)'*Ps(:,:,t)) - (N*Ps(:,:,t)'*Ps(:,:,t))'*(N*Ps(:,:,t)'*Ps(:,:,t));
    end
end