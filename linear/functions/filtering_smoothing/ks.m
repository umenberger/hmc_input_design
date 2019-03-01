function [xs,Ps,K,N,r] = ks(y,u,theta)
%KS Kalman smoother
%
%  [xs,Ps] = KS(y,u,theta) Compute the expected value and variance of the
%  states x given measurements y.
%  [~,~,~,~,K,M,r] = KS(y,u,theta) also returns the predictive
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
%   xs      - (nx * T+1)      expected value of x given y
%   Ps      - (nx * nx * T+1) variance of x given y
%   K       - (nx * ny * T)   predictive Kalman gain
%   N       - (nx * nx * T+1) quantity from backwards recursion
%   r       - (ny * T+1)      quantity from backwards recursion
%
% By: Johan Wågberg, Uppsala 2015
theta = check_input_dimensions(theta,u,y);
A = theta.A;
C = theta.C;

[xs,Ps,K,Lambda,epsilon] = kf(y,u,theta);

T = size(y,2);
nx = size(A,1);

if nargout > 2
    r = zeros(nx,T+1);
    N = zeros(nx,nx,T+1);
    for t = T:-1:1
        L = A-K(:,:,t)*C;
        r(:,t) = C'*(Lambda(:,:,t)\epsilon(:,t)) + L'*r(:,t+1);
        N(:,:,t) = C'*(Lambda(:,:,t)\C) + L'*N(:,:,t+1)*L;
        xs(:,t) = xs(:,t) + Ps(:,:,t)*r(:,t);
        Ps(:,:,t) = Ps(:,:,t) - Ps(:,:,t)*N(:,:,t)*Ps(:,:,t);
    end
else
    r = zeros(nx,1);
    N = zeros(nx);
    for t = T:-1:1
        L = A-K(:,:,t)*C;
        r = C'*(Lambda(:,:,t)\epsilon(:,t)) + L'*r;
        N = C'*(Lambda(:,:,t)\C) + L'*N*L;
        xs(:,t) = xs(:,t) + Ps(:,:,t)*r;
        Ps(:,:,t) = Ps(:,:,t) - Ps(:,:,t)*N*Ps(:,:,t);
    end
end

