function l = robust_loglik(y,u,theta)
%LOGLIK  Log likelihood 
%
%  l = LOGLIK(y,u,theta) Computes the log likelihood for a linear Gaussian
%  state spaec model.
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
%   l - log likelihood
%
% By: Johan Wågberg, Uppsala 2015

[xp,~,~,Lambda,~] = robust_kf(y,u,theta);
C = theta.C;
D = theta.D;
T = size(y,2);
l = 0;
for t = 1:T
    l = l + mvnpdfln(y(:,t),C*xp(:,t) + D*u(:,t),Lambda(:,:,t)'*Lambda(:,:,t));
end