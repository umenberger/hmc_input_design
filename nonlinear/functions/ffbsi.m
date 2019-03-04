function [x_bwd, ll] = ffbsi(y, u, model, N, M, varargin)
% Runs a particle forward filter backwards simulator FFBSi on the model
% specifiend in MODEL and data in (Y,U) using N particles.

% inputs:
%		y = ny * T - measurements
%       u = nu * T - inputs
%		model -
%			model.nx = dimension of state space
%           model.e = state transformation
%			model.f = prediction
%			model.g = measurement
%			model.processnoise.sample = sample noise
%			model.processnoise.eval = log density of process noise
%			model.measurementnoise.sample = sample measurement noise
%			model.measurementnoise.eval = log density of measurement noise
%			model.initialcondition.sample = sample initial state
%       N = number of backwards trajectories
%		M = number of particles in forward pass

% outputs:
%       x_bwd - smoothed states
%       z_bwd - smoothed transformed states, i.e., z = e(x)


if nargin < 4
    M = N;
end

nx = model.nx;                      % dimension of state spare
% e = model.e;                        % state transformation
f = model.f;                        % determinitstic prediction
r = model.processnoise.sample;      % sample process noise
q = model.processnoise.eval;        % log density of process noise
v = model.measurementnoise.eval;    % log density of measurement noise
mu = model.initialcondition.sample; % sample initial state
T = size(y,2);                      % Number of time steps
theta = model.theta;                % parameter vector

x = zeros(nx,T,M); % particles
w = zeros(M,T+1); % particle weights

ll = zeros(T,1);

x(:,1,:) = mu(x(:,1,:),theta); % Sample initial conditions

%%

for t = 1:T
    
	logw = v(x(:,t,:),y(t),y,theta);
	% Normalize weights
    wmax = max(logw);
	w(:,t) = exp(logw-wmax);
	w(:,t) = w(:,t)/sum(w(:,t));
    
    ll(t) = wmax + log(sum(w(:,t))) - log(M);
	
	% Resample
	ind = resampling(w(:,t),M);
	
% -------------------------------------------------------------------------     
  xpred = f(x(:,t,:),u(:,t),theta);
% -------------------------------------------------------------------------     

	x(:,t+1,:) = xpred(ind) + r(xpred(ind),t,theta);
end
x_bwd = zeros(nx,T,N);% Backwards trajectories

ind = resampling(w(:,T),N);
x_bwd(:,T,:) = x(:,T,ind);

for t = T-1:-1:1
    bins = [0 cumsum(w(:,t))'];
    L = 1:N;
    counter = 0;
    while ~isempty(L) && counter < M/5
        n = length(L);
        [~,I] = histc(rand(n,1), bins);
        U = rand(n, 1);
        x_t1 = x_bwd(:,t+1,L);
        
% -------------------------------------------------------------------------        
        x_t = x(:,t,I);
        p = squeeze(exp(q(x_t1-f(x_t,u(:,t),theta),t,theta)));
% -------------------------------------------------------------------------        
        
        accept = (U <= p); % rho has already been taken care of here
        x_bwd(:,t,L(accept)) = x_t(:,:,accept);
        L = L(~accept);
        counter = counter + 1;
    end
    if ~isempty(L)
        
% -------------------------------------------------------------------------
        fxt = f(x(:,t,:),u(:,t),theta);
% ------------------------------------------------------------------------- 

        for j=L
            wt = log(w(:,t)) + squeeze(q(x_bwd(:,t+1,j)-fxt,t,theta));
            wt = exp(wt-max(wt));
            wt = wt/sum(wt);
            
            ind = find(rand(1) < cumsum(wt),1,'first');
            x_bwd(:,t,j) = x(:,t,ind);
        end
    end

end
