function [x_bwd, ll] = ffbsi(y, u, model, N, M, varargin)
% Runs a particle forward filter backwards simulator FFBSi on the model
% specifiend in MODEL and data in (Y,U) using N particles.

% The forward pass is done by a bootstrap particle filter on an IMPLICIT
% model of the form
%
% e(x(t+1)) = f(x(t),u(t)) + w(t),
%   y(t) = g(x(t)) + v(t).

% NOTE: we will essentially apply the standard FFBSi to a fully-determined
% model of the form:
% z(t+1) = f(e^{-1}(z(t)),u(t)) + w(t)
% y(t) = g(e^{-1}(z(t))) + v(t)
% in terms of the transformed state z = e(x).
% Rather than change all the variable names, we will refer to z as x, and z
% as x (hopefully this won't cause any confusion).

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
z = zeros(nx,T,M); % transformed particles (i.e. e(z) = x)

ll = zeros(T,1);

% x(:,1,:) = mu(theta); % Sample initial conditions
% x(:,1,:) = mu(theta,M); % Sample initial conditions
x(:,1,:) = mu(x(:,1,:),theta); % Sample initial conditions

%%

for t = 1:T
    
% Start by transforming the freshly sampled (i.e. not yet resampled) states
%     [z(:,t,:),~] = fsolve(@(d) e(d,theta) - x(:,t,:), zeros(nx,1,M), options);
    
%   minFunc (cannot handle 3D input) 
%     for m = 1:M
%         z(:,t,m) = minFunc(@(d) model.ede2(d,x(:,t,m),theta), zeros(model.nx,1,1),ops);
%     end    
   
% It is these transformed states that we should be putting into 
%     logw = v(z(:,t,:),y(t),y,theta);
	logw = v(x(:,t,:),y(t),y,theta);
	% Normalize weights
    wmax = max(logw);
	w(:,t) = exp(logw-wmax);
	w(:,t) = w(:,t)/sum(w(:,t));
    
    ll(t) = wmax + log(sum(w(:,t))) - log(M);
	
	% Resample
	ind = resampling(w(:,t),M);
	
% -------------------------------------------------------------------------
% Modified  
	% Predict next state:
    % Transform the state variable (for now, use fsolve)
    
%   fsolve (r2015b) 
%     [z(:,t,:),~] = fsolve(@(d) e(d,theta) - x(:,t,:), zeros(nx,1,M), options);
%     [z(:,t,:),~] = fsolve(@(d) e(d,theta) - x(:,t,:), z_prev, options); z_prev = z(:,t,ind);
%     z(:,t,:) = nthroot(x(:,t,:),3);
%   fsolve (r2016a)

%   minFunc (cannot handle 3D input) 
%     for m = 1:M
%         z(:,t,m) = minFunc(@(d) model.ede2(d,x(:,t,m),theta), zeros(model.nx,1,1),ops);
%     end
%     For now, initialize with zero (find something better later)

% %     try
% %         [z(:,t,:),~] = fsolve(@(d) e(d,theta) - x(:,t,:), zeros(nx,1,M), options);
%         [z(:,t,:),~] = fsolve(@(d) model.ede(d, x(:,t,:),theta), zeros(nx,1,M), options);    
% %     catch
% %         dum = 1;
% %         error('Something went wrong solving for z!\n')
% %     end

% Can we check for accuracy?
%     x_chk = e(z(:,t,:),theta);
%     fprintf('Error: t = %d.\n',t)
%     max(abs(x_chk-x(:,t,:)))

%     xpred = f(z(:,t,:),u(:,t),theta);
% -------------------------------------------------------------------------     
  xpred = f(x(:,t,:),u(:,t),theta);
% -------------------------------------------------------------------------     

	x(:,t+1,:) = xpred(ind) + r(xpred(ind),t,theta);
end
x_bwd = zeros(nx,T,N);% Backwards trajectories
% z_bwd = zeros(nx,T,N);

ind = resampling(w(:,T),N);
x_bwd(:,T,:) = x(:,T,ind);
% z_bwd(:,T,:) = z(:,T,ind);

% Remember: x = e(z); so we may as well take advantage of the fact that we
% have already computed z = e^{-1}(x).

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
% Modified         
%         x_t = x(:,t,I);
% %       We should have already computed the transformation...
%         z_t = z(:,t,I);
% %         [z_t,~] = fsolve(@(x) e(x,theta) - x_t, z_t, options); % The only time we used transformed variables is in f. 
%         p = squeeze(exp(q(x_t1-f(z_t,u(:,t),theta),t,theta)));
% -------------------------------------------------------------------------        
        x_t = x(:,t,I);
        p = squeeze(exp(q(x_t1-f(x_t,u(:,t),theta),t,theta)));
% -------------------------------------------------------------------------        
        
        accept = (U <= p); % rho has already been taken cara of here
        x_bwd(:,t,L(accept)) = x_t(:,:,accept);
%         z_bwd(:,t,L(accept)) = z_t(:,:,accept); % Same deal: store transformed state
        L = L(~accept);
        counter = counter + 1;
    end
    if ~isempty(L)
        
% -------------------------------------------------------------------------
% Modified
%         Again, we should be able to reuse the transformed variables.
% Thought: we have already computed this quantity as xpred, right?
% Depending on memory restrictions vs cost of evaluating f, it might be
% beneficial to store the result from above, right?
%             fxt = f(z(:,t,:),u(:,t),theta);
% -------------------------------------------------------------------------
        fxt = f(x(:,t,:),u(:,t),theta);
% ------------------------------------------------------------------------- 

        for j=L
            wt = log(w(:,t)) + squeeze(q(x_bwd(:,t+1,j)-fxt,t,theta));
            wt = exp(wt-max(wt));
            wt = wt/sum(wt);
            
            ind = find(rand(1) < cumsum(wt),1,'first');
            x_bwd(:,t,j) = x(:,t,ind);
%             z_bwd(:,t,j) = z(:,t,ind);
        end
    end

end

% Finally, to avoid confusion externally, we switch back:
% x_bwd is the smoothed state
% z_bwd is the smoothed transformed state, i.e., z = e(x)
% tmp = x_bwd;
% x_bwd = z_bwd;
% z_bwd = tmp;