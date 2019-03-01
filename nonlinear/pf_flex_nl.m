function [particles,weights,xf,loglik,ancestors] = pf_flex_nl(ops)
%UNTITLED15 Summary of this function goes here
% This version assumes that we begin from x_1 (i.e. there is no unobserved
% x0).

%% process

u = ops.u;
y = ops.y;

f = ops.model.f;
g = ops.model.g;

[~,T] = size(y);

theta = ops.theta;

Sw = theta(6);
Sv = theta(7);
S1 = theta(8);


N = ops.num_particles;

%%

% Initialization:
% --------------

particles = zeros(N,T); 
ancestors = zeros(N,T); % It's probably not really necessary to keep track
weights   = zeros(N,T);
loglik    = 0;

% Sample from initial state distribution
% let's assume that we know that the initial state is zero
particles(:,1) = mvnrnd(0,S1,N);

% Now we should use measurements in the weight update (I guess)?          
tmpWeights = lognorm(y(:,1),g(particles(:,1),u(:,1),theta),Sv);          

% Now we do a prelim normalization, and divide by the largest weight
% (I don't really know why)
maxWeight = max(tmpWeights);
% The exp() converts back to the actual distribution (rather than log)
tmpWeights = exp(tmpWeights - maxWeight); 
% Now we normalize to sum to unity
sumWeights = sum(tmpWeights);
weights(:,1) = tmpWeights/sumWeights;

% May as well compute the log-likelihood as we go...
loglik = loglik + maxWeight + log(sumWeights) - log(N);


% Loop:
% ----
% Use k to avoid confusion with time index t

for k = 2:T
   
% Resampling:
% Sample the ancestor indices
% We use sampling with replacement, with probabilities given by the weights
% (fortunately, there is a matlab fnct that does just this).
% if k == 20
%     dung= 1;
% end

newAncestors = randsample(N,N,true,weights(:,k-1)); 

ancestors(:,1:k-1) = ancestors(newAncestors, 1:k-1);
ancestors(:,k) = newAncestors;

% Propogate:
% Generate new states with dynamics (and process noise)
xtmp = particles(newAncestors,k-1);
% particles(:,k) = tha*xtmp + thb*xtmp./(1+xtmp.^2) + thc*ones(N,1)*u(:,k-1) + mvnrnd(0,Sw,N);
particles(:,k) = f(xtmp,u(:,k-1),theta) + mvnrnd(0,Sw,N);


% Compute weights:
% Evaluate new states with output distribution
% We are guided by JD here, and we evaluate the logarithm of the
% distribution
% tmpWeights = lognorm(y(:,k),thd*particles(:,k) + the*particles(:,k).^2 + ...
%              thg*particles(:,k).^3 + thf*u(:,k),Sv);
         
tmpWeights = lognorm(y(:,k),g(particles(:,k),u(:,k),theta),Sv);          
         

% Now we do a prelim normalization, and divide by the largest weight
% (I don't really know why)
maxWeight = max(tmpWeights);
% The exp() converts back to the actual distribution (rather than log)
tmpWeights = exp(tmpWeights - maxWeight); 
% Now we normalize to sum to unity
sumWeights = sum(tmpWeights);
weights(:,k) = tmpWeights/sumWeights;

% May as well compute the log-likelihood as we go...
loglik = loglik + maxWeight + log(sumWeights) - log(N);
        
end

% This makes no sense to me; shouldn't we be using the weighted particle
% system? Anyway, this is what JD has, and we won't really use this
% quantity anyway.
xf = mean(particles);


end

