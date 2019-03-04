function [particles,weights,xf,loglik,ancestors] = pf_flex_nl(ops)
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
ancestors = zeros(N,T); 
weights   = zeros(N,T);
loglik    = 0;

% Sample from initial state distribution
particles(:,1) = mvnrnd(0,S1,N);

tmpWeights = lognorm(y(:,1),g(particles(:,1),u(:,1),theta),Sv);          

maxWeight = max(tmpWeights);
% The exp() converts back to the actual distribution (rather than log)
tmpWeights = exp(tmpWeights - maxWeight); 
% Now we normalize to sum to unity
sumWeights = sum(tmpWeights);
weights(:,1) = tmpWeights/sumWeights;

loglik = loglik + maxWeight + log(sumWeights) - log(N);


% Loop:
% ----
% Use k to avoid confusion with time index t

for k = 2:T

newAncestors = randsample(N,N,true,weights(:,k-1)); 

ancestors(:,1:k-1) = ancestors(newAncestors, 1:k-1);
ancestors(:,k) = newAncestors;

% Propogate:
% Generate new states with dynamics (and process noise)
xtmp = particles(newAncestors,k-1);
particles(:,k) = f(xtmp,u(:,k-1),theta) + mvnrnd(0,Sw,N);
         
tmpWeights = lognorm(y(:,k),g(particles(:,k),u(:,k),theta),Sv);          
         

% Now we do a prelim normalization, and divide by the largest weight
maxWeight = max(tmpWeights);
% The exp() converts back to the actual distribution (rather than log)
tmpWeights = exp(tmpWeights - maxWeight); 
% Now we normalize to sum to unity
sumWeights = sum(tmpWeights);
weights(:,k) = tmpWeights/sumWeights;

loglik = loglik + maxWeight + log(sumWeights) - log(N);
        
end

xf = mean(particles);


end

