%%

% theta(1) = -0.8; theta(2) = 1; theta(3) = 1; theta(4) = 1; theta(5) = 0.1;
% theta(6) = 0.1; theta(7) = 0.1; theta(8) = 0.01;
% 
% 
% f = @(x,u,th) th(1)*(x.*(x-1).*(x+1))./(1+x.^2) + th(2)*u; 
% % f = @(x,u,th) th(1)*(x.*(x-1).*(x+1).*(x-10).*(x+10)) + th(2)*u; 
% 
% g = @(x,u,th) th(3)*x + th(4)*x.^2 + th(5)*x.^3;
% 
% 
% xr = linspace(-10,10,100);
% 
% figure
% plot(xr,f(xr,0,theta))
% hold on
% plot(xr,xr)
% plot(xr,-xr)
% axis equal




%%

% simulate
close all

clear theta

% original
% theta(1) = -0.5; theta(2) = 1; theta(3) = 1; theta(4) = 1; theta(5) = 0.1;
% theta(6) = 0.1; theta(7) = 0.1; theta(8) = 0.01;
% % f = @(x,u,th) th(1)*(x.*(x-1).*(x+1)) + th(2)*u; 
% f = @(x,u,th) sign(th(1)*(x.*(x-1).*(x+1)) + th(2)*u).*min(abs(th(1)*(x.*(x-1).*(x+1)) + th(2)*u),1); 
% g = @(x,u,th) th(3)*x + th(4)*x.^2 + th(5)*x.^3;


% modified
theta(1) = -0.5; theta(2) = 1; theta(3) = 1; theta(4) = 1; theta(5) = 0.1;
theta(6) = 0.1; theta(7) = 0.1; theta(8) = 0.01;

f = @(x,u,th) th(1)*(x.*(x-1).*(x+1))./(1+x.^2) + th(2)*u; 
g = @(x,u,th) th(3)*x + th(4)*x.^2 + th(5)*x.^3;


clear model
model.f = f;
model.g = g;

% model.dfdx = @(x,u,th) th(1)*(3*x.^2-1);
model.dfdx = @(x,u,th) th(1)*((1+x.^2).*(3*x.^2-1) - (x.^3-x).*(1+x.^2))./((1+x.^2).^2);
model.dgdx = @(x,u,th) th(3) + 2*th(4)*x + 3*th(5)*x.^2;

% model.dfdth = @(x,u,th) (x.*(x-1).*(x+1));
% model.dfdth = @(x,u,th) sign((x.*(x-1).*(x+1))).*min(abs((x.*(x-1).*(x+1))),1);
model.dfdth = @(x,u,th) (x.*(x-1).*(x+1))./(1+x.^2);

nx = 1;
ny = 1;
nu = 1;

T = 30;
x1 = 0;
u = 0.1*mvnrnd(0,1,T)';

u_max = 1;
u_min = -1;

% u = sign(u);
u(u>u_max) = u_max;
u(u<u_min) = u_min;

u_sgn = u_max*sign(u);

w = 0*mvnrnd(0,theta(6),T)';
v = 0*mvnrnd(0,theta(7),T)';

w_max = 0.2;
w_min = -0.2;

w(w>w_max) = w_max;
w(u<w_min) = w_min;

[y,x] = sim_flex_nl(model,theta,x1,u,w,v);

figure
subplot(3,1,1)
plot(1:T,y)
title('output')
grid on

subplot(3,1,2)
plot(1:T,x)
title('state')
grid on

subplot(3,1,3)
plot(1:T,u)
title('input')
grid on

figure
xr = linspace(-3,3,100);
fr = f(xr,0,theta);
plot(xr,fr)
hold on
plot(xr,xr)
grid on

pause(0.01)

%% compute likelihood with pf

param_index = 1;

if param_index == 1
    ths = linspace(-2,0.5,500);
%     ths = linspace(theta(param_index)-0.1,theta(param_index)+0.1,500);
else
    ths = linspace(theta(param_index)-0.5,theta(param_index)+0.5,500);
%     ths = linspace(4.5,5.5,500);
end

num_particles = 3e3;

n = length(ths);
ls = zeros(1,n);
lls = zeros(1,n);
liks = zeros(1,n);


fprintf('Running particle filter...')

clear ops_pf

ops_pf.num_particles = num_particles;
ops_pf.model = model;
ops_pf.u = u;
ops_pf.y = y;

theta_tmp = theta;

for i = 1:n
    
    theta_tmp(param_index) = ths(i);
    ops_pf.theta = theta_tmp; 
    
    [particles,weights,xf,loglik,ancestors] = pf_flex_nl(ops_pf);
    
%     [~,~,~,liks(i),~] = pf_usual_nl_lik(ops_pf);    
    
    lls(i) = (loglik);
    ls(i) = exp(loglik);
    
end

%
sl = trapz(ths,ls);

ls_nom = ls/sl;

liks_nom = liks/trapz(ths,liks);

figure
plot(ths,ls_nom)
hold on
plot(ths,liks_nom)
legend('pf - loglik','pf - lik')
title('nominal inputs')

%% compute likelihood with pf

% u_sgn = sign(u);
[y_sgn,x_sgn] = sim_flex_nl(model,theta,x1,u_sgn,w,v);

n = length(ths);
ls = zeros(1,n);
lls = zeros(1,n);

fprintf('Running particle filter...')

clear ops_pf

ops_pf.num_particles = num_particles;
ops_pf.model = model;
ops_pf.u = u_sgn;
ops_pf.y = y_sgn;

theta_tmp = theta;

for i = 1:n
    
    theta_tmp(param_index) = ths(i);
    ops_pf.theta = theta_tmp; 
    
    [particles,weights,xf,loglik,ancestors] = pf_flex_nl(ops_pf);
    
%     [~,~,~,liks(i),~] = pf_usual_nl_lik(ops_pf);  
    
    lls(i) = (loglik);
    ls(i) = exp(loglik);
    
end

%
sl = trapz(ths,ls);

ls_sgn = ls/sl;

liks_sgn = liks/trapz(ths,liks);

figure
plot(ths,ls_sgn)
hold on
plot(ths,liks_sgn)
legend('pf - loglik','pf - lik')
title('sign inputs')

figure
% subplot(2,1,2)
plot(ths,ls_nom)
hold on
plot(ths,ls_sgn)
title('likelihood via pf')
legend('nominal','sign')
grid on
fprintf('done.\n')

%%

figure
subplot(3,1,1)
plot(1:T,y)
hold on
plot(1:T,y_sgn)
title('output')
grid on

subplot(3,1,2)
plot(1:T,x)
hold on
plot(1:T,x_sgn)
title('state')
grid on

subplot(3,1,3)
plot(1:T,u)
hold on
plot(1:T,u_sgn)
title('input')
grid on


% stop

%% run old linear hmc

if 0 

q_init = [x,theta(1)]';

theta_s = packTheta(theta(1),theta(3),theta(4),0,eye(nx),G,theta(6),theta(7),zeros(nx,1),theta(8));

ops_hmc_l.theta = theta_s;
ops_hmc_l.iters = 3e3;
ops_hmc_l.inertia = 1*eye(length(q_init));
ops_hmc_l.stepsize = 1e-2;
ops_hmc_l.simlength = 30;
ops_hmc_l.y = y;
ops_hmc_l.u = u;
ops_hmc_l.q_init = q_init;

res_hmc_l = hmc_fullstate_lgss_scalar(ops_hmc_l);

sum(res_hmc_l.acpt)/ops_hmc_l.iters

noBurnInIterations = 500;
noBins = 50;

pdfs = []; bes = [];

ftmp = figure;

for i = 1:1

ops = [];
ops.markovChain = res_hmc_l.q(T*nx+i,1:2:end);
ops.noBurnInIterations = noBurnInIterations;
ops.noBins = noBins;
ops.normalization = 'pdf';
ops.title = ['param ' num2str(i)] ;
ops.subplots = 4;

res_plot = process_mcmc(ops);
% close(res_plot.fighandle);

pdfs = [pdfs; res_plot.binCount];
bes = [bes; res_plot.binEdges];

figure(ftmp)
subplot(3,1,i)
tmp = diff(res_plot.binEdges);
d = tmp(1)/2;
plot(res_plot.binEdges(1:end-1)+d,res_plot.binCount)
hold on
plot(ths,ls/sl,'r')
legend('hmc','pf')

end

end

%% run hmc

use_sgn = 1;

if use_sgn
    
    y_use = y_sgn;
    u_use = u_sgn;
    
    q_init = [x_sgn,theta(param_index)]';

    
else
    
    y_use = y;
    u_use = u;
    
    q_init = [x,theta(param_index)]';

    
end


clear ops_hmc


ops_hmc.theta = theta;
ops_hmc.param_index = param_index;
ops_hmc.iters = 4e3;
ops_hmc.inertia = 1*eye(length(q_init));
ops_hmc.model = model;

% this is what we had been using
ops_hmc.stepsize = 0.1e-2;
ops_hmc.simlength = 75;

% ops_hmc.stepsize = 0.5e-2;
% ops_hmc.simlength = 200;

% ops_hmc.stepsize = 0.01e-2;
% ops_hmc.simlength = 200;

ops = ops_hmc;

ops.y = y_use;
ops.u = u_use;
ops.q_init = q_init;

fprintf('Running HMC...')

res_hmc = hmc_flex_nl(ops)

fprintf('done.\n')

sum(res_hmc.acpt)/ops.iters


%

noBurnInIterations = 500;
noBins = 50;

pdfs = []; bes = [];

ftmp = figure;

for i = 1:1

ops = [];
ops.markovChain = res_hmc.q(T*nx+i,1:1:end);
ops.noBurnInIterations = noBurnInIterations;
ops.noBins = noBins;
ops.normalization = 'pdf';
ops.title = ['param ' num2str(i)] ;
ops.subplots = 4;

res_plot = process_mcmc(ops);
% close(res_plot.fighandle);

pdfs = [pdfs; res_plot.binCount];
bes = [bes; res_plot.binEdges];

figure(ftmp)
subplot(1,1,i)
tmp = diff(res_plot.binEdges);
d = tmp(1)/2;
plot(res_plot.binEdges(1:end-1)+d,res_plot.binCount)
hold on
if use_sgn
    plot(ths,ls_sgn,'r')
else
    plot(ths,ls_nom,'r')
end
legend('hmc','pf')

end












































