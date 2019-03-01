% test particle smoother for EM purposes

em_model = model;

% Dimensions
em_model.nx = 1;
em_model.ny = 1;
em_model.nw = 1;
em_model.nv = 1;
em_model.nu = 1;

% note: parameters are variances
em_model.processnoise.sample = @(x,t,theta) sqrt(theta(6))*randn(size(x));
em_model.processnoise.eval = @(x,t,theta) -.5*x.^2/theta(6);
em_model.measurementnoise.eval = @(x,y,t,theta) -.5*(theta(3)*x + theta(4)*x.^2 + theta(5)*x.^3 - y).^2/theta(7);
em_model.measurementnoise.sample = @(x,t,theta) sqrt(theta(7))*randn(size(x));

em_model.initialcondition.sample = @(x,theta) sqrt(theta(8))*randn(size(x));

em_model.theta = theta;

%%

N = 10;
M = 100;

[x_bwd,ll] = ffbsi(y, u_tmp, em_model, N, M);

%%

figure
plot(1:T,x,'linewidth',1.5)
hold on
for i = 1:N
    plot(1:T,x_bwd(:,:,i),'color',[1,0,0,0.2])
end
























































