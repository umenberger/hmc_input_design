%%

color_scheme = 0;

if color_scheme

    col_hmc = [1,0,0];
    col_dp = 'k';
    col_nom = [0,0,1];
    col_sgn = [0,0,0];
    
else

    col_nom = [1.000000 0.860000 0.350000]; 
    col_hmc = [1.000000 0.310000 0.000000]; 
    col_dp = [0.000000 0.420000 0.240000]; 
    col_sgn = [0.000000 0.420000 0.240000];
      
end


%% compute posteriors with particle filter

ths = linspace(-2.5,1.5,600);

num_particles = 5e3;
% num_particles = 1e3;


n = length(ths);
ls = zeros(1,n);
lls = zeros(1,n);

fprintf('Running particle filter...')
clear ops_pf

u_tmp = u;

[y_tmp,x] = sim_flex_nl(model,theta,x1,u_tmp,0*w,0*v);  

ops_pf.num_particles = num_particles;
ops_pf.model = model;
ops_pf.u = u_tmp;
ops_pf.y = y_tmp;

theta_tmp = theta;

for i = 1:n
    
    theta_tmp(param_index) = ths(i);
    ops_pf.theta = theta_tmp; 
    
    [particles,weights,xf,loglik,ancestors] = pf_flex_nl(ops_pf);
    
    lls(i) = (loglik);
    ls(i) = exp(loglik);
    
end
fprintf('done\n')
sl = trapz(ths,ls);
ls_nom = ls/sl;


fprintf('Running particle filter...')
clear ops_pf

u_tmp = u_sgn;

[y_tmp,x] = sim_flex_nl(model,theta,x1,u_tmp,0*w,0*v);  

ops_pf.num_particles = num_particles;
ops_pf.model = model;
ops_pf.u = u_tmp;
ops_pf.y = y_tmp;

theta_tmp = theta;

for i = 1:n
    
    theta_tmp(param_index) = ths(i);
    ops_pf.theta = theta_tmp; 
    
    [particles,weights,xf,loglik,ancestors] = pf_flex_nl(ops_pf);
    
    lls(i) = (loglik);
    ls(i) = exp(loglik);
    
end
fprintf('done\n')
sl = trapz(ths,ls);
ls_sgn = ls/sl;


fprintf('Running particle filter...')
clear ops_pf

u_tmp = u_lo;

[y_tmp,x] = sim_flex_nl(model,theta,x1,u_tmp,0*w,0*v);  

ops_pf.num_particles = num_particles;
ops_pf.model = model;
ops_pf.u = u_tmp;
ops_pf.y = y_tmp;

theta_tmp = theta;

for i = 1:n
    
    theta_tmp(param_index) = ths(i);
    ops_pf.theta = theta_tmp; 
    
    [particles,weights,xf,loglik,ancestors] = pf_flex_nl(ops_pf);
    
    lls(i) = (loglik);
    ls(i) = exp(loglik);
    
end
fprintf('done\n')
sl = trapz(ths,ls);
ls_lo = ls/sl;


%%

fs = 14;
transp = 0.2;

lw = 1.5;
figure
subplot(1,2,1)
plot(theta(param_index),0,'ko')
hold on
plot(ths,ls_nom,'linewidth',lw,'color',col_nom)
plot(ths,ls_sgn,'linewidth',lw,'color',col_sgn)
plot(ths,ls_lo,'linewidth',lw,'color',col_hmc)
 
plot(theta(param_index)*[1 1],[0 4],'color',0.5*ones(1,3))

tmp_leg = {'true parameter','nominal, $\tilde{u}$','pseudo-random','optimized, $u^*$'};

axis([-1.25 1.25,0,4])
legend boxoff

hf = fill([ths],[ls_nom],'b');
set(hf,'facecolor',col_nom)
set(hf,'facealpha',transp);
set(hf,'edgealpha',0);

hf = fill([ths],[ls_sgn],'b');
set(hf,'facecolor',col_sgn)
set(hf,'facealpha',transp);
set(hf,'edgealpha',0);  

hf = fill([ths],[ls_lo],'b');
set(hf,'facecolor',col_hmc)
set(hf,'facealpha',transp);
set(hf,'edgealpha',0);

xlabel('$\theta$','interpreter','latex')
ylabel('pdf','interpreter','latex')
grid on
legend(tmp_leg,'interpreter','latex','fontsize',fs)
% set(gca,'fontsize',fs)
% set(gca,'TickLabelInterpreter','latex')


%% inputs

lw = 1.9;

figure
plot(1:T,u,'-.','color',col_nom,'linewidth',lw)
hold on
plot(1:T,u_sgn,'-','color',col_sgn,'linewidth',lw)
plot(1:T,u_lo,'--','color',col_hmc,'linewidth',lw)

% set(gca,'ylim',[-1.1,1.1])
axis([0 30 1.1*u_min 1.1*u_max])

plot([0,T+1],u_max*[1,1],'color',0.5*ones(1,4))
plot([0,T+1],u_min*[1,1],'color',0.5*ones(1,4))
grid on

tmp_leg = {'nominal, $\tilde{u}$','pseudo-random','optimized, $u^*$'};

xlabel('time','interpreter','latex','fontsize',fs)
ylabel('input','interpreter','latex','fontsize',fs)
% grid on
% % title('T_2 relaxation posteriors')   
legend(tmp_leg,'interpreter','latex','fontsize',fs)
% set(gca,'fontsize',fs)
% set(gca,'TickLabelInterpreter','latex')
% legend boxoff
