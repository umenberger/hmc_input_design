%% set colors

c_nom = [0,0,1];
c_hmc = [1,0,0];
c_sgn = [0,0,0];


%% plot the inputs

fs = 14;

figure
plot(1:T,u,'-.','linewidth',2,'color',[c_nom,1])
hold on
plot(1:T,u_sgn,'linewidth',3.0,'color',[c_sgn,0.5])
plot(1:T,u_lo,'--','linewidth',1.2,'color',c_hmc)
set(gca,'ylim',[-1.1,1.1])
% set(gca,'xlim',[50 100])
xlabel('time','interpreter','latex')
title('inputs','interpreter','latex')
grid on
legend({'nominal, $\tilde{u}$','sign of nominal, $u_\textup{sgn}$','optimized, $u^*$'},'interpreter','latex','fontsize',fs)
set(gca,'fontsize',fs)
set(gca,'TickLabelInterpreter','latex')


%% plot the posteriors   

transp = 0.05;

fs = 14;
lw = 1.2;

figure
subplot(1,2,1)

plot(nan,nan,'ko')
hold on
plot(nan,nan,'color',c_hmc,'linewidth',lw)
plot(nan,nan,'color',c_sgn,'linewidth',lw)
%     plot(nan,nan,'color',c_nom,'linewidth',lw)

plot(A(2,1),0,'ko')
hold on
plot(ths(1,:),posteriors_nom(1,:,trial_index),'color',c_nom,'linewidth',lw)
plot(ths(1,:),posteriors_sgn(1,:,trial_index),'color',c_sgn,'linewidth',lw)
plot(ths(1,:),posteriors_lo(1,:,trial_index),'color',c_hmc,'linewidth',lw)
plot([A(2,1) A(2,1)],[0,10],'color',[0.5,0.5,0.5])
xlabel('$A_{21}$','interpreter','latex')
ylabel('pdf','interpreter','latex')
grid on
axis([-1 1 0 3])

hf = fill([ths(1,:)],[posteriors_nom(1,:,trial_index)],'b');
set(hf,'facecolor',c_nom)
set(hf,'facealpha',transp);
set(hf,'edgealpha',0);

hf = fill([ths(1,:)],[posteriors_sgn(1,:,trial_index)],'b');
set(hf,'facecolor',c_sgn)
set(hf,'facealpha',transp);
set(hf,'edgealpha',0);  

hf = fill([ths(1,:)],[posteriors_lo(1,:,trial_index)],'b');
set(hf,'facecolor',c_hmc)
set(hf,'facealpha',transp);
set(hf,'edgealpha',0);     


%     legend({'$\theta^*$','nominal, $\tilde{u}$','sign of nominal, $u_\textup{sgn}$','optimized, $u^*$'},'interpreter','latex','fontsize',fs)
legend({'$\theta^*$','optimized, $u^*$','sign of nominal, $u_\textup{sgn}$'},'interpreter','latex','fontsize',fs)
legend boxoff

set(gca,'fontsize',fs)
set(gca,'TickLabelInterpreter','latex')

%     figure
subplot(1,2,2)

plot(nan,nan,'color',c_nom,'linewidth',lw)
hold on

plot(A(2,2),0,'ko')
hold on
plot(ths(2,:),posteriors_nom(2,:,trial_index),'color',c_nom,'linewidth',lw)
plot(ths(2,:),posteriors_sgn(2,:,trial_index),'color',c_sgn,'linewidth',lw)
plot(ths(2,:),posteriors_lo(2,:,trial_index),'color',c_hmc,'linewidth',lw)
plot([A(2,2) A(2,2)],[0,10],'color',[0.5,0.5,0.5])
xlabel('$A_{22}$','interpreter','latex')
grid on
axis([0.1 1.1 0 8])

hf = fill([ths(2,:)],[posteriors_nom(2,:,trial_index)],'b');
set(hf,'facecolor',c_nom)
set(hf,'facealpha',transp);
set(hf,'edgealpha',0);

hf = fill([ths(2,:)],[posteriors_sgn(2,:,trial_index)],'b');
set(hf,'facecolor',c_sgn)
set(hf,'facealpha',transp);
set(hf,'edgealpha',0);  

hf = fill([ths(2,:)],[posteriors_lo(2,:,trial_index)],'b');
set(hf,'facecolor',c_hmc)
set(hf,'facealpha',transp);
set(hf,'edgealpha',0);      

set(gca,'fontsize',fs)
set(gca,'TickLabelInterpreter','latex')

legend({'nominal, $\tilde{u}$'},'interpreter','latex','fontsize',fs)
legend boxoff


%% compare hmc trajectories

% some level sets
levrad = [0.1:0.1:0.4];
circr = linspace(0,2*pi+0.01,50);
circ = [cos(circr);sin(circr)];

f1 = figure;
hold on
plot(nan,nan,'k.')
plot(nan,nan,'x','color',c_nom,'linewidth',2,'markersize',7)
plot(nan,nan,'^','color',c_hmc)
plot(theta_true.A(2,1),theta_true.A(2,2),'k+','linewidth',2,'markersize',7)
plot(nan,nan,'.','color',[0.5,0.5,0.5])

xlabel('$q_1=A_{21}$','interpreter','latex')
ylabel('$q_2=A_{22}$','interpreter','latex')

for ii = 1:length(levrad)
    plot(theta_true.A(2,1)+levrad(ii)*circ(1,:),theta_true.A(2,2)+levrad(ii)*circ(2,:),'color',[0.5,0.5,0.5,0.5])
end

plot(res_hmc.q(1,:),res_hmc.q(2,:),'.','color',[0.5,0.5,0.5,0.1])

% compute trajectories for sampled initial conditions given nominal input

ops = ops_hmc;
ops.q_nom = Gamma(param_select);

cost_1 = 0;
cost_2 = 0;

% recompute outputs for safety
[ys_nom,~] = sim_lgss(theta_true,zeros(nx,1),us_nom,w0,v0);

% for ii = 1:1:size(q_samples,2)
for ii = 1:1:15

    
    ops.u           = us_nom;
    ops.y           = ys_nom;    
    
    [f,df,q,p] = hmc_lgss_trajectory(ops,q_samples(:,ii),p_samples(:,ii));
    
    cost_1 = cost_1 + f;
   
    
    figure(f1)
    plot(q_samples(1,ii),q_samples(2,ii),'k.');
    plot([q_samples(1,ii),q(1,:)],[q_samples(2,ii),q(2,:)],'-.','color',[c_nom,0.5])
    plot(q(1,end),q(2,end),'x','color',c_nom,'linewidth',2,'markersize',8)
     
    ops.u           = u_lo; 
    ops.y           = y_lo;

    
    [f,df,q,p] = hmc_lgss_trajectory(ops,q_samples(:,ii),p_samples(:,ii));
    
    cost_2 = cost_2 + f;
      
    figure(f1)
    plot([q_samples(1,ii),q(1,:)],[q_samples(2,ii),q(2,:)],'color',[c_hmc,0.5])
    plot(q(1,end),q(2,end),'^','color',c_hmc)   
    
end

plot(theta_true.A(2,1),theta_true.A(2,2),'k+','linewidth',2,'markersize',10)

fs = 14;

figure(f1)
legend({'$q(0)$','$q(L)$, nominal','$q(L)$, optimized','$\theta^*$','HMC samples'},'interpreter','latex','fontsize',fs)
% title(['cost on samples (1): u1 = ' num2str(cost_1) ', u2 = ' num2str(cost_2)])

set(gca,'fontsize',fs)
set(gca,'TickLabelInterpreter','latex')

box on

axis equal

fprintf('\n2-norm from nominal state, i.e. |q(L)-q*|\n')
fprintf('nominal input: %.3f\n',cost_1)
fprintf('optimized input: %.3f\n',cost_2)
    
    
    
    