function [y,x] = sim_flex_nl(m,theta,x1,u,w,v)

T = size(u,2);

x = zeros(1,T);
y = zeros(1,T);

xt = x1;

for t = 1:T
    
    y(:,t) = m.g(xt,u(:,t),theta) + v(:,t);
    x(:,t) = xt;
    xt = m.f(xt,u(:,t),theta) + w(:,t);
    
end


end