function [y,x] = sim_lgss(theta,x1,u,w,v)

    A = theta.A;
    B = theta.B;
    C = theta.C;
    D = theta.D;

    [ny,nx] = size(theta.C);
    T = size(u,2);

    x = zeros(nx,T);
    y = zeros(ny,T);

    % Note: now we have state x1
    xt = x1;

    for t = 1:T

        x(:,t) = xt; 
        y(:,t) = C*x(:,t) + v(:,t);
        xt = A*xt + B*u(:,t) + w(:,t);

    end

    
end