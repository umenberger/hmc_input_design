%% innovations form

theta(1) = -0.5; theta(2) = 1; theta(3) = 1; theta(4) = 1; theta(5) = 0.1;
theta(6) = 0.1; theta(7) = 0.1; theta(8) = 0.01;

f = @(x,u,th) th(1)*(x.*(x-1).*(x+1)) + th(2)*u; 
g = @(x,u,th) th(3)*x + th(4)*x.^2 + th(5)*x.^3;

a = -theta(1);
b = theta(2);
c = theta(3);
d = 0;

% lin = ss(a,b,c.d.1);
% k = kalman(lin,

k = 1;

%% simulate state-space rep

T = 50;

v = randn(T,1);
u = randn(T,1);

xs = zeros(1,T);
ys = zeros(1,T);

for t = 1:T    
    xs(:,t+1) = a*xs(:,t) + b*u(t) + k*v(t);
    ys(:,t) = c*xs(:,t) + v(t);
end


%% transfer function rep

G = tf(c*b,[1,-a],1);
H = tf([1, k-a],[1, -a],1);

g = impulse(G,T);
h = impulse(H,T);

GT = toeplitz(g(1:T),0*g(1:T));
HT = toeplitz(h(1:T),0*h(1:T));

y = GT*u + HT*v;

figure
plot(1:T,ys)
hold on
plot(1:T,y,'r--')

% yes, they seem to agree

%% derivatives for input design

% note: Ian's method assumes indepedently parametrized G and H.

% p = [a]

sF1 = tf(-1,[1,a1,a2],1); % b
sF2 = tf([-b,0],[1,2*a1,2*a2+a1^2,2*a1*a2,a2^2],1); % a1
sF3 = tf([-b],[1,2*a1,2*a2+a1^2,2*a1*a2,a2^2],1); % a2

f1 = impulse(sF1);
F1 = toeplitz(f1(1:T),0*f1(1:T));

f2 = impulse(sF2);
F2 = toeplitz(f2(1:T),0*f2(1:T));

f3 = impulse(sF3);
F3 = toeplitz(f3(1:T),0*f3(1:T));


















