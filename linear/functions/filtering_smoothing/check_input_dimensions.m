function theta = check_input_dimensions(theta,u,y)

[A,B,C,D,E,G,Q,R,m1,P1] = unpackTheta(theta);


% Probably my fault for not explaining properly, but I don't think we want
% to be doing this...

% if norm(size(E)-size(A))>1e-10 || rank(E) < size(E,1)
%     error('E is of wrong dimension or not full rank')
% end
% 
% if norm(E-eye(size(A))) > 1e-10
%     A = E\A;
%     B = E\B;
%     G = E\G;
%     E = eye(size(A));
% end

% Check that sizes of matrices fit
nx = size(A,1);
if size(B,1) ~= nx
    error('B must have same number of rows as A')
elseif size(C,2) ~= nx
    error('C must have same number of columns as A')
elseif size(G,1) ~= nx
    error('N must have same number of rows as A')
elseif size(m1,1) ~= nx
    error('m1 must have same number of rows as A')
elseif mean(size(P1)) ~= nx
    error('P1 must have dimensions as A')
end

ny = size(C,1);
if size(D,1) ~= ny
    error('D must have same number of columns as C')
elseif mean(size(R)) ~= ny
    error('R must have same number of rows and columns as C has rows')
end

nw = size(Q,1);
if size(G,2) ~= nw
    error('N must have same number of columns as Q')
end

nu = size(B,2);
if size(D,2) ~= nu
    error('D must have same number of columns as B')
end

if nargin > 1
    if size(u,1) ~= nu
        error('u must have the same number of rows as B and D has columns')
    end
end
if nargin > 2
    if size(y,1) ~= ny
        error('y must have the same number of rows av C and R')
    end
    if size(y,2) ~= size(u,2)
        error('y and u meust have the same number of columns')
    end
end

theta = packTheta(A,B,C,D,E,G,Q,R,m1,P1);
end