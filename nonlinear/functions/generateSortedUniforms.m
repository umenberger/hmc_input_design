function U = generateSortedUniforms(N)
E = exprnd(1,N+1,1);
C = cumsum(E);
U = C(1:N)/C(N+1);
