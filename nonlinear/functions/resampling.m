function a = resampling(w,N)
% Sample N from weights w

% Multinomail resampling
U = generateSortedUniforms(N);
a = zeros(N,1);
s = w(1);
m = 1;
for n = 1:N
	while s < U(n)
		m = m + 1;
		s = s + w(m);
	end
	a(n) = m;
end
