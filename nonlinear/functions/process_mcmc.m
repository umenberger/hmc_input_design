function res = process_mcmc(ops)
% Function to extract some relevant stuff from mcmc results

%% Process inputs

mc = ops.markovChain;

noIterations = length(mc);

if isfield(ops,'noBurnInIterations')
    noBurnInIterations = ops.noBurnInIterations;
else
    noBurnInIterations = 0;
end

if isfield(ops,'noBurnInIterations')
    noBins = ops.noBins;
else
    noBins = 0;
end

if isfield(ops,'normalization')
    normalization = ops.normalization;
else
    normalization = 'pdf';
end

if isfield(ops,'title')
    plottitle = ops.title;
else
    plottitle = 'who knows?';
end

if isfield(ops,'subplots')
    numsp = ops.subplots;
else
    numsp = 3;
end

if isfield(ops,'suppressPlots')
    noplots = ops.suppressPlots;
else
    noplots = 0;
end



%% Process results

% noBins = floor(sqrt(noIterations - noBurnInIterations));
% noBins = 50;
iterange = noBurnInIterations:noIterations;
phi_pmh_pbi = mc(noBurnInIterations:noIterations);

fhandle = figure;

% Plot the parameter posterior estimate (solid black line = posterior mean)
subplot(numsp, 1, 1);
% hg = histogram(phi_pmh_pbi, noBins);
hg = histogram(phi_pmh_pbi, noBins,'Normalization',normalization);
xlabel('\theta=A'); 
ylabel('p(\theta|y)');

% h = findobj(gca, 'Type', 'patch');
set(hg, 'FaceColor', [117 112 179] / 256, 'EdgeColor', 'w');

hold on;
plot([1 1] * mean(phi_pmh_pbi), [0 max(hg.Values)], 'LineWidth', 2,'color','b');
hold off;

title(plottitle)

% Plot the trace of the Markov chain after burn-in  (solid black line = posterior mean)
subplot(numsp, 1, 2);
plot(iterange, phi_pmh_pbi, 'Color', [117 112 179] / 256, 'LineWidth', 1);
xlabel('iteration'); 
ylabel('\theta');

hold on;
plot([iterange(1) iterange(end)], [1 1] * mean(phi_pmh_pbi), 'k', 'LineWidth', 3);
hold off;

% Plot ACF of the Markov chain after burn-in
subplot(numsp, 1, 3);
[acf, lags] = xcorr(phi_pmh_pbi - mean(phi_pmh_pbi), 100, 'coeff');
stem(lags(101:200), acf(101:200), 'Color', [117 112 179] / 256, 'LineWidth', 2);
xlabel('lag'); 
ylabel('ACF of \theta');

if noplots
    close(fhandle)
end

%% Process results

res.binCount = hg.Values;
res.binEdges = hg.BinEdges;
res.fighandle = fhandle;

end

