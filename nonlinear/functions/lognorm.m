function [out] = lognorm(x,mu,sigma)
% logarithm of scalar normal distribution
% NOTE: sigma is the VARIANCE not the standard deviation

    out = -0.5 .* log(2 * pi) - 0.5 .* log(sigma) - 0.5 ./ sigma .* (x - mu).^2;


end

