function [truncvar, d_truncvar] = truncnormvar(upper_loc, mu, variance)
if nargout == 2

    b = (upper_loc' - mu) ./ sqrt(variance)
    % if it is this low, it is very, very unlikely to occur (10 standard devs away)
    b = min(max(b, -10), 10)
    density_b = normpdf(b)

    % Z cannot be zero, so we will have to have some lower bound here (probably on b)
    Z = normcdf(b)
    relative_variance_reduction = b .* density_b ./ Z + power(density_beta ./ Z, 2)
    truncvar = variance .* (1 - relative_variance_reduction)

else
    b = (upper_loc' - mu) ./ sqrt(variance);
    % if it is this low, it is very, very unlikely to occur (10 standard devs away)
    b = min(max(b, -10), 10);
    
    density_b = normpdf(b);

    % Z cannot be zero, so we will have to have some lower bound here (probably on b)
    Z = normcdf(b);
    relative_variance_reduction = b .* density_b ./ Z + power(density_b ./ Z, 2);
    truncvar = variance .* (1 - relative_variance_reduction);

end
