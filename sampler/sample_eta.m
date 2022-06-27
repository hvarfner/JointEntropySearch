function lntheta_eta = sample_eta(hypsamples, x_es,y_es, Nsamples,mean_ln_yminob_minus_eta,var_ln_yminob_minus_eta, ess)
    % data and the prior for eta (as well as the sampler we want to use)
    d=size(x_es,2);
    burnin = 300;

    Nhypersets = size(hypsamples, 1);
    % define the log distri from which we sample 
    % log_like_fn  = @(x)loglike(x, x_es, y_es,mean_ln_yminob_minus_eta);
    lntheta_eta = zeros(Nhypersets, Nsamples);
    for i = 1:Nhypersets
        log_like_fn = @(x)logposter_pos_eta(x, log(hypsamples(i, :)), x_es,y_es,mean_ln_yminob_minus_eta,var_ln_yminob_minus_eta);
    
        % initial guess
        % should be the same as before
        
        % should be 1 sample
        ln_theta0 = [mean_ln_yminob_minus_eta];

        % should retrieve 10 samples
        ln_eta = slicesample(ln_theta0, Nsamples,'logpdf',log_like_fn, 'thin',2,'burnin',burnin);
        lntheta_eta(i, :) = ln_eta;
    end
    
    


 