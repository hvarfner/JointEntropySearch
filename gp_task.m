function success = run_exp(path, seed, approach, dim)
    % add necessary paths
    deploy;

    % Define function
    dx = dim;
    xmin = zeros(dx,1);
    xmax = ones(dx,1);

    % Creting the path to the relevant experiment
    exp_path = fullfile(fullfile(pwd, 'experiments'), path);
   
    % The seed matters due to the GP index to run
    f = python_pipe(seed, exp_path, '', approach);

    % Set the GP hyper-parameters if you would like to fix them.
    if strcmp(path, 'gp/gp_2dim.py')
        options.l = ones(1,dx)* 1 ./ (0.1.^2);
    elseif strcmp(path, 'gp/gp_4dim.py')
        options.l = ones(1,dx)* 1 ./ (0.2.^2); 
    elseif strcmp(path, 'gp/gp_6dim.py')
        options.l = ones(1,dx)* 1 ./ (0.3.^2);
    elseif strcmp(path, 'gp/gp_12dim.py')
        options.l = ones(1,dx)* 1 ./ (0.6.^2);
    else
        return
    end
    
    options.sigma = 10;
    options.sigma0 = 0.01;
    options.n_init = dim + 1
    options.bo_method = approach
    options.seed = seed
    options.normalize_y = 0
    % Start BO
    % Set the number of GP hyper-parameter settings to be sampled
    % options.nM = 10;
    n_iters = dim * 50
    
    disp('Running GPOpt')
    gpopt(f, xmin, xmax, n_iters, [], [], options);
end
