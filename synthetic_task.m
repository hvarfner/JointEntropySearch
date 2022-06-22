function success = synthetic_task(path, seed, approach, dim, gam)
    % add necessary paths
    deploy;

    % Define function
    dx = dim;
    xmin = zeros(dx,1);
    xmax = ones(dx,1);
    % Creting the path to the relevant experiment

    exp_path = fullfile(fullfile(pwd, 'experiments'), path);
    % here, we just give the path to the experiment and the seed to run
    % gamma - gamma-greedy parameter
    f = python_pipe(seed, exp_path, append('gam', num2str(gam)), approach);

    % Save the file to a path


    options.n_init = dim + 1;
    options.bo_method = approach;
    options.seed = seed;
    options.nFeatures = 500;
    options.normalize = 1;
    options.epsilon = gam

    n_iters = dim * 50
    
    % Start BO
    gpopt(f, xmin, xmax, n_iters, [], [], options);
   
end
