function python_objective = python_pipe(seed, path, n_samples, approach)

    cmd = ["python3" path append(lower(approach), append("", string(n_samples))) append("run_", string(seed))];
    python_objective = @(x) call_python(cmd, x);

end

