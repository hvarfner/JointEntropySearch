function python_res = call_python(command, x)
    python_cmd = join([command string(x)]);
    [success, res] = system(python_cmd);
    disp(success)
    disp(res)
    if success > 0
        error(append('Python function failed\n\n', res))
    end
        
    python_res = str2double(res)
end

