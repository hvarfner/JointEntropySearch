import numpy as np


def format_arguments(config_space, args):
    hparams = config_space.get_hyperparameters_dict()
    formatted_hparams = {}
    for param, param_values in hparams.items():
        formatted_hparams[param] = param_values.sequence[args[param]]
    return formatted_hparams

def rescale_arguments(args):
    width, batch_size, alpha, learning_rate_init = args
    
    # 3 values, nearest alternative
    #depth = np.round((depth + 1e-10) * 2.999999999 + 0.5)
    width = np.round(np.power(2, 4 + (width * 6)))
    batch_size = np.round(np.power(2, 2 + (batch_size * 6)))
    alpha = np.power(10, -8 + (alpha * 5))
    learning_rate_init = np.power(10, -5 + (learning_rate_init * 5))
    return \
    {
        'depth': 2,
        'width': int(width),
        'batch_size': int(batch_size),
        'alpha': alpha,
        'learning_rate_init': learning_rate_init
    }