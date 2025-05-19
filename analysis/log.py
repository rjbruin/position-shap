def log_position_biases(experiment, biases, shape='scalar'):
    if shape != 'scalar':
        raise NotImplementedError(f'Only scalar shape supported. Shape given: {shape}')

    if len(biases.keys()) > 1:
        print(f'WARNING! Bias logging supports only one target. Targets given: {biases.keys()}')

    biases = biases[list(biases.keys())[0]]

    if 'bias' in biases:
        experiment.log('sensitivity/bias/mean', biases['bias'][0])
        experiment.log('sensitivity/bias/std', biases['bias'][1])
    if 'position' in biases:
        experiment.log('sensitivity/position/mean', biases['position'][0])
        experiment.log('sensitivity/position/std', biases['position'][1])
    if 'appearance' in biases:
        experiment.log('sensitivity/appearance/mean', biases['appearance'][0])
        experiment.log('sensitivity/appearance/std', biases['appearance'][1])
    if 'relative_position' in biases:
        experiment.log('sensitivity/relative_position/mean', biases['relative_position'][0])
        experiment.log('sensitivity/relative_position/std', biases['relative_position'][1])
    if 'learned_relative_position' in biases:
        experiment.log('sensitivity/learned_relative_position/mean', biases['learned_relative_position'][0])
        experiment.log('sensitivity/learned_relative_position/std', biases['learned_relative_position'][1])