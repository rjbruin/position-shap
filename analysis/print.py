def print_position_biases(overall_biases, class_biases, shape='scalar', attribution_method='input_gradient'):
    """
    Args:
        overall_biases (dict(dict(tuple(float, float)))): Dictionary of overall
            biases, indexed by target, then by bias type, given as a tuple of
            (mean, stddev), as returned by position_biases().
        class_biases (dict(dict(list(tuple(float, float))))): Dictionary of
            class-specific biases, indexed by target, then by bias type, given
            as a list of tuples of (mean, stddev), as returned by
            position_biases().
        shape (str): Shape of the input. Either 'scalar' or 'head'.
        attribution_method (str): Attribution method used to compute the
            position biases.
    """
    for target in overall_biases:
        target_name = 'predicted class' if target == 'pred_class' else target
        method_name = 'input-gradient' if attribution_method == 'input_gradient' else '<not specified>'
        print(f"--- Position biases w.r.t. {target_name} using {method_name} attribution ---")

        if shape == 'scalar':
            print("Mean over classes:")
            print(f"\tBias: {overall_biases[target]['bias'][0]:.2f} +- {overall_biases[target]['bias'][1]:.2f}")
            print(f"\tAppearance: {overall_biases[target]['appearance'][0]:.2f} +- {overall_biases[target]['appearance'][1]:.2f}")
            if 'position' in overall_biases[target]:
                print(f"\tPosition: {overall_biases[target]['position'][0]:.2f} +- {overall_biases[target]['position'][1]:.2f}")
                if 'learned_relative_position' in overall_biases[target]:
                    print(f"\tLearned relative position: {overall_biases[target]['learned_relative_position'][0]:.2f} +- {overall_biases[target]['learned_relative_position'][1]:.2f}")
            if 'relative_position' in overall_biases[target]:
                print(f"\tRelative position: {overall_biases[target]['relative_position'][0]:.2f} +- {overall_biases[target]['relative_position'][1]:.2f}")
            if class_biases is not None:
                for c in range(2):
                    print(f"Class {c}:")
                    print(f"\tBias: {class_biases[target]['bias'][c][0]:.2f} +- {class_biases[target]['bias'][c][1]:.2f}")
                    print(f"\tAppearance: {class_biases[target]['appearance'][c][0]:.2f} +- {class_biases[target]['appearance'][c][1]:.2f}")
                    if 'position' in class_biases[target]:
                        print(f"\tPosition: {class_biases[target]['position'][c][0]:.2f} +- {class_biases[target]['position'][c][1]:.2f}")
                        if 'learned_relative_position' in class_biases[target]:
                            print(f"\tLearned relative position: {class_biases[target]['learned_relative_position'][c][0]:.2f} +- {class_biases[target]['learned_relative_position'][c][1]:.2f}")
                    if 'relative_position' in class_biases[target]:
                        print(f"\tRelative position: {class_biases[target]['relative_position'][c][0]:.2f} +- {class_biases[target]['relative_position'][c][1]:.2f}")

        elif shape == 'head':
            print("Mean over classes:")
            print(f"\tBias: " + ", ".join([f"{a[0]:.2f} +- {a[1]:.2f}"  for a in overall_biases[target]['bias']]))
            print(f"\tAppearance: " + ", ".join([f"{a[0]:.2f} +- {a[1]:.2f}"  for a in overall_biases[target]['appearance']]))
            if 'position' in overall_biases[target]:
                print(f"\tPosition: " + ", ".join([f"{a[0]:.2f} +- {a[1]:.2f}"  for a in overall_biases[target]['position']]))
                if 'learned_relative_position' in overall_biases[target]:
                    print(f"\tLearned relative position: " + ", ".join([f"{a[0]:.2f} +- {a[1]:.2f}"  for a in overall_biases[target]['learned_relative_position']]))
            if 'relative_position' in overall_biases[target]:
                print(f"\tRelative position: " + ", ".join([f"{a[0]:.2f} +- {a[1]:.2f}"  for a in overall_biases[target]['relative_position']]))
            if class_biases is not None:
                for c in range(2):
                    print(f"Class {c}:")
                    print(f"\tBias: " + ", ".join([f"{a[0]:.2f} +- {a[1]:.2f}"  for a in class_biases[target]['bias'][c]]))
                    print(f"\tAppearance: " + ", ".join([f"{a[0]:.2f} +- {a[1]:.2f}"  for a in class_biases[target]['appearance'][c]]))
                    if 'position' in class_biases[target]:
                        print(f"\tPosition: " + ", ".join([f"{a[0]:.2f} +- {a[1]:.2f}"  for a in class_biases[target]['position'][c]]))
                        if 'learned_relative_position' in class_biases[target]:
                            print(f"\tLearned relative position: " + ", ".join([f"{a[0]:.2f} +- {a[1]:.2f}"  for a in class_biases[target]['learned_relative_position'][c]]))
                    if 'relative_position' in class_biases[target]:
                        print(f"\tRelative position: " + ", ".join([f"{a[0]:.2f} +- {a[1]:.2f}"  for a in class_biases[target]['relative_position'][c]]))

        else:
            raise NotImplementedError()