import argparse
import json
import itertools
import copy
import pandas as pd
import torch

import experiments
import datasets


def main():
    parser = argparse.ArgumentParser()

    # Add arguments for:
    parser.add_argument('--setting', type=str, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, nargs='+', default=[4e-3])
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--mixed_input_d', type=int, default=7)
    parser.add_argument('--n_heads', type=int, required=True)
    parser.add_argument('--pos_emb', type=str, default='absolute')
    parser.add_argument('--use_rel_pos', action='store_true')
    parser.add_argument('--report_every_n', type=int, default=250)
    parser.add_argument('--attribution_method', type=str, default='input_gradient_withnegative')
    parser.add_argument('--aggregate_fn', type=str, default='sum')
    parser.add_argument('--target', type=str, default='loss_all_classes')
    parser.add_argument('--pos_add', type=str, default='add')
    parser.add_argument('--pos_emb_factor', type=float, nargs='+', default=[0.1])
    parser.add_argument('--pos_emb_init', type=str, default='uniform')
    parser.add_argument('--weight_decay', type=float, nargs='+', default=[0.])
    parser.add_argument('--pos_emb_weight_decay', type=float, default=None)
    parser.add_argument('--position_mix', type=str, nargs='+', default=['6/0'])

    args = parser.parse_args()

    if args.setting in ['appearance', 'absolute_position', 'relative_position']:
        dataset_fn = {
            'appearance': datasets.dataset_appearance,
            'absolute_position': datasets.dataset_absolute_position,
            'relative_position': datasets.dataset_relative_position,
        }[args.setting]
        images, labels, n_classes = dataset_fn()
        train_images, train_labels, test_images, test_labels, analysis_images, analysis_labels = datasets.split_appearance(images, labels)
        dataset_args = {
            'n_classes': n_classes,
            'train_images': train_images,
            'train_labels': train_labels,
            'test_images': test_images,
            'test_labels': test_labels,
            'analysis_images': analysis_images,
            'analysis_labels': analysis_labels,
        }

    elif args.setting == 'mixed_position':
        dataset_args = {}
        for mix in args.position_mix:
            n_mixed, n_appearance = [int(x) for x in mix.split('/')]
            images, labels, n_classes, sort_groups = datasets.dataset_mixed_position(n_mixed // 2, n_appearance, input_d=args.mixed_input_d)
            train_images, train_labels, test_images, test_labels, analysis_images, analysis_labels = datasets.split_appearance(images, labels)
            dataset_args[mix] = {
                'n_classes': n_classes,
                'train_images': train_images,
                'train_labels': train_labels,
                'test_images': test_images,
                'test_labels': test_labels,
                'analysis_images': analysis_images,
                'analysis_labels': analysis_labels,
                'sort_groups': sort_groups,
            }

    else:
        raise ValueError(f"Unknown setting: {args.setting}")

    del args.mixed_input_d

    # Check for multiple values
    narg_args = ['lr', 'pos_emb_factor', 'weight_decay', 'position_mix']
    values = [getattr(args, key) for key in narg_args]
    perms = list(itertools.product(*values))

    print(f"Running {len(perms)} instances")

    results = []
    for i, perm in enumerate(perms):
        perm_dict = dict(zip(narg_args, perm))
        print(f"({i+1}/{len(perms)}) Running instance: {perm_dict}")
        perm_args = copy.deepcopy(vars(args))
        perm_args.update(perm_dict)
        print(json.dumps(perm_args, indent=4))

        setting = perm_args['setting']
        del perm_args['setting']

        # Get dataset args, which for "mixed_position" depends on the position mix
        perm_dataset_args = dataset_args
        if setting == 'mixed_position':
            perm_dataset_args = dataset_args[perm_args['position_mix']]
        del perm_args['position_mix']

        _, accs, mean_biases, std_biases = experiments.run(setting, **perm_dataset_args, **perm_args)

        result = {
            'mean_acc': torch.mean(torch.stack(accs)).item(),
            'mean_appearance': mean_biases['appearance'].item(),
            'std_appearance': std_biases['appearance'].item(),
        }
        if 'position' in mean_biases:
            result['mean_position'] = mean_biases['position'].item()
            result['std_position'] = std_biases['position'].item()
        else:
            result['mean_position'] = 0.
            result['std_position'] = 0.
        if 'relative_position' in mean_biases:
            result['mean_relative_position'] = mean_biases['relative_position'].item()
            result['std_relative_position'] = std_biases['relative_position'].item()
        else:
            result['mean_relative_position'] = 0.
            result['std_relative_position'] = 0.

        result.update(perm_dict)
        results.append(result)

    print('\n\n')
    df = pd.DataFrame(results)
    print(df.to_csv())

if __name__ == '__main__':
    main()