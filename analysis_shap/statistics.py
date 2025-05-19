import pandas as pd
from torch import Tensor
from scipy.stats import mannwhitneyu
import numpy as np

# Settings
alpha = 0.05


def pe_discriminated_test(df, col='pos_emb_all'):
    """
    Given a Pandas DataFrame describing the attribution values over all samples
    in the dataset, compute the one-sided Mann-Whitney U-test for the
    alternative hypothesis that the values in `col` are higher for the PE group
    than for the non-PE group, where the PE group is given by `uses-PE`.
    """

    # Get the indices of the two groups
    pe_inds = df['uses-PE']
    non_pe_inds = ~pe_inds

    if len(df.loc[pe_inds]) == 0 or len(df.loc[non_pe_inds]) == 0:
        pass

    # Get the values for the two groups
    pe_values = df.loc[pe_inds, col]
    non_pe_values = df.loc[non_pe_inds, col]

    # Perform the test
    _, p = mannwhitneyu(pe_values, non_pe_values, alternative='greater')
    return p

def values_separated_by_pe(df, col='pos_emb_all'):
    # Get the indices of the two groups
    pe_inds = df['uses-PE']
    non_pe_inds = ~pe_inds

    if len(df.loc[pe_inds]) == 0 or len(df.loc[non_pe_inds]) == 0:
        pass

    # Get the values for the two groups
    pe_values = df.loc[pe_inds, col]
    non_pe_values = df.loc[non_pe_inds, col]

    return np.mean(pe_values), np.std(pe_values), np.mean(non_pe_values), np.std(non_pe_values)

def remove_bias_class(df, total_attribution_col='all'):
    class_total_shap_max = df.groupby('label')[total_attribution_col].mean()
    bias_class = class_total_shap_max.idxmin()
    return df[df['label'] != bias_class], bias_class

def stats(shap_df, has_pe_labels=False):
    # Remove from each dataframe the samples with the lowest total_shap_sum for being bias class
    shap_df_filtered, shap_bias_class = remove_bias_class(shap_df, 'total_shap_sum')

    # Compute mean and std SHAP values
    mean_unnormed, std_unnormed = shap_df['pe_shap_sum'].mean(), shap_df['pe_shap_sum'].std()
    mean_unnormed_filtered, std_unnormed_filtered = shap_df_filtered['pe_shap_sum'].mean(), shap_df_filtered['pe_shap_sum'].std()
    mean_normed, std_normed = shap_df['pe_sum_frac'].mean(), shap_df['pe_sum_frac'].std()
    mean_normed_filtered, std_normed_filtered = shap_df_filtered['pe_sum_frac'].mean(), shap_df_filtered['pe_sum_frac'].std()

    # Compute mean and std SHAP values only for correct samples
    correct_df = shap_df[shap_df['predicted'] == shap_df['label']]
    if len(correct_df) > 0:
        correct_df_filtered, correct_bias_class = remove_bias_class(correct_df, 'total_shap_sum')
        correct_mean_unnormed, correct_std_unnormed = correct_df['pe_shap_sum'].mean(), correct_df['pe_shap_sum'].std()
        correct_mean_unnormed_filtered, correct_std_unnormed_filtered = correct_df_filtered['pe_shap_sum'].mean(), correct_df_filtered['pe_shap_sum'].std()
        correct_mean_normed, correct_std_normed = correct_df['pe_sum_frac'].mean(), correct_df['pe_sum_frac'].std()
        correct_mean_normed_filtered, correct_std_normed_filtered = correct_df_filtered['pe_sum_frac'].mean(), correct_df_filtered['pe_sum_frac'].std()
    else:
        correct_mean_unnormed, correct_std_unnormed = 0, 0
        correct_mean_unnormed_filtered, correct_std_unnormed_filtered = 0, 0
        correct_mean_normed, correct_std_normed = 0, 0
        correct_mean_normed_filtered, correct_std_normed_filtered = 0, 0
        correct_bias_class = -1

    stats = {
        'misc/bias_class': shap_bias_class,
        'misc/bias_class_correct': correct_bias_class,
        'values/mean': mean_unnormed,
        'values/std': std_unnormed,
        'values/mean_filtered': mean_unnormed_filtered,
        'values/std_filtered': std_unnormed_filtered,
        'values/mean_normed': mean_normed,
        'values/std_normed': std_normed,
        'values/mean_normed_filtered': mean_normed_filtered,
        'values/std_normed_filtered': std_normed_filtered,
        'values/mean_correct': correct_mean_unnormed,
        'values/std_correct': correct_std_unnormed,
        'values/mean_correct_filtered': correct_mean_unnormed_filtered,
        'values/std_correct_filtered': correct_std_unnormed_filtered,
        'values/mean_normed_correct': correct_mean_normed,
        'values/std_normed_correct': correct_std_normed,
        'values/mean_normed_correct_filtered': correct_mean_normed_filtered,
        'values/std_normed_correct_filtered': correct_std_normed_filtered,
    }

    if has_pe_labels:
        # Compute one-sided Mann-Whitney U-test for SHAP values
        shap_p_unnormed = pe_discriminated_test(shap_df, 'pe_shap_sum')
        shap_p_normed = pe_discriminated_test(shap_df, 'pe_sum_frac')
        shap_p_unnormed_filtered = pe_discriminated_test(shap_df_filtered, 'pe_shap_sum')
        shap_p_normed_filtered = pe_discriminated_test(shap_df_filtered, 'pe_sum_frac')

        # Compute mean and std SHAP values for PE and non-PE samples
        shap_mean_pe_unnormed, shap_std_pe_unnormed, shap_mean_non_pe_unnormed, shap_std_non_pe_unnormed = values_separated_by_pe(shap_df, 'pe_shap_sum')
        shap_mean_pe_normed, shap_std_pe_normed, shap_mean_non_pe_normed, shap_std_non_pe_normed = values_separated_by_pe(shap_df, 'pe_sum_frac')
        shap_mean_pe_unnormed_filtered, shap_std_pe_unnormed_filtered, shap_mean_non_pe_unnormed_filtered, shap_std_non_pe_unnormed_filtered = values_separated_by_pe(shap_df_filtered, 'pe_shap_sum')
        shap_mean_pe_normed_filtered, shap_std_pe_normed_filtered, shap_mean_non_pe_normed_filtered, shap_std_non_pe_normed_filtered = values_separated_by_pe(shap_df_filtered, 'pe_sum_frac')

        if len(correct_df) > 0:
            # Compute one-sided Mann-Whitney U-test for SHAP values only for correct samples
            shap_p_unnormed_correct = pe_discriminated_test(correct_df, 'pe_shap_sum')
            shap_p_normed_correct = pe_discriminated_test(correct_df, 'pe_sum_frac')
            shap_p_unnormed_correct_filtered = pe_discriminated_test(correct_df_filtered, 'pe_shap_sum')
            shap_p_normed_correct_filtered = pe_discriminated_test(correct_df_filtered, 'pe_sum_frac')

            # Compute mean and std SHAP values for PE and non-PE samples only for correct samples
            correct_shap_mean_pe_unnormed, correct_shap_std_pe_unnormed, correct_shap_mean_non_pe_unnormed, correct_shap_std_non_pe_unnormed = values_separated_by_pe(correct_df, 'pe_shap_sum')
            correct_shap_mean_pe_normed, correct_shap_std_pe_normed, correct_shap_mean_non_pe_normed, correct_shap_std_non_pe_normed = values_separated_by_pe(correct_df, 'pe_sum_frac')
            correct_shap_mean_pe_unnormed_filtered, correct_shap_std_pe_unnormed_filtered, correct_shap_mean_non_pe_unnormed_filtered, correct_shap_std_non_pe_unnormed_filtered = values_separated_by_pe(correct_df_filtered, 'pe_shap_sum')
            correct_shap_mean_pe_normed_filtered, correct_shap_std_pe_normed_filtered, correct_shap_mean_non_pe_normed_filtered, correct_shap_std_non_pe_normed_filtered = values_separated_by_pe(correct_df_filtered, 'pe_sum_frac')
        else:
            shap_p_unnormed_correct = None
            shap_p_normed_correct = None
            shap_p_unnormed_correct_filtered = None
            shap_p_normed_correct_filtered = None
            correct_shap_mean_pe_unnormed, correct_shap_std_pe_unnormed = 0, 0
            correct_shap_mean_non_pe_unnormed, correct_shap_std_non_pe_unnormed = 0, 0
            correct_shap_mean_pe_normed, correct_shap_std_pe_normed = 0, 0
            correct_shap_mean_non_pe_normed, correct_shap_std_non_pe_normed = 0, 0
            correct_shap_mean_pe_unnormed_filtered, correct_shap_std_pe_unnormed_filtered = 0, 0
            correct_shap_mean_non_pe_unnormed_filtered, correct_shap_std_non_pe_unnormed_filtered = 0, 0
            correct_shap_mean_pe_normed_filtered, correct_shap_std_pe_normed_filtered = 0, 0
            correct_shap_mean_non_pe_normed_filtered, correct_shap_std_non_pe_normed_filtered = 0, 0

        stats.update({
            'p-values/p_unnormed': shap_p_unnormed,
            'p-values/p_normed': shap_p_normed,
            'p-values/p_unnormed_filtered': shap_p_unnormed_filtered,
            'p-values/p_normed_filtered': shap_p_normed_filtered,

            'values/mean_pe_unnormed': shap_mean_pe_unnormed,
            'values/std_pe_unnormed': shap_std_pe_unnormed,
            'values/mean_non_pe_unnormed': shap_mean_non_pe_unnormed,
            'values/std_non_pe_unnormed': shap_std_non_pe_unnormed,
            'values/mean_pe_normed': shap_mean_pe_normed,
            'values/std_pe_normed': shap_std_pe_normed,
            'values/mean_non_pe_normed': shap_mean_non_pe_normed,
            'values/std_non_pe_normed': shap_std_non_pe_normed,
            'values/mean_pe_unnormed_filtered': shap_mean_pe_unnormed_filtered,
            'values/std_pe_unnormed_filtered': shap_std_pe_unnormed_filtered,
            'values/mean_non_pe_unnormed_filtered': shap_mean_non_pe_unnormed_filtered,
            'values/std_non_pe_unnormed_filtered': shap_std_non_pe_unnormed_filtered,
            'values/mean_pe_normed_filtered': shap_mean_pe_normed_filtered,
            'values/std_pe_normed_filtered': shap_std_pe_normed_filtered,
            'values/mean_non_pe_normed_filtered': shap_mean_non_pe_normed_filtered,
            'values/std_non_pe_normed_filtered': shap_std_non_pe_normed_filtered,

            'p-values/p_unnormed_correct': shap_p_unnormed_correct,
            'p-values/p_normed_correct': shap_p_normed_correct,
            'p-values/p_unnormed_correct_filtered': shap_p_unnormed_correct_filtered,
            'p-values/p_normed_correct_filtered': shap_p_normed_correct_filtered,

            'values/mean_pe_unnormed_correct': correct_shap_mean_pe_unnormed,
            'values/std_pe_unnormed_correct': correct_shap_std_pe_unnormed,
            'values/mean_non_pe_unnormed_correct': correct_shap_mean_non_pe_unnormed,
            'values/std_non_pe_unnormed_correct': correct_shap_std_non_pe_unnormed,
            'values/mean_pe_normed_correct': correct_shap_mean_pe_normed,
            'values/std_pe_normed_correct': correct_shap_std_pe_normed,
            'values/mean_non_pe_normed_correct': correct_shap_mean_non_pe_normed,
            'values/std_non_pe_normed_correct': correct_shap_std_non_pe_normed,
            'values/mean_pe_unnormed_correct_filtered': correct_shap_mean_pe_unnormed_filtered,
            'values/std_pe_unnormed_correct_filtered': correct_shap_std_pe_unnormed_filtered,
            'values/mean_non_pe_unnormed_correct_filtered': correct_shap_mean_non_pe_unnormed_filtered,
            'values/std_non_pe_unnormed_correct_filtered': correct_shap_std_non_pe_unnormed_filtered,
            'values/mean_pe_normed_correct_filtered': correct_shap_mean_pe_normed_filtered,
            'values/std_pe_normed_correct_filtered': correct_shap_std_pe_normed_filtered,
            'values/mean_non_pe_normed_correct_filtered': correct_shap_mean_non_pe_normed_filtered,
            'values/std_non_pe_normed_correct_filtered': correct_shap_std_non_pe_normed_filtered,
        })

    return stats