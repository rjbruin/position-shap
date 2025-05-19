import os
import pickle
import time
import pandas as pd
import tqdm
import numpy as np

from experiments import run
from attribution_analysis import attribution_stats, sens_scatterplots, shap_scatterplots

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn'])
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Palatino"]})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


# Settings to run
class Setting():
    def __init__(self, **parameters):
        for key, val in parameters.items():
            setattr(self, key, val)
        self.parameter_names = parameters.keys()

    def settings_dict(self):
        return {k: getattr(self, k) for k in self.parameter_names}

    def setting_name(self, include=['weight_decay', 'lr', 'd/n_heads', 'n_blocks', 'pos_emb_factor', 'postprocessing', 'pos_add']):
        name = []
        setting_names = {'weight_decay': 'WD', 'lr': 'LR', 'n_blocks': 'n_blocks', 'pos_emb_factor': 'PEx', 'postprocessing': 'post', 'pos_add': 'PE+'}
        for key in include:
            if key == 'd/n_heads':
                if hasattr(self, 'd') and hasattr(self, 'n_heads'):
                    name.append(f"dnheads_{getattr(self, 'd')}+{getattr(self, 'n_heads')}")
                else:
                    raise ValueError("d/n_heads is to be included in setting name, but setting does not have both 'd' and 'n_heads' attributes.")
            else:
                name.append(f"{setting_names[key]}_{getattr(self, key)}")
        return "-".join(name)


def run_sens_and_shap(run_name, settings, dataset_settings,
                      train_images, train_labels,
                      test_images, test_labels,
                      save_plots=True, settings_start=None, settings_end=None,
                      dry=False,
                      do_shap=True, do_sens=True, get_tokens=True):
    """
    Train toy models, then do sensitivity and SHAP analysis.

    This method plots the PE attributions of FG (for sensitivity) and KernelSHAP
    (for SHAP) methods for all samples (normalized by the respective total
    attribution for the sample) in one scatterplot per trained model (seed).
    Plots are saved to `debug-attribution-variance-plots/{run_name}`.

    It also saves these values and more to pickles in
    `debug-attribution-variance-results/{run_name}` for more extensive analysis
    elsewhere.

    Arguments:
        - run_name: str, name of the run, used for saving plots and results;
        - settings: list of Setting objects, each object contains parameters for
          a single run;
        - dataset_settings: Setting object, contains settings for the dataset;
        - train_images, train_labels, test_images, test_labels: torch.Tensor,
          data for training and testing;
        - save_plots: bool, whether to save plots;
        - settings_start, settings_end: int, optional start and end indices of
          settings to run from the settings list;
        - dry: bool, if True, only print the settings and exit;
        - do_shap: bool or str, whether to run SHAP analysis. If str, one of
          'kernel' or 'deep' or None;
        - do_sens: bool, whether to run sensitivity analysis;
        - get_tokens: bool, whether to pickle tokens from the model.
    """
    total_settings = len(settings)
    if settings_end is None:
        settings_end = len(settings)
    if settings_start is None:
        settings_start = 0
    settings = settings[settings_start:settings_end]
    if isinstance(do_shap, bool):
        do_shap = 'kernel' if do_shap else None
        print(f"do_shap is bool, setting to {do_shap}")

    print(f"Running {len(settings)}/{total_settings} settings ([{settings_start}:{settings_end}])...")

    for si, setting in tqdm.tqdm(enumerate(settings), desc='settings', total=len(settings), leave=False):
        setting_start = time.time()

        seeds = setting.seeds
        n_epochs = setting.n_epochs
        weight_decay = setting.weight_decay
        lr = setting.lr
        d = setting.d
        n_heads = setting.n_heads
        if hasattr(setting, 'n_blocks'):
            n_blocks = setting.n_blocks
        else:
            n_blocks = 1
        pos_emb_factor = setting.pos_emb_factor
        postprocessing = setting.postprocessing
        pos_add = setting.pos_add
        pos_emb = setting.pos_emb
        use_rel_pos = setting.use_rel_pos
        report_every_n = setting.report_every_n
        shap_bg_folds = setting.shap_bg_folds

        size = dataset_settings.size
        patch_size = dataset_settings.patch_size
        n_classes = dataset_settings.n_classes

        # Backwards compatibility with runs before run 20
        if not hasattr(dataset_settings, 'colors'):
            dataset_settings.colors = ['red', 'green', 'blue']
            dataset_settings.pos_colors = ['red']

        # run() will save raw attributions to a separate pickle if save_raw is
        # set to a filename. For run_sens_and_shap, we only want to use the
        # sensitivity analysis implemented in run() with this pickle-saving, not
        # with it returning the attributions, so we set save_raw.
        # If do_sens is False, we do not set save_raw.
        # Either way, run() will not return the attributions, even if we do
        # sensitivity analysis.
        save_raw = f"{run_name}_{setting.setting_name()}" if do_sens else None

        #
        # TRAIN, COMPUTE PREREQUISITES FOR ANALYSIS
        #

        if dry:
            print("Dry run; exiting...")
            return

        print(f"\n\nWeight decay: {weight_decay} / LR: {lr} / d: {d} / n_heads: {n_heads} / n_blocks: {n_blocks} / PE factor: {pos_emb_factor} / Post-processing: {postprocessing} / pos_add: {pos_add}")
        rets = \
            run('absolute_position', seeds, n_epochs, lr, d, n_heads,
                n_classes, pos_emb, use_rel_pos, train_images, train_labels,
                test_images, test_labels, test_images, test_labels,
                size=size,
                report_every_n=report_every_n, weight_decay=weight_decay,
                pos_emb_factor=pos_emb_factor, pos_add=pos_add,
                patch_size=patch_size, save_raw=save_raw,
                sensitivity_analysis=do_sens, do_shap=do_shap,
                shap_bg_folds=shap_bg_folds, notebook=False, cleanup_models=True,
                get_tokens=get_tokens, n_blocks=n_blocks)

        accs = rets['accs']
        if get_tokens:
            im_tokens = rets['image_tokens']
            pe_tokens = rets['pe_tokens']
            tok_labels = rets['token_labels']
        if do_shap:
            image_shap_values = rets['image_shap_values']
            pe_shap_values = rets['pe_shap_values']

        #
        # ANALYSIS settings
        #

        # Sensitivity settings
        # batched = whether sensitivity values are spread over multiple pickles
        # when saved in run()
        batched = False
        # Normalize saliency maps by the maximum over all maps or by itself
        rank_by = 'pos_emb'
        sens_rows = 1

        # SHAP settings
        plot_pe_shap_sum = False
        plot_pe_sum_frac = True
        plot_label_sum_frac = False
        shap_rows = 1

        # Dataset settings
        label_names = []
        assert len(dataset_settings.pos_colors) <= 1, "Only one position color is supported"
        if len(dataset_settings.pos_colors) == 1:
            assert dataset_settings.pos_colors[0] == 'red', "Only red is supported as position color"
        pos_labels = range(len(dataset_settings.pos_colors) * 2)
        for c in dataset_settings.colors:
            color_letter = c[0].upper()
            if c in dataset_settings.pos_colors:
                label_names.append(f"{color_letter}/L")
                label_names.append(f"{color_letter}/R")
            else:
                label_names.append(f"{color_letter}")
        print(label_names, pos_labels)

        #
        # ANALYSIS AND PLOTTING SETUP
        #

        rows = sens_rows + shap_rows
        fig, axs = plt.subplots(rows, len(seeds), figsize=(4 * len(seeds), 4 * rows), dpi=120)
        if len(seeds) == 1:
            axs = np.expand_dims(axs, 1)
        plot_name = f"{setting.setting_name()}-seeds_{'_'.join(map(str, seeds))}"
        # If the task failed to reach 100% accuracy, mark
        # this in the name and prevent the seed from being
        # marked as a success
        seeds_fail = []
        for i, acc in enumerate(accs):
            if acc < 1.0:
                plot_name += f"-S{i}ACC{acc:.3f}"
                seeds_fail.append(i)
        fig.suptitle(f"PE vs. non-PE attribution / {plot_name}")

        # We will store all results to be pickled here
        setting_analysis_results = [{} for _ in seeds]
        for i, seed in enumerate(seeds):
            setting_analysis_results[i]['seed'] = seed
            setting_analysis_results[i]['settings'] = setting.settings_dict()
            setting_analysis_results[i]['setting_name'] = setting.setting_name()
            setting_analysis_results[i]['test_acc'] = accs[i]

        #
        # FG
        #

        if do_sens:
            indexes = []
            for i, seed in enumerate(seeds):
                filepath = f"./toy_saliency_maps/{save_raw}_{seed}.pt"
                index = attribution_stats(filepath, batched, rank_by=rank_by, postprocessing=postprocessing, notebook=False)
                indexes.append(index)

            setting_analysis_results = sens_scatterplots(setting_analysis_results, indexes, seeds, accs, axs, label_names=label_names, pos_labels=pos_labels)

        #
        # SHAP
        #

        if do_shap:
            shap_results = \
                shap_scatterplots(image_shap_values, pe_shap_values,
                                seeds, axs, n_classes, test_labels,
                                start_row=sens_rows,
                                plot_pe_shap_sum=plot_pe_shap_sum,
                                plot_pe_sum_frac=plot_pe_sum_frac,
                                plot_label_sum_frac=plot_label_sum_frac,
                                label_names=label_names, pos_labels=pos_labels)
            for i, seed in enumerate(seeds):
                setting_analysis_results[i].update(shap_results[i])

        #
        # TOKENS
        #

        if get_tokens:
            for i, seed in enumerate(seeds):
                setting_analysis_results[i]['im_tokens'] = im_tokens[i].detach().cpu().numpy()
                setting_analysis_results[i]['pe_tokens'] = pe_tokens[i].detach().cpu().numpy()
                setting_analysis_results[i]['tok_labels'] = tok_labels[i].detach().cpu().numpy()

        #
        # Finish plotting
        #

        # Put success/fail in the plot name
        for i, seed in enumerate(seeds):
            if do_sens and setting_analysis_results[i]['sens_success']:
                plot_name += f"-S{i}SE"
            if do_shap and setting_analysis_results[i]['shap_success']:
                plot_name += f"-S{i}SA"

        plt.tight_layout()
        if save_plots:
            if not os.path.exists(f"debug-attribution-variance-plots/{run_name}"):
                os.makedirs(f"debug-attribution-variance-plots/{run_name}")
            plt.savefig(f"debug-attribution-variance-plots/{run_name}/{plot_name}.pdf")
            plt.close()

        #
        # Save analysis_results to unique pickle
        #
        setting_time = time.time() - setting_start
        print(f"Setting took {setting_time:.2f}s")
        for i, seed in enumerate(seeds):
            setting_analysis_results[i]['setting_time'] = setting_time

        if not os.path.exists(f"debug-attribution-variance-results/{run_name}"):
            os.makedirs(f"debug-attribution-variance-results/{run_name}")

        setting_index = settings_start + si
        filename = f"analysis_results_{setting_index}"
        with open(f"debug-attribution-variance-results/{run_name}/{filename}.pkl", 'wb') as f:
            pickle.dump(setting_analysis_results, f)
