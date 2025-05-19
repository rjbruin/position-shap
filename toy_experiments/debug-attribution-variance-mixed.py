"""
Usage:
    debug-attribution-variance-mixed-cluster.py [<setting_first> <setting_last>] [options]

Options:
    --dry-settings  Do not run the settings, just print the number of settings
    --dry-run       Compute up to training, then exit.
    -h --help       Show this screen
"""
import docopt
args = docopt.docopt(__doc__)

from run_sens_and_shap import run_sens_and_shap, Setting

# %% [markdown]
# # Debugging the inconsistency in saliency-based attribution for PEs using Full-Gradient
#
# Models trained with a different seed show varying magnitudes of saliency-based attribution for the position embeddings, even in a toy setting. This notebook explores this problem by computing not only saliency-based attribution (using Full-Gradient) but also perturbation-based(?) attribution using KernelSHAP. We run toy experiments, analyzing models with both methods, which both return attribution values for all inputs and are therefore directly comparable.
#
# ## Usage
#
# Run any of the implemented toy settings using the code below. The "Toy setting" chapter implements the following settings:
#
# 1) Two-class position only
# 2) Four-class mixed position and appearance
# 3) Four-class mixed position and appearance, with test set and patch size 2
#
# The "Run" chapter executes the experiment by training models over a given range of hyperparameters and analyzing the learned models using the Full-Gradient method and the KernelSHAP method, both extended to support attribution of the position embeddings (PE). The experiment will save a scatterplot PDF to a named subdirectory of `debug-attribution-variance-plots` (which needs to be created before the experiment is run).

# %% [markdown]
# # Toy setting
#
# Saves a scatterplot image to `debug-attribution-variance-plots` for each setting.

# %% [markdown]
# ## Datasets
#
# Run any of the three dataset cells below to use that dataset in the toy setting.

# %% [markdown]
# ### Two-class position only

# # %%
# import datasets

# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.style.use(['seaborn'])
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Palatino"]})
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

# images, labels, n_classes = datasets.dataset_absolute_position()
# print(f"{n_classes} classes")
# train_images, train_labels, test_images, test_labels, analysis_images, analysis_labels = datasets.split_dataset(images, labels)

# size = 6
# patch_size = 1
# dataset_settings = Setting(size=size, patch_size=patch_size, n_classes=n_classes)

# fig, axs = plt.subplots(1, n_classes * 2, figsize=(1 + n_classes * 2, 2), dpi=120)
# j = 0
# for c in range(n_classes):
#     for i in range(2):
#         inds = train_labels == c
#         axs[j].imshow(train_images[inds][i].permute((1, 2, 0)))
#         axs[j].set_title(f"Class {train_labels[inds][i]}")
#         axs[j].axis('off')
#         j += 1

# plt.tight_layout()
# pass

# %% [markdown]
# ### Two-class appearance only

# %%
# import datasets

# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.style.use(['seaborn'])
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Palatino"]})
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

# images, labels, n_classes = datasets.dataset_appearance(size=[6,6])
# print(f"{n_classes} classes")
# train_images, train_labels, test_images, test_labels, analysis_images, analysis_labels = datasets.split_dataset(images, labels)

# size = 6
# patch_size = 1
# dataset_settings = Setting(size=size, patch_size=patch_size, n_classes=n_classes)

# fig, axs = plt.subplots(1, n_classes * 2, figsize=(1 + n_classes * 2, 2), dpi=120)
# j = 0
# for c in range(n_classes):
#     for i in range(2):
#         inds = train_labels == c
#         axs[j].imshow(train_images[inds][i].permute((1, 2, 0)))
#         axs[j].set_title(f"Class {train_labels[inds][i]}")
#         axs[j].axis('off')
#         j += 1

# plt.tight_layout()
# pass

# %% [markdown]
# ### Four-class mixed position and appearance

# # %%
# import datasets

# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.style.use(['seaborn'])
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Palatino"]})
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

# images, labels, n_classes = datasets.dataset_appearance_absolute_position_three_colors()
# print(f"{n_classes} classes")
# train_images, train_labels, test_images, test_labels, analysis_images, analysis_labels = datasets.split_dataset(images, labels)

# size = 6
# patch_size = 1
# dataset_settings = Setting(size=size, patch_size=patch_size, n_classes=n_classes)

# fig, axs = plt.subplots(1, n_classes * 2, figsize=(1 + n_classes * 2, 2), dpi=120)
# j = 0
# for c in range(n_classes):
#     for i in range(2):
#         inds = train_labels == c
#         axs[j].imshow(train_images[inds][i].permute((1, 2, 0)))
#         axs[j].set_title(f"Class {train_labels[inds][i]}")
#         axs[j].axis('off')
#         j += 1

# plt.tight_layout()
# pass

# %% [markdown]
# ### Four-class mixed position and appearance with test set and patch size 2

# %%
import datasets

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

train_images, train_labels, test_images, test_labels, n_classes = datasets.dataset_appearance_absolute_position_three_colors(size=[8,8], split_quadrants=True)
print(f"{n_classes} classes")

size = 8
patch_size = 2
dataset_settings = Setting(size=size, patch_size=patch_size, n_classes=n_classes)

fig, axs = plt.subplots(2, n_classes * 2, figsize=(1 + n_classes * 2, 2), dpi=120)
j = 0
for c in range(n_classes):
    for i in range(2):
        inds = train_labels == c
        axs[0,j].imshow(train_images[inds][i].permute((1, 2, 0)))
        axs[0,j].set_title(f"Train class {train_labels[inds][i]}")
        axs[0,j].axis('off')

        inds = test_labels == c
        axs[1,j].imshow(test_images[inds][i].permute((1, 2, 0)))
        axs[1,j].set_title(f"Test class {test_labels[inds][i]}")
        axs[1,j].axis('off')

        j += 1

plt.tight_layout()
pass

# %% [markdown]
# ## Run

# %%
import seaborn as sns

base_settings = {
    'n_epochs': 500,
    'report_every_n': 100,
    'pos_emb': 'absolute',
    'use_rel_pos': False,
    'shap_bg_folds': 'all',
    'postprocessing': 'abs',
}

# Run 01
# run_name = "run01"
# debug = False
# lrs = [4e-2, 4e-3, 4e-4]
# weight_decays = [1e-3, 1e-4]
# ds = [8, 4]
# ns_heads = [4, 2]
# pos_emb_factors = [0.1, 0.01]
# postprocessing = 'abs'
# pos_add = 'add'

# Run 02: `WD=[1e-3, 1e-4], LR=[4e-4], d/n_heads=[16/8, 12/6, 8/4, 4/2], PEx=[0.02, 0.01, 0.005]`
# run_name = "run02"
# debug = False
# lrs = [4e-4]
# weight_decays = [1e-3, 1e-4]
# # ds = [12, 8]
# # ns_heads = [6, 4]
# ds = [16]
# ns_heads = [8]
# pos_emb_factors = [0.02, 0.01, 0.005]
# postprocessing = 'abs'
# pos_add = 'add'

# Run 03: `WD=[1e-3, 1e-4], LR=[4e-4], d/n_heads=[16/8, 16/4, 16/2], PEx=[0.02]`
# run_name = "run03"
# debug = False
# lrs = [4e-4]
# weight_decays = [1e-3, 1e-4]
# # ds = [16]
# # ns_heads = [8]
# ds = [16, 16]
# ns_heads = [4, 2]
# pos_emb_factors = [0.02]
# postprocessing = 'abs'
# pos_add = 'add'

# Run 04: more seeds
# run_name = "run04"
# debug = False
# lrs = [4e-4]
# weight_decays = [1e-4]
# ds = [16]
# ns_heads = [8]
# pos_emb_factors = [0.02]
# seeds = range(10)
# postprocessing = 'abs'
# pos_add = 'add'

# Run 05: without postprocessing
# run_name = "run05"
# debug = False
# lrs = [4e-4]
# weight_decays = [1e-4]
# ds = [16]
# ns_heads = [8]
# pos_emb_factors = [0.02]
# seeds = range(3)
# postprocessing = 'none'
# pos_add = 'add'

# Run 06: no postprocessing, wider search
# run_name = "run06"
# debug = False
# lrs = [4e-2, 4e-3, 4e-4]
# weight_decays = [1e-3, 1e-4]
# ds = [16, 16, 4]
# ns_heads = [8, 4, 2]
# pos_emb_factors = [0.1, 0.02]
# seeds = range(3)
# postprocessing = 'none'
# pos_add = 'add'

# Run 07: post-processing "abs", pos_add concat_equald, wider search
# run_name = "run07"
# debug = False
# lrs = [4e-2, 4e-3, 4e-4]
# weight_decays = [1e-3, 1e-4]
# ds = [16, 16, 4]
# ns_heads = [8, 4, 2]
# pos_emb_factors = [0.1, 0.02]
# seeds = range(3)
# postprocessing = 'abs'
# pos_add = 'concat_equald'

# Run 08: best settings from run 07, with more seeds
# run_name = "run08"
# debug = False
# lrs = [4e-3]
# weight_decays = [1e-3]
# ds = [16]
# ns_heads = [4]
# pos_emb_factors = [0.1, 0.02]
# seeds = range(10)
# postprocessing = 'abs'
# pos_add = 'concat_equald'

# Run 09: wide search, post-processing "max" with pos_add concat_equald
# run_name = "run09"
# debug = False
# lrs = [4e-2, 4e-3, 4e-4]
# weight_decays = [1e-3, 1e-4]
# ds = [16]
# ns_heads = [4]
# pos_emb_factors = [0.1, 0.02]
# seeds = range(3)
# postprocessing = 'max'
# pos_add = 'concat_equald'

# Run 10: wide search on new setting with test set
# run_name = "run10"
# debug = False
# lrs = [4e-2, 4e-3, 4e-4]
# weight_decays = [1e-3, 1e-4]
# ds = [16, 16]
# ns_heads = [8, 4]
# pos_emb_factors = [0.1, 0.02]
# seeds = range(5)
# postprocessings = ['abs', 'none', 'max']
# pos_adds = ['add', 'concat_equald']

# Run 12: narrowed search (based on run 11 results) on mixed+test
# with FG and SHAP
# run_name = "run12-sens+shap-mixed+test"
# lrs = [4e-3]
# weight_decays = [1e-4]
# # ds = [4, 8]
# # ns_heads = [2, 4]
# # ds = [4, 8, 8]
# # ns_heads = [2, 4, 2]
# ds = [16, 16, 32]
# ns_heads = [4, 8, 4]
# pos_emb_factors = [0.1, 0.02]
# seeds = range(5)
# postprocessings = ['abs']
# pos_adds = ['add', 'concat_equald']
# shap_bg_folds = 'all'

# Run 15: wide search on mixed toy with test set, run on cluster
run_name = 'run15-sens+shap-mixed+test-fixsave2'
save_plots = True
# len(settings) = 30
settings = [
    Setting(lr=4e-3, weight_decay=1e-3, d=4,  n_heads=2, pos_emb_factor=0.1,  pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=4,  n_heads=2, pos_emb_factor=0.1,  pos_add='add',           seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=8,  n_heads=4, pos_emb_factor=0.1,  pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=8,  n_heads=4, pos_emb_factor=0.1,  pos_add='add',           seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=16, n_heads=4, pos_emb_factor=0.1,  pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=16, n_heads=4, pos_emb_factor=0.1,  pos_add='add',           seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=24, n_heads=6, pos_emb_factor=0.1,  pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=24, n_heads=6, pos_emb_factor=0.1,  pos_add='add',           seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=24, n_heads=4, pos_emb_factor=0.1,  pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=24, n_heads=4, pos_emb_factor=0.1,  pos_add='add',           seeds=range(5), **base_settings),

    Setting(lr=4e-3, weight_decay=1e-3, d=4,  n_heads=2, pos_emb_factor=0.02, pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=4,  n_heads=2, pos_emb_factor=0.02, pos_add='add',           seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=8,  n_heads=4, pos_emb_factor=0.02, pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=8,  n_heads=4, pos_emb_factor=0.02, pos_add='add',           seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=16, n_heads=4, pos_emb_factor=0.02, pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=16, n_heads=4, pos_emb_factor=0.02, pos_add='add',           seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=24, n_heads=6, pos_emb_factor=0.02, pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=24, n_heads=6, pos_emb_factor=0.02, pos_add='add',           seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=24, n_heads=4, pos_emb_factor=0.02, pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-3, weight_decay=1e-3, d=24, n_heads=4, pos_emb_factor=0.02, pos_add='add',           seeds=range(5), **base_settings),

    Setting(lr=4e-2, weight_decay=1e-4, d=4,  n_heads=2, pos_emb_factor=0.1,  pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-2, weight_decay=1e-4, d=4,  n_heads=2, pos_emb_factor=0.1,  pos_add='add',           seeds=range(5), **base_settings),
    Setting(lr=4e-2, weight_decay=1e-4, d=8,  n_heads=4, pos_emb_factor=0.1,  pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-2, weight_decay=1e-4, d=8,  n_heads=4, pos_emb_factor=0.1,  pos_add='add',           seeds=range(5), **base_settings),
    Setting(lr=4e-2, weight_decay=1e-4, d=16, n_heads=4, pos_emb_factor=0.1,  pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-2, weight_decay=1e-4, d=16, n_heads=4, pos_emb_factor=0.1,  pos_add='add',           seeds=range(5), **base_settings),
    Setting(lr=4e-2, weight_decay=1e-4, d=24, n_heads=6, pos_emb_factor=0.1,  pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-2, weight_decay=1e-4, d=24, n_heads=6, pos_emb_factor=0.1,  pos_add='add',           seeds=range(5), **base_settings),
    Setting(lr=4e-2, weight_decay=1e-4, d=24, n_heads=4, pos_emb_factor=0.1,  pos_add='concat_equald', seeds=range(5), **base_settings),
    Setting(lr=4e-2, weight_decay=1e-4, d=24, n_heads=4, pos_emb_factor=0.1,  pos_add='add',           seeds=range(5), **base_settings),
]

# DEBUG
# run_name = 'debug'
# save_plots = True
# del base_settings['postprocessing']
# del base_settings['shap_bg_folds']
# settings = [
#     Setting(lr=4e-3, weight_decay=4e-3, d=4, n_heads=2, pos_emb_factor=0.1,
#             postprocessing='abs', pos_add='concat_equald', seeds=[0],
#             shap_bg_folds=1, **base_settings),
#     Setting(lr=4e-3, weight_decay=4e-3, d=4, n_heads=2, pos_emb_factor=0.1,
#             postprocessing='abs', pos_add='concat_equald', seeds=[1],
#             shap_bg_folds=1, **base_settings),
# ]


#
# RUN
#

if args['--dry-settings']:
    print(f"Number of settings: {len(settings)}")
    exit()

settings_start = 0
if args['<setting_first>'] is not None:
    settings_start = int(args['<setting_first>'])
settings_end = len(settings)
if args['<setting_last>'] is not None:
    settings_end = int(args['<setting_last>'])


run_sens_and_shap(run_name, settings, dataset_settings,
                  train_images, train_labels,
                  test_images, test_labels,
                  save_plots=save_plots,
                  settings_start=settings_start, settings_end=settings_end,
                  dry=args['--dry-run'])