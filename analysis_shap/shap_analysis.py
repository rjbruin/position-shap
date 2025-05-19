from functools import reduce
import os
import pickle
import time
import pandas as pd
import torch
import shap as shap_lib
import numpy as np
from shap.utils._legacy import DenseData

from analysis_shap.statistics import stats


class PositionSHAP():
    PRINT_TIMINGS_NONE = 0
    PRINT_TIMINGS_STATS = 1
    PRINT_TIMINGS_STATSANDBG = 2
    PRINT_TIMINGS_ALL = 3

    def __init__(self, experiment, method, model, sample_images, n_classes,
                 pos_labels=None, save_batches_to_pickles=False,
                 reduce_shap_samples=False, on_cpu=False, batch_size=None,
                 spatial_features=False, image_channels_features=False,
                 print_timings=2, n_batches=None, debug_mode=False,
                 exp_name=None, read_only=False):
        """
        Perform SHAP analysis for image and position features. This
        implementation uses a single set of samples as background samples for
        all batches, to save inference cost. Background samples are passed at
        initialization time, then analysis of each batch is done by calling
        `batch_shap()` with the foreground samples of that batch. Finally,
        calling `finalize_shap()` will return the SHAP values for all samples.

        Arguments:
            experiment: PyTorch Lightning experiment object to log SHAP analysis
                results;
            method: The SHAP method to use, either 'kernel' or 'deep';
            model: The model to explain, which needs to implement a SHAP model
                interface;
            sample_images: Samples only used to infer shapes;
            n_classes: Number of classes in the classification problem;
            pos_labels: List of positive labels to use for analysis. If `None`,
                no separation between position and non-position classes is made
                in SHAP post-processing;
            save_batches_to_pickles: Whether to save batches to pickles for
                later analysis. Enable when you are running parallel SHAP
                analysis jobs on disjoint subsets of batches;
            reduce_shap_samples: Use half the number of samples for SHAP
                analysis to save inference time;
            on_cpu: Whether to run the model on CPU. If True, a CUDA model will
                be moved to CPU before running the SHAP explainer;
            batch_size: The batch size to use for internal inference of
                synthetic samples through the model in the SHAP explainer. If
                None, the explainer will run all synthetic samples in one large
                batch. This may lead to CUDA OOM if the model is large and uses
                CUDA;
            spatial_features: If True, explanations will be in the spatial shape
                of the input features: (N, Cin, H, W, Cout) for image and (N, D,
                [Hp, Wp]/[P], Cout) for PE. If False, spatial dimensions will
                not be discriminated: (N, Cin, Cout) for image and (N, Cout) for
                PE;
            image_channels_features: If True, explanations will be given for
                each image channel separately, i.e. Cin will not be a singleton
                dimension in the outputs;
            print_timings: Whether to print timings of SHAP analysis. 0: no
                timings, 1: print statistics at beginning and end of analysis,
                2: print statistics at beginning, end and for every batch;
            n_batches: Number of batches expected to be analysed. If set,
                PositionSHAP can provide progress updates through progress();
            debug_mode: In debug mode, SHAP checks that the PE transformations
                it applies are valid;
            exp_name: Name of the experiment. If None, will use the name given
                in the arguments;
            read_only: If True, will only load existing pickled results and
                cannot generate new ones.
        """
        if not read_only and method not in ['kernel', 'deep']:
            raise ValueError("method must be 'kernel' or 'deep'")
        if spatial_features and image_channels_features:
            raise ValueError("Cannot have spatial_features and image_channels_features simultaneously")

        self.read_only = read_only
        self.experiment = experiment
        if not read_only:
            self.n_classes = n_classes
        self.pos_labels = pos_labels
        self.save_batches_to_pickles = save_batches_to_pickles
        self.reduce_shap_samples = reduce_shap_samples
        self.print_timings = print_timings
        self.n_batches = n_batches

        self.image_shaps = []
        self.pe_shaps = []
        self.indices = []
        self.predictions = []
        self.labels = []
        self.batch_times = []
        self.start_time = None
        self._background_set = False
        self.batch_idx = 0
        self.fold_idx = 0

        if exp_name is None:
            if self.experiment.args.exp_name is None:
                raise ValueError("Batched SHAP and --shap_save require --exp_name to be set.")
            exp_name = self.experiment.args.exp_name
            self.shap_dirname = exp_name.replace('/','-') + f"_seed{self.experiment.args.seed}"
        else:
            self.shap_dirname = exp_name
        self.shap_dir = os.path.join("analysis_shap/results", self.shap_dirname)
        if os.path.exists(self.shap_dir):
            print(f"Found existing SHAP pickles at {self.shap_dir}.")
        else:
            print(f"Created new directory for SHAP pickles at {self.shap_dir}.")
            os.makedirs(self.shap_dir, exist_ok=True)

        # Pre-load existing pickles from disk
        self.pickles = self._discover_pickles()

        if not read_only:
            # Get model methods for interfacing the SHAP method (either
            # Kernel or Deep) with Torch model (taking 4D Torch tensors)
            self.shap_class, self.shap_model, self.shap_explainer_input_fn, self.shap_explainer_kwargs, self.shap_output_fn, self.pos_embeddings = \
                model.get_shap_interface(experiment.args, method, sample_images, on_cpu=on_cpu, batch_size=batch_size, spatial_features=spatial_features, image_channels_features=image_channels_features, debug_mode=debug_mode)

    @staticmethod
    def from_pickles(exp_name):
        return PositionSHAP(
            None, None, None, None, None,
            pos_labels=None, save_batches_to_pickles=True,
            exp_name=exp_name, read_only=True,
        )

    @property
    def background_set(self):
        return self._background_set

    def set_background(self, images):
        """
        Set background samples for SHAP analysis. Can be called once or for every batch.

        Arguments:
            bg_images: Background samples to use for SHAP analysis;
        """
        # Set up SHAP explainer by passing background samples
        start_time = time.time()
        self.explainer = self.shap_class(self.shap_model, self.shap_explainer_input_fn(images, self.pos_embeddings, background=True), **self.shap_explainer_kwargs)
        end_time = time.time()
        if self.print_timings >= self.PRINT_TIMINGS_STATSANDBG:
            print(f"SHAP explainer setup took {end_time - start_time:.3f} seconds")

        self._background_set = True

    def batch_shap_and_bg(self, images, labels, batch_idx=None, indices=None, predictions=None, fold_size=None, force=False):
        """
        Perform SHAP analysis for a batch of images by using one half of the
        batch as background for the other, and vice versa.

        Arguments:
            images: The foreground samples to explain;
            labels: The labels of the samples;
            batch_idx: Batch number. If not set, will use the automatically
                internally tracked number;
            indices: The indices of the samples in the dataset. If None, the
                indices will be generated from the batch number, which is
                automatically tracked if not given.
            predictions: The predicted labels of the samples.
        """
        # Initialize timer for progress()
        if self.start_time is None:
            self.start_time = time.time()

        # If we pre-loaded existing pickles and this batch has been done before,
        # don't analyze it again
        if not force and batch_idx is not None and batch_idx in self.pickles:
            self.batch_idx += 1
            return

        # Make folds
        if fold_size is None:
            fold_size = images.shape[0] // 2
        folds = self._folds(images.shape[0], fold_size)

        if indices is None:
            if batch_idx is None:
                batch_idx = self.batch_idx
            indices = torch.arange(batch_idx * images.shape[0],
                                   (batch_idx + 1) * images.shape[0])

        # For each fold, use the other folds as background. Apply SHAP to each
        # fold then concatenate in original batch order.
        folds_image_shaps = []
        folds_pe_shaps = []
        folds_fg_indices = []
        folds_predictions = []
        for i, fold in enumerate(folds):
            # Use all other folds as BG
            fg_inds = fold
            bg_inds = torch.cat(folds[:i] + folds[i+1:])

            # elif isinstance(bg_folds, int):
            #     # Use next n folds as BG
            #     if bg_folds <= 0:
            #         raise ValueError("shap_bg_folds must be non-zero positive")
            #     elif bg_folds > len(folds) - 1:
            #         # Less than (folds - 1) because we should not use the
            #         # foreground fold in the background samples
            #         print(f"NOTE: shap_bg_folds must be at most (nr_folds - 1) = {len(folds) - 1} < {bg_folds}")
            #         print(f"Using all other folds as background")
            #         bg_inds = torch.cat(folds[:i] + folds[i+1:])
            #     else:
            #         bg_inds = []
            #         j = i + 1
            #         while j - (i + 1) < bg_folds:
            #             bg_inds.extend(folds[j % len(folds)])
            #             j += 1
            #         bg_inds = torch.tensor(bg_inds)
            # else:
            #     raise ValueError("shap_bg_folds must be 'all' or an integer")

            self.set_background(images[bg_inds])

            image_shaps, pe_shaps = self._shap_values(images[fg_inds],
                                                      labels[fg_inds],
                                                      indices=indices[fg_inds],
                                                      predictions=predictions[fg_inds] if predictions is not None else None,)

            folds_image_shaps.append(image_shaps)
            folds_pe_shaps.append(pe_shaps)
            folds_fg_indices.append(fg_inds)
            if predictions is not None:
                folds_predictions.append(predictions[fg_inds])

        # Concatenate folds, then reorder to original batch order
        fg_indices = torch.cat(folds_fg_indices, dim=0)
        image_shaps = torch.cat(folds_image_shaps, dim=0)[fg_indices]
        pe_shaps = torch.cat(folds_pe_shaps, dim=0)[fg_indices]
        if predictions is not None:
            predictions = torch.cat(folds_predictions, dim=0)[fg_indices]

        if self.save_batches_to_pickles:
            # Postprocess analysis
            batch_shap_df = shap_postprocess(image_shaps,
                                             pe_shaps,
                                             indices,
                                             self.experiment.args.num_classes,
                                             labels,
                                             pos_labels=self.pos_labels,
                                             predictions=predictions)
            self._save_batch(batch_shap_df, batch_idx=batch_idx)

        self.batch_idx += 1

    def batch_shap(self, images, labels, batch_idx=None, indices=None, predictions=None, force=False):
        """
        Perform SHAP analysis for a batch of images.

        Arguments:
            images: The foreground samples to explain;
            labels: The labels of the samples;
            batch_idx: Batch number. If not set, will use the automatically
                internally tracked number;
            indices: The indices of the samples in the dataset. If None, the
                indices will be generated from batch_idx. If batch_idx is not
                set, will use the automatically internally tracked number;
            predictions: The predicted labels of the samples.
        """
        # Initialize timer for progress()
        if self.start_time is None:
            self.start_time = time.time()

        # If we pre-loaded existing pickles and this batch has been done before,
        # don't analyze it again
        if not force and batch_idx is not None and batch_idx in self.pickles:
            self.batch_idx += 1
            return

        self._shap_values(images, labels, indices=indices, predictions=predictions, batch_idx=batch_idx)

        if self.save_batches_to_pickles:
            # Postprocess analysis
            batch_shap_df = shap_postprocess(self.image_shaps[-1],
                                             self.pe_shaps[-1],
                                             self.indices[-1],
                                             self.experiment.args.num_classes,
                                             self.labels[-1],
                                             pos_labels=self.pos_labels,
                                             predictions=predictions)
            self._save_batch(batch_shap_df, batch_idx=batch_idx)

        self.batch_idx += 1

    def _shap_values(self, images, labels, indices=None, predictions=None, batch_idx=None):
        if not self._background_set:
            raise ValueError("Background samples have not been set yet.")

        start_time = time.time()

        # Do SHAP analysis on foreground images
        shap_inputs = self.shap_explainer_input_fn(images, self.pos_embeddings)
        nsamples = shap_inputs.shape[1] // 2 if self.reduce_shap_samples else "auto"
        fold_shaps = torch.tensor(self.explainer.shap_values(shap_inputs, nsamples=nsamples, **self.shap_explainer_kwargs))
        image_shaps, pe_shaps = self.shap_output_fn(fold_shaps)
        self.image_shaps.append(image_shaps)
        self.pe_shaps.append(pe_shaps)
        self.labels.append(labels)
        if predictions is not None:
            self.predictions.append(predictions)

        end_time = time.time()
        shap_time = end_time - start_time
        if self.print_timings >= self.PRINT_TIMINGS_ALL:
            print(f"Batch of SHAP analysis took {shap_time:.3f} seconds")
        self.batch_times.append(shap_time)

        if indices is None:
            if batch_idx is None:
                batch_idx = self.batch_idx
            self.indices.append(torch.arange(batch_idx * images.shape[0],
                                             (batch_idx + 1) * images.shape[0]))
        else:
            self.indices.append(indices)

        return image_shaps, pe_shaps

    def _save_batch(self, batch_shap_df, batch_idx=None):
        if batch_idx is None:
            batch_idx = self.batch_idx

        pickle_data = {
            'experiment': self.experiment.args.exp_name,
            'batch': batch_idx,
            'batch_shap_df': batch_shap_df['shap_df']
        }

        # Store in pickle
        filename = f"batch_{batch_idx:05d}.pkl"
        filename = os.path.join(self.shap_dir, filename)
        with open(filename, 'wb') as f:
            pickle.dump(pickle_data, f)

    def _folds(self, n_samples, fold_size):
        # Make folds of randomly permuted samples for SHAP, so we can
        # analyse all samples
        inds = torch.randperm(n_samples)
        folds = []
        i = 0
        while i < n_samples:
            folds.append(inds[i:i+fold_size])
            i += fold_size

        return folds

    def finalize_shap(self, force=False):
        """
        Finalize SHAP analysis, log and store the results under
        `analysis_shap/results/<exp_name>`.

        Arguments:
            force: If True, will force the analysis to be finalized even if
                not all batches have been processed. This is useful when SHAP
                analysis was applied to subsets of batches in parallel jobs and
                their resulting SHAP values were saved to pickles.
        """
        if self.save_batches_to_pickles:
            return self._finalize_shap_from_pickles(force=force)
        else:
            return self._finalize_shap_from_cache()

    def _finalize_shap_from_cache(self):
        """
        Finalize SHAP analysis, log and store the results under
        `analysis_shap/results/<exp_name>`.

        This method should be used when SHAP
        analysis was applied to all batches in sequence (i.e. when setting
        `save_batches_to_pickles=False`).
        """
        image_shaps = torch.cat(self.image_shaps, dim=0)
        pe_shaps = torch.cat(self.pe_shaps, dim=0)
        sample_indices = torch.cat(self.indices, dim=0)
        test_labels = torch.cat(self.labels, dim=0)
        predictions = torch.cat(self.predictions, dim=0) if len(self.predictions) > 0 else None

        self.batch_times = torch.tensor(self.batch_times)
        self.total_time = self.batch_times.sum()
        self.mean_time = self.batch_times.mean()
        self.std_time = self.batch_times.std()
        if self.print_timings >= self.PRINT_TIMINGS_STATSANDBG:
            print(f"SHAP analysis took {self.total_time:.3f}s in total, {self.mean_time:.3f}s ± {self.std_time:.3f}s per batch")
        self.experiment.log('p-shap/time/total', self.total_time)
        self.experiment.log('p-shap/time/mean', self.mean_time)
        self.experiment.log('p-shap/time/std', self.std_time)

        results = shap_postprocess(image_shaps, pe_shaps, sample_indices,
                                   self.n_classes, test_labels,
                                   label_names=None,
                                   pos_labels=self.pos_labels,
                                   predictions=predictions)

        return self._test_and_save(results)

    def _discover_pickles(self):
        # Discover all available pickles; continue only if number
        # of pickles matches `self.trainer.num_val_batches[0]`
        directory = os.path.join("analysis_shap/results", self.shap_dirname)
        pickles = sorted([f for f in os.listdir(directory) if f.startswith('batch') and f.endswith('.pkl')])
        by_index = {int(x.split('_')[1].split('.')[0]): x for x in pickles}
        return by_index

    def _finalize_shap_from_pickles(self, force=False):
        """
        Finalize SHAP analysis, log and store the results under
        `analysis_shap/results/<exp_name>`.

        This method should be used when SHAP
        analysis was applied to subsets of batches in parallel jobs and their
        resulting SHAP values were saved to pickles (i.e. when setting
        `save_batches_to_pickles=True`).
        """
        pickle_files = self._discover_pickles()
        shap_dfs = []
        if force or len(pickle_files) == self.experiment.trainer.num_val_batches[0]:
            for filename in pickle_files.values():
                filename = os.path.join("analysis_shap/results", self.shap_dirname, filename)
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                shap_dfs.append(data['batch_shap_df'])
            results = {'shap_df': pd.concat(shap_dfs, ignore_index=True)}
        else:
            print(f"Batched SHAP analysis incomplete: {len(pickle_files)} pickles found, expected {self.experiment.trainer.num_val_batches[0]}.")
            return

        return self._test_and_save(results)

    def _test_and_save(self, shap_values):
        results = stats(shap_values['shap_df'], has_pe_labels=self.pos_labels is not None)
        print(f"SHAP analysis results:")
        for key, value in results.items():
            print(f"{key}: {value}")
            if not self.read_only:
                self.experiment.log(f'p-shap/{key}', value)

        if self.pos_labels is None:
            print(f"No position labels found for SHAP analysis -> saving pickles for later analysis.")

        return self._save(shap_values, results)

    def _save(self, shap_values, stats):
        pickle_data = {
            'shap_df': shap_values['shap_df'],
            'results': stats,
        }
        # Save SHAP analysis
        directory = os.path.join("analysis_shap/results", self.shap_dirname)
        if not os.path.exists("analysis_shap/results"):
            os.makedirs("analysis_shap/results", exist_ok=True)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, "full.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(pickle_data, f)

        print("SHAP analysis saved to", filename)

    def get_timings(self):
        if not hasattr(self, 'total_time'):
            raise ValueError("SHAP analysis has not been finalized yet")
        return self.total_time, self.mean_time, self.std_time

    def progress(self):
        """
        Provides progress information for SHAP analysis.

        Returns:
            percentage: Percentage of batches processed (between 0 and 100);
            hours_left: Estimated time left for SHAP analysis to complete in
                hours.
        """
        if self.n_batches is None:
            return 0.0, ""
        # self.batch_idx is incremented by one after each batch is processed
        percentage = self.batch_idx / self.n_batches * 100
        time_per_batch = (time.time() - self.start_time) / self.batch_idx
        seconds_left = time_per_batch * (self.n_batches - self.batch_idx)
        hours_left = seconds_left / 3600
        return percentage, hours_left


def shap_postprocess(image_shap_values, pe_shap_values, sample_indices, n_classes, test_labels, label_names=None, pos_labels=None, image_channels=False, predictions=None):
    """
    Takes SHAP values and postprocesses them into a DataFrame for further
    analysis.

    Arguments:
        image_shap_values: SHAP values for image features;
        pe_shap_values: SHAP values for position features;
        sample_indices: Indices of the samples in the dataset;
        n_classes: Number of classes in the classification problem;
        test_labels: Labels of the samples;
        label_names: Names of the classes. If given, the `label` column will be
            transformed from a class number to a class name;
        pos_labels: List of positive labels to use for analysis. If `None`,
            no separation between position and non-position classes is made
            in SHAP post-processing;
        image_channels: If `True`, will add columns for SHAP-based attribution
            of each individual image channel. This requires `image_shap_values`
            to contain such values;
        predicted_labels: Predicted labels of the samples.
    """
    results = {}

    # Detect whether SHAP values are given for spatial dimensions or not.
    spatial_outputs = image_shap_values.dim() == 5

    df = []
    for c in range(n_classes):
        class_mask = test_labels == c
        if spatial_outputs:
            class_shaps = image_shap_values[:,:,:,:,c]
        else:
            class_shaps = image_shap_values[:,:,c]
        class_shaps = class_shaps[class_mask]
        # class_shaps = (N_c, Cin/[1], H, W) or (N_c, Cin/[1])
        class_sample_indices = sample_indices[class_mask]
        class_sample_indices = class_sample_indices.cpu()
        class_predicted = predictions[class_mask] if predictions is not None else None

        # Automatically adapt to spatial or flat PE shapes
        pe_spatial = len(pe_shap_values.shape) == 5

        if spatial_outputs:
            if pe_spatial:
                class_pe_shaps = pe_shap_values[:,:,:,:,c]
            else:
                class_pe_shaps = pe_shap_values[:,:,:,c]
        else:
            class_pe_shaps = pe_shap_values[:,:,c]
        class_pe_shaps = class_pe_shaps[class_mask]
        # Sum over [H*W] or [P] or [1]
        class_pe_shaps = class_pe_shaps.sum(dim=1)
        # class_pe_shaps = (N_c, [H, W]/[P]) or (N_c)

        if spatial_outputs:
            image_shap_max = class_shaps.abs().max(dim=3)[0].max(dim=2)[0].max(dim=1)[0]
            image_shap_sum = class_shaps.abs().sum(dim=3).sum(dim=2).sum(dim=1)
            if image_channels:
                r_shap_max = class_shaps[:,0].abs().max(dim=2)[0].max(dim=1)[0]
                g_shap_max = class_shaps[:,1].abs().max(dim=2)[0].max(dim=1)[0]
                b_shap_max = class_shaps[:,2].abs().max(dim=2)[0].max(dim=1)[0]
            if pe_spatial:
                pe_shap_max = class_pe_shaps.abs().max(dim=2)[0].max(dim=1)[0]
                pe_shap_sum = class_pe_shaps.abs().sum(dim=2).sum(dim=1)
            else:
                pe_shap_max = class_pe_shaps.abs().max(dim=1)[0]
                pe_shap_sum = class_pe_shaps.abs().sum(dim=1)
        else:
            image_shap_max = class_shaps.abs().max(dim=1)[0]
            image_shap_sum = class_shaps.abs().sum(dim=1)
            if image_channels:
                r_shap_max = class_shaps[:,0].abs()
                g_shap_max = class_shaps[:,1].abs()
                b_shap_max = class_shaps[:,2].abs()
            pe_shap_max = class_pe_shaps.abs()
            pe_shap_sum = class_pe_shaps.abs()
        total_shap_max = torch.stack([image_shap_max, pe_shap_max], dim=1).max(dim=1)[0]
        total_shap_sum = image_shap_sum + pe_shap_sum

        label = torch.full((len(class_shaps),), c)

        # data = (N_c, ...)
        column_names = ['total_shap_max', 'total_shap_sum', 'image_shap_max', 'image_shap_sum', 'pe_shap_max', 'pe_shap_sum', 'label', 'val_index']
        column_values = [total_shap_max, total_shap_sum, image_shap_max, image_shap_sum, pe_shap_max, pe_shap_sum, label, class_sample_indices]
        if image_channels:
            column_names.extend(['r_shap_max', 'g_shap_max', 'b_shap_max'])
            column_values.extend([r_shap_max, g_shap_max, b_shap_max])
        if predictions is not None:
            column_names.append('predicted')
            column_values.append(class_predicted.cpu())
        data = torch.stack(column_values, dim=1)
        df.append(data)

    df = torch.cat(df, dim=0)
    df = pd.DataFrame(df.numpy(), columns=column_names)

    # Convert columns from tensor to float
    df['PE-label'] = df['label'].apply(lambda x: "left" if x == 0 else ("right" if x == 1 else "no"))
    df['pe_max_frac'] = df['pe_shap_max'] / df['total_shap_max']
    df['pe_sum_frac'] = df['pe_shap_sum'] / df['total_shap_sum']
    df['label'] = df['label'].apply(lambda x: int(x))
    if pos_labels is not None:
        df['uses-PE'] = df['label'].apply(lambda x: x in pos_labels)
    if label_names is not None:
        df['label'] = df['label'].apply(lambda x: label_names[x])

    results['shap_df'] = df

    return results

def model_agnostic_interface(model, pos_emb_attribute_name, input_images,
                             pos_emb_format='B P D',
                             input_format_spatial=True,
                             input_format_channels_first=True,
                             model_out_unpack_fn=None, on_cpu=False,
                             batch_size=None, spatial_features=False,
                             image_channels_features=False,
                             use_hooks=False,
                             pe_not_broadcastable=True,
                             debug_mode=False):
    """Returns various methods to implement an interface of this model to be
    compatible with the KernelSHAP implementation in the shap library, (almost
    entirely) agnostic to the underlying model. The only assumption made is that
    the model has a Parameter for a position embedding in the shape `(Np, D, Hp,
    Wp)`.

    KernelExplainer takes a model that takes NumPy inputs, takes background data
    in NumPy format, and outputs NumPy SHAP values.

    This interface also takes a position embedding as input, and this function
    returns the interface as well as the position embeddings to use as input,
    representing the model state at the time this function was called.

    This interface makes the model to run in "PE per sample" mode, where the PE
    tensor is set to the provided value for each sample in the input, instead of
    expanding a single value over all samples at inference time. This is
    necessary because KernelSHAP requires input features for each sample.

    Arguments:
        model: The model to explain, with a position embedding parameter.
        pos_emb_attribute_name: The name of the attribute of the model that
            holds the position embedding tensor. Possibly nested, e.g.
            `model.encoder.pos_embedding`.
        input_images: A 4D NumPy tensor of input images to the model, to be
            used to define the interface.
        pos_emb_format: Einstein-notation for the format of the PE tensor. Use
            `B` for batch dimension, `D1`, `D2` etc. for channel dimension(s),
            including any dimensions pertaining to heads, `H`, `W` for spatial
            dimensions, `P` for patch dimension, and `D` for embedding
            dimension. The letters should whitespace-separated;
        input_format_spatial: The format of the input images tensor as pertaining
            to the spatial dimensions. If True, the input images tensor has
            spatial dimensions (e.g. `(N, C, H, W)`). If False, the input images
            tensor has one dimension for all patches (e.g. `(N, C, P)`).
        input_format_channels_first: The format of the input images tensor as
            pertaining to the channel dimension. If True, the input images tensor
            has the channel dimension first (e.g. `(N, C, H, W)`). If False, the
            input images tensor has the channel dimension last (e.g. `(N, H, W,
            C)`).
        model_out_unpack_fn: A function that takes the output of the model
            and returns the class logits. If None, the model output is assumed
            to be the class logits.
        on_cpu: Whether to run the model on CPU. If True, a CUDA model will be
            moved to CPU before running the SHAP explainer.
        batch_size: The batch size to use for internal inference of synthetic
            samples through the model in the SHAP explainer. If None, the
            explainer will run all synthetic samples in one large batch. This
            may lead to CUDA OOM if the model is large and uses CUDA.
        spatial_features: If True, explanations will be in the spatial shape of
            the input features: (N, Cin, H, W, Cout) for image and (N, D, [Hp,
            Wp]/[P], Cout) for PE. If False, spatial dimensions will not be
            discriminated: (N, Cin, Cout) for image and (N, Cout) for PE.
        image_channels_features: If True, explanations will be given for each
            image channel separately;
        use_hooks: Using a hook is necessary for PE methods that are updated
            during the forward pass, e.g. ones that need to know the number of
            patches. Not using hooks is slightly faster, but will not give
            correct results for such methods. Note that using hooks requires the
            Module that contains the position embeddings to implement a method
            `forward_fixed_pos_emb()`;
        pe_not_broadcastable: Set to True if the position embedding of the model
            does not support adding a batch dimension.
        debug_mode: If True, will run checks to ensure the position embedding
            shape is correctly inferred.

    Returns:
        shap_class: The class of the SHAP explainer to use (e.g.
            `shap.KernelExplainer`) shap_model: A function that takes a 2D NumPy
            tensor of input features and a 2D NumPy tensor of PE tokens, and
            returns the model's output as a NumPy tensor.
        explainer_input: A function that takes a 4D Torch tensor of input
            features and a 4D Torch tensor of PE tokens, and returns a 2D NumPy
            tensor of input features and PE tokens, in a format compatible with
            `shap.KernelExplainer()`.
        explainer_kwargs: A dictionary of keyword arguments to pass to the
            `shap.KernelExplainer()` constructor.
        shap_output: A function that takes the output of
            `shap.KernelExplainer:shap_values()` as a NumPy tensor, and returns
            the SHAP values as 5D NumPy tensors for image and PE tokens,
            respectively.
        pos_embedding: The position embedding tensor to use as input to the
            SHAP explainer, representing the model state at the time this
            function was called.
    """
    def get_nested_attr(obj, attr_name):
        """Get a nested attribute of an object."""
        attr_names = attr_name.split('.')
        for attr_name in attr_names:
            if attr_name[-1] == ']':
                number = int(attr_name[attr_name.index('[')+1:attr_name.index(']')])
                attr_name = attr_name[:attr_name.index('[')]
                obj = getattr(obj, attr_name)[number]
            else:
                obj = getattr(obj, attr_name)
        return obj


    # Run model on a single sample to set position embedding
    model(input_images[0:1])

    # Get all position embeddings: (module, module_name, attribute_name)
    # model.transformer.encoder.layer[x].attn.rel_pos.emb_h.rel_pos_embed
    pos_embeddings = []
    if not isinstance(pos_emb_attribute_name, list):
        pos_emb_attribute_name = [pos_emb_attribute_name]
    for name in pos_emb_attribute_name:
        if '[x]' in name:
            modules, suffix = name.split('[x]')
            modules = get_nested_attr(model, modules)
            for module in modules:
                no_dot_suffix = suffix[1:]
                pe = get_nested_attr(module, no_dot_suffix)
                words = no_dot_suffix.split(".")
                if len(words) > 1:
                    pe_module = get_nested_attr(module, ".".join(words[:-1]))
                else:
                    pe_module = module
                pos_embeddings.append((pe, pe_module))
        elif '[' in name and ']' in name:
            number = int(name[name.index('[')+1:name.index(']')])
            modules, suffix = name.split('[' + str(number) + ']')
            module = get_nested_attr(model, modules + f"[{number}]")
            no_dot_suffix = suffix[1:]
            pe = get_nested_attr(module, no_dot_suffix)
            words = no_dot_suffix.split(".")
            if len(words) > 1:
                pe_module = get_nested_attr(module, ".".join(words[:-1]))
            else:
                pe_module = module
            pos_embeddings.append((pe, pe_module))
        else:
            pe = get_nested_attr(model, name)
            words = name.split(".")
            if len(words) > 1:
                pe_module = get_nested_attr(model, ".".join(words[:-1]))
            else:
                pe_module = model
            pos_embeddings.append((pe, pe_module))

    x_shape = input_images.shape
    # NOTE: though input_images are given to define the interface, use of
    # the interface may be done with varying number of samples, so we cannot
    # use the predetermined shapes to set the number of samples in the
    # interface.
    if input_format_spatial:
        if input_format_channels_first:
            _, C, H, W = x_shape
        else:
            _, H, W, C = x_shape
        Pim = H * W
    else:
        if input_format_channels_first:
            N, C, Pim = x_shape
        else:
            N, Pim, C = x_shape

    pe_fmt_words = pos_emb_format.upper().split(' ')
    pe_dims = {}
    if 'H' in pe_fmt_words and 'W' in pe_fmt_words:
        pe_dims['H'] = pe_fmt_words.index('H')
        pe_dims['W'] = pe_fmt_words.index('W')
        pe_dims['P'] = [pe_dims['H'], pe_dims['W']]
    elif 'P' in pe_fmt_words:
        pe_dims['P'] = [pe_fmt_words.index('P')]
    else:
        raise ValueError('H and W or P dimension must be present in position embedding format')
    if 'D' in pe_fmt_words:
        pe_dims['D'] = [pe_fmt_words.index('D')]
    elif 'D1' in pe_fmt_words:
        pe_dims['D'] = []
        for i in range(10):
            if f'D{i+1}' in pe_fmt_words:
                idx = pe_fmt_words.index(f'D{i+1}')
                pe_dims[f'D{i+1}'] = idx
                pe_dims['D'].append(idx)
            else:
                break
    else:
        raise ValueError('D dimension must be present in position embedding format')
    if 'B' in pe_fmt_words:
        pe_sample_dim = True
        pe_dims['B'] = pe_fmt_words.index('B')
    else:
        pe_sample_dim = False

    # Check if there are any unused (and therefore unknown) words in the format
    # string
    for word in pe_fmt_words:
        if word not in pe_dims:
            raise ValueError(f"Unknown word '{word}' in position embedding format")

    # We try to implement a flexible interface that can handle different
    # formats of the position embedding tensor.
    pe_shape = pos_embeddings[0][0].data.shape
    assert len(pe_shape) == len(pe_fmt_words), f"Position embedding shape {pe_shape} does not match format {pos_emb_format}"

    # Get sizes of each dimension
    Np = pe_shape[pe_dims['B']] if pe_sample_dim else 1
    assert ('H' in pe_dims and 'W' in pe_dims) or ('P' in pe_dims), \
        f"Position embedding format {pos_emb_format} does not have spatial dimensions or patch dimension"
    if 'H' in pe_dims and 'W' in pe_dims:
        pe_spatial = True
        Hp = pe_shape[pe_dims['H']]
        Wp = pe_shape[pe_dims['W']]
        Pp = Hp * Wp
    else:
        pe_spatial = False
        Pp = pe_shape[pe_dims['P'][0]]
    Ds = [pe_shape[d] for d in pe_dims['D']]
    D = reduce(lambda x, y: x * y, Ds, 1)
    pos_emb_numel = torch.tensor([pe.numel() for pe, _ in pos_embeddings])

    # The interface implemented here assumes the model's PE parameter does
    # not have a sample dimension.
    assert Np == 1, f"This interface assumes PE parameter does not have a sample dimension, but got {Np} != 1"

    # Move to CUDA if available and requested
    if torch.cuda.is_available() and not on_cpu:
        cuda = True
        model.cuda()
    else:
        cuda = False
        model.cpu()

    def shap_model(x):
        """
        Wrapper around model for the model to be compatible with
        `shap.KernelExplainer`s internal usage of the model. Takes 2D NumPy
        tensor of input features, converts to 4D Torch tensor and
        "per-sample" PE for them to be inputtable to the Torch model, and
        runs model in "PE per-sample" mode, returning the result of the
        inference as a NumPy tensor representing class logits.
        """
        # (N, C*H*W + D*H*W) -> (N, C, H, W) and (N, D, [Hp, Wp] or [P])
        x = torch.tensor(x)
        Ni = x.shape[0]
        if cuda:
            x = x.cuda()
        if input_format_spatial:
            if input_format_channels_first:
                image = x[:, :C*Pim].reshape(Ni, C, H, W)
            else:
                image = x[:, :Pim*C].reshape(Ni, H, W, C)
        else:
            if input_format_channels_first:
                image = x[:, :C*Pim].reshape(Ni, C, Pim)
            else:
                image = x[:, :Pim*C].reshape(Ni, Pim, C)
        if torch.is_complex(image):
            image = image.float()

        old_pos_emb_values = []
        offset = C * Pim
        pos_embs = []
        hooks = []
        for pe_i, (pe, pe_module) in enumerate(pos_embeddings):
            pe_num_elem = pos_emb_numel[pe_i]

            pos_emb = x[:, offset:offset+pe_num_elem]

            if pe_spatial:
                pos_emb = pos_emb.reshape(x.shape[0], *Ds, Hp, Wp)
            else:
                pos_emb = pos_emb.reshape(x.shape[0], *Ds, Pp)

            offset += pe_num_elem

            # We retrieve the current value of the PE, so we can reset it later
            old_pos_emb_values.append(pe.data.clone())

            # B [Ds] [Ps] -> whatever the input was
            original = {'B': 0}
            j = 1
            if len(pe_dims['D']) > 1:
                for i in range(len(pe_dims['D'])):
                    original[f'D{i+1}'] = j
                    j += 1
            else:
                original['D'] = j
                j += 1
            if len(pe_dims['P']) > 1:
                names = ['H', 'W']
                for i in range(len(pe_dims['P'])):
                    original[names[i]] = j
                    j += 1
            else:
                original['P'] = j
                j += 1
            pos_emb_order = [original[w] for w in pe_fmt_words]
            if 'B' not in pe_fmt_words:
                pos_emb_order = [0] + pos_emb_order
            pos_emb = pos_emb.permute(*pos_emb_order)

            if not pe_not_broadcastable:
                pos_emb = pos_emb[0]

            if use_hooks:
                pos_embs.append(pos_emb.clone())
                # Using a hook is necessary for PE methods that are updated during
                # the forward pass, e.g. ones that need to know the number of
                # patches
                def hook_custom_pos_emb(module, input, output, hook):
                    # Remove itself after the first call
                    hook.remove()
                    # Call embeddings-only forward pass with modified pos_emb
                    if isinstance(output, tuple):
                        output = output[0]
                    output.data = module.forward_fixed_pos_emb(input[0], module.pos_emb_to_use)

                pe_module.pos_emb_to_use = pos_embs[pe_i]
                hook = lambda module, input, output: \
                    hook_custom_pos_emb(module, input, output, hook)
                hook = pe_module.register_forward_hook(hook)
                hooks.append(hook)
            else:
                pe.data = pos_emb

        result = model(image)
        result = model_out_unpack_fn(result) if model_out_unpack_fn is not None else result
        result = result.detach().cpu().numpy()

        for hook in hooks:
            hook.remove()

        for i, (pe, _) in enumerate(pos_embeddings):
            # Reset PE to original value
            pe.data = old_pos_emb_values[i]

        return result

    def explainer_input(x, pos_embeddings, background=False):
        """
        Takes 4D Torch tensor and original PE tensor (shape (1, D, Hp, Wp),
        not "per sample"), converts to 2D NumPy tensor representing all
        model input features, in a format compatible with
        `shap.KernelExplainer()`.

        If background=True, we shuffle the PE tokens along H*W dimensions
        (not D!), so that each D-dimensional PE token is intact but randomly
        displaced.

        If background=True, the data is grouped by image channels and PE (four
        feature groups in total).
        """
        # image = (N, C, H, W) -> (N, C*H*W)
        x = x.cpu()
        image = x.numpy().reshape(x.shape[0], -1)

        pos_embs = []
        for i, (pe, _) in enumerate(pos_embeddings):
            pos_emb = pe.clone()
            if cuda:
                pos_emb = pos_emb.cpu()

            # Allow for shape flexibility: any PE shape -> (N, D*[Hp*Wp or P])
            pe_standard_order = ['B', 'D', 'P']
            if pe_spatial:
                pe_standard_order = ['B', 'D', 'H', 'W']
            pe_no_batch_dim = False
            order = []
            offset = 0
            if 'B' not in pe_fmt_words:
                pos_emb = pos_emb.unsqueeze(0)
                order.append([0])
                pe_no_batch_dim = True
                offset += 1
                pe_standard_order = pe_standard_order[1:]
            for w in pe_standard_order:
                if not isinstance(pe_dims[w], list):
                    order.append([pe_dims[w] + offset])
                else:
                    order.append([p + offset for p in pe_dims[w]])
            # Flatten list
            pe_standard_order = [item for sublist in order for item in sublist]
            # Apply transformation
            pos_emb = pos_emb.permute(*pe_standard_order)
            # Expand batch dim
            pos_emb = pos_emb.expand(x.shape[0], *([-1] * (len(pe_standard_order) - 1)))
            # Flatten all but batch dim
            pos_emb = pos_emb.reshape(x.shape[0], -1).numpy()
            pos_embs.append(pos_emb)

            if debug_mode:
                # DEBUG: transform the flattened PE back to the original shape,
                # and check if the shapes and values are the same
                pos_emb_flat = torch.tensor(pos_emb.copy())
                if pe_spatial:
                    pos_emb_flat = pos_emb_flat.reshape(x.shape[0], *Ds, Hp, Wp)
                else:
                    pos_emb_flat = pos_emb_flat.reshape(x.shape[0], *Ds, Pp)
                # B [Ds] [Ps] -> whatever the input was
                original = {'B': 0}
                j = 1
                if len(pe_dims['D']) > 1:
                    for i in range(len(pe_dims['D'])):
                        original[f'D{i+1}'] = j
                        j += 1
                else:
                    original['D'] = j
                    j += 1
                if len(pe_dims['P']) > 1:
                    names = ['H', 'W']
                    for i in range(len(pe_dims['P'])):
                        original[names[i]] = j
                        j += 1
                else:
                    original['P'] = j
                    j += 1

                pos_emb_order = [original[w] for w in pe_fmt_words]
                if 'B' not in pe_fmt_words:
                    pos_emb_order = [0] + pos_emb_order
                pos_emb_flat = pos_emb_flat.permute(*pos_emb_order)

                if pe_no_batch_dim:
                    orig = pe.data.clone().unsqueeze(0).expand(x.shape[0], *([-1] * (len(pe.data.shape))))
                else:
                    orig = pe.data.clone().expand(x.shape[0], *([-1] * (len(pe.data.shape) - 1)))
                assert pos_emb_flat.shape == orig.shape
                assert torch.allclose(pos_emb_flat, orig)

            if background:
                if pe_not_broadcastable:
                    # The PE cannot be broadcast to the number of samples,
                    # likely because it is a relative position bias or some
                    # other internal component. In this case, all background
                    # samples in this batch will have the same shuffling applied
                    # to the PE.
                    pos_emb = pos_emb.reshape((x.shape[0], D, -1)).transpose((0, 2, 1))
                    for i in range(x.shape[0]):
                        pos_emb_i = pos_emb[i].copy()
                        np.random.shuffle(pos_emb_i)
                        pos_emb[i] = pos_emb_i
                    pos_emb = pos_emb.transpose((0, 2, 1)).reshape((x.shape[0], -1))
                else:
                    # Only shuffle first element, and paste it everywhere
                    pos_emb = pos_emb.reshape((x.shape[0], D, -1)).transpose((0, 2, 1))
                    pos_emb_i = pos_emb[0].copy()
                    np.random.shuffle(pos_emb_i)
                    for i in range(1, x.shape[0]):
                        pos_emb[i] = pos_emb_i.copy()
                    pos_emb = pos_emb.transpose((0, 2, 1)).reshape((x.shape[0], -1))

        data = np.concatenate((image, *pos_embs), axis=1)

        if background and not spatial_features:
            # Wrap in DenseData to apply grouping:
            # https://github.com/shap/shap/issues/1199#issuecomment-626876077
            # Only apply to background data, because shap_values() input will
            # inherit groups from it and is not compatible with DenseData
            n_vals_per_color_channel = Pim
            if image_channels_features:
                group_names = ['image_R', 'image_G', 'image_B', 'PE']
                groups = [
                    list(range(n_vals_per_color_channel * 0, n_vals_per_color_channel * 1)),
                    list(range(n_vals_per_color_channel * 1, n_vals_per_color_channel * 2)),
                    list(range(n_vals_per_color_channel * 2, n_vals_per_color_channel * 3)),
                    list(range(n_vals_per_color_channel * 3, data.shape[1]))]
            else:
                group_names = ['image', 'PE']
                groups = [
                    list(range(n_vals_per_color_channel * 0, n_vals_per_color_channel * 3)),
                    list(range(n_vals_per_color_channel * 3, data.shape[1]))]
            data = DenseData(data, group_names, groups)

        return data

    def shap_output(shap_values):
        """
        Reshape NumPy array outputs of `shap.KernelExplainer:shap_values()`
        to [if spatial_outputs == True: (N, Cin, H, W, Cout), else (N, Cin)] for
        image and [if spatial_outputs == True: (N, D, H, W, Cout), else (N, 1)]
        for PE, for visualization.

        TODO: PE shap values should be identical (up to estimation error)
        for all values; build in a check for this.
        """
        if spatial_features:
            N, PCD, Cout = shap_values.shape
            if pe_spatial:
                assert PCD == C * Pim + D * Hp * Wp, \
                    f'Error in SHAP values shape: {PCD} != C={C} * H={H} * W={W} + D={D} * Hp={Hp} * Wp={Wp}'
            else:
                assert PCD == C * Pim + D * Pp, \
                    f'Error in SHAP values shape: {PCD} != C={C} * H={H} * W={W} + D={D} * P={Pp}'
            cutoff = C * Pim
        else:
            if image_channels_features:
                cutoff = C
            else:
                cutoff = 1

        image = shap_values[:,:cutoff]
        pos_emb = shap_values[:,cutoff:]

        if spatial_features:
            image = image.reshape(N, C, H, W, Cout)
            if pe_spatial:
                pos_emb = pos_emb.reshape(N, *Ds, Hp, Wp, Cout)
            else:
                pos_emb = pos_emb.reshape(N, *Ds, Pp, Cout)

        return image, pos_emb

    explainer_kwargs = {"silent": True}
    if batch_size is not None:
        explainer_kwargs["batch_size"] = batch_size

    return shap_lib.KernelExplainer, shap_model, explainer_input, explainer_kwargs, shap_output, pos_embeddings