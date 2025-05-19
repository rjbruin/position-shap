"""
EXAMPLE NAME
"""
from published_models import BasePublishedModel


class ExamplePublishedModel(BasePublishedModel):
    # NOTE: provide a list of all possible configurations (args dictionary passed to
    # factory.get()), so that they can be tested automatically
    # See google_research_vit for example usage.
    CONFIGURATIONS = [
        # {}, # TODO
    ]

    def __init__(self):
        super().__init__(self)
        # TODO: implement net
        self.net = None

    def patch_size(self):
        """
        Returns:
            tuple(int): The patch size of the model, in (y, x).
        """
        # TODO: implement
        pass

    def get_attribution_sources(self):
        """
        Returns model attributes pertaining to bias/information sources for
        gradient-based attribution analysis.

        Returns:
            dict: A dictionary mapping source names `image`, `pos_emb` and
                `bias` to model attributes.
        """
        # TODO: Implement this
        return {
            'image': None, # TODO
            'pos_emb': None, # TODO
            'bias': None, # TODO
        }

    def get_logits(self, outputs):
        """
        Given all model outputs, returns the logits. Used as a wrapper function
        for abstracting away model details in gradient-based analysis.
        """
        # TODO: Implement this
        pass

    @staticmethod
    def dataset_policy(experiment, args):
        """
        Returns a dataset policy for the model. If this method is implemented,
        these dataset definitions will be used by train.py. Otherwise, one needs
        to specify a --dataset_policy that is implemented in
        datasets/datasets.py.

        NOTE: a `@staticmethod` is just a method implemented for the class
        rather than the class instance. The only difference with a regular class
        method is that there is no `self` attribute. This is because nothing can
        (or should) be saved in `self`, but rather this is code just executed
        from `train.py`. You *can* assign things to `experiment`, which is the
        instance of the `Experiment` class in `train.py`.
        """
        # TODO: either 1) use one of the centrally implemented dataset policies
        # in datasets/dataset.py, 2) use a dataset policy from an existing
        # published model, or 3) implement a dataset policy yourself here, in
        # case your model requires a specific way to set up the data

        # NOTE: example code for (1)
        # from datasets.datasets import policy_i_want_to_reuse
        # return policy_i_want_to_reuse(experiment, args)

        # NOTE: example code for (2)
        # implementation
        # import published_models.model_implementations.another_published_model.another_published_model as AnotherPublishedModel
        # return AnotherPublishedModel.dataset_policy(experiment, args)