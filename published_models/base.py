"""
Base class for published models. All published models should inherit from this
class.
"""
from torch import nn


class BasePublishedModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Make sure the initialized and weights-loaded model is assigned to self.model
        self.model = None
        self.weights_metadata = None

    def args_from_weights_metadata(self, args):
        """
        TODO: document

        pretrained
        img_size
        reported_val_acc
        """
        if self.weights_metadata is not None:
            args.model_pretrained = self.weights_metadata['model_pretrained']
            if 'img_size' in self.weights_metadata:
                args.internal_img_size = self.weights_metadata['img_size']
            if 'reported_val_acc' in self.weights_metadata and not args.training:
                args.reported_val_acc = self.weights_metadata['reported_val_acc']

    def get_attribution_sources(self):
        """
        Returns model attributes pertaining to bias/information sources for
        gradient-based attribution analysis.

        Returns:
            dict: A dictionary mapping source names `image`, `pos_emb` and
                `bias` to model attributes.
        """
        # TODO: Implement this in any subclasses
        pass

    def get_logits(self, outputs):
        """
        Given all model outputs, returns the logits. Used as a wrapper function
        for abstracting away model details in gradient-based analysis.
        """
        # TODO: Implement this in any subclasses
        pass

    def forward_logits(self, x):
        """
        Forward pass of the model, returning logits
        """
        return self.get_logits(self.model(x))