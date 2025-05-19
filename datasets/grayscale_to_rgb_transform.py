import numpy as np


class GrayscaleToRGBNumpy(object):

    def __call__(self, pic):
        """
        Args:
            pic (NumPy array): Image to be converted.

        Returns:
            NumPy array: Image converted.

        """
        # Return pic (np array) with new first dimension, repeated three times
        return pic.repeat(3, axis=0).transpose((1, 2, 0))


class GrayscaleToRGBTensor(object):

    def __call__(self, pic):
        """
        Args:
            pic (Tensor): Image to be converted.

        Returns:
            Tensor: Image converted.

        """
        # Return pic (tensor) with first dimension repeated three times
        return pic.repeat(3, 1, 1)