import pytest
import torch
from analysis.inference import inference_to_gradbased_analysis

class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = torch.nn.Conv2d(1, 2, kernel_size=1, padding=0)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        self.input = x
        x = self.linear(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

def test_inference_to_gradbased_analysis():
    """
    Tests analysis/inference.py::inference_to_gradbased_analysis(), one of the
    core functions of the analysis module, by using a dummy model with very
    simple hardcoded weights and comparing the activations and gradients
    computed by the function to the known activations and gradients derived by
    hand.
    """
    model = DummyModel()
    model.linear.weight = torch.nn.Parameter(torch.tensor([[[[1.]]], [[[0.]]]]))
    model.linear.bias = torch.nn.Parameter(torch.tensor([1., 0.]))

    # x.shape = (B, H, W, C)
    test_x = torch.tensor([[[[1.]]]])
    # y.shape = (B)
    test_y = torch.tensor([0])
    # Image can be a named attributed of the model, but bias needs to be a list,
    # and we need to use a callable to retrieve a nested attribute
    sources = {'image': 'input', 'bias': lambda m: [m.linear.bias]}

    activations, gradients = \
        inference_to_gradbased_analysis(model, test_x, test_y, sources,
                                        lrp=False,
                                        patch_size_x=1, patch_size_y=1)

    # Test that the activations and gradients are as expected

    # Use torch.testing.assert_close() instead of Python equals, because the
    # underlying tensors are not always the same object and this is what PyTorch
    # compares, instead of the values of the tensors
    torch.testing.assert_close(activations['pred_class']['image'][(0,0)][0],
                               test_x[0])
    torch.testing.assert_close(activations['pred_class']['bias'][(0,0)][0],
                               model.linear.bias)
    torch.testing.assert_close(gradients['pred_class']['image'][(0,0)][0],
                               torch.tensor([[[1.]]]))
    torch.testing.assert_close(gradients['pred_class']['bias'][(0,0)][0],
                               torch.tensor([1., 0.]))

    # Test that sum(activations * gradients) = sum(model output)
    # This is a property of the Full-Gradient method (Srinivas 2019, Eq. 5)
    out = model(test_x)[0].sum()
    ig_image = (activations['pred_class']['image'][(0,0)][0] * gradients['pred_class']['image'][(0,0)][0]).sum()
    ig_bias = (activations['pred_class']['bias'][(0,0)][0] * gradients['pred_class']['bias'][(0,0)][0]).sum()
    torch.testing.assert_close(ig_image + ig_bias, out)

if __name__ == "__main__":
    pytest.main()