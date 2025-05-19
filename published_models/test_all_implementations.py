import pytest
import os

import published_models
from published_models.factory import IMPLEMENTATIONS
from analysis.sources import get_attribute_from_name_or_callable


class AttrDict(dict):
    """
    https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, attr):
        if attr not in self:
            raise AttributeError(f"Args has no attribute {attr}")
        return attr


class TestAllImplementations:
    def test_implementations_count(self):
        # Get all directories in published_models/model_implementations
        implementations = os.listdir(os.path.join(os.path.dirname(__file__), 'model_implementations'))
        # Remove example directory
        implementations.remove('example')
        # Check count matches entries in IMPLEMENTATIONS
        assert len(implementations) == len(IMPLEMENTATIONS), "IMPLEMENTATIONS count does not match published_models/model_implementations count"

    @pytest.mark.parametrize('model_name', IMPLEMENTATIONS.keys())
    def test_get_attribution_sources(self, model_name):
        assert model_name in IMPLEMENTATIONS
        try:
            configurations = IMPLEMENTATIONS[model_name].CONFIGURATIONS
        except AttributeError:
            assert False, f"Model {model_name} does not implement CONFIGURATIONS, which are necessary for testing"
        for config in configurations:
            args = AttrDict(config)
            try:
                model = published_models.get(model_name, args=args)
            except Exception as e:
                assert False, f"Model {model_name} is misconfigured for test case, or is broken, with args {args}.\nType of error: {type(e)}.\nError raised:\n{e}"
            sources = model.get_attribution_sources()
            assert sources is not None
            assert isinstance(sources, dict)
            assert 'image' in sources
            assert 'bias' in sources

    # TODO: test if attribution works, on fake data