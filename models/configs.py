# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections


def get_testing():
    """Returns a minimal configuration for testing.

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config


def get_b16_config():
    """Returns the ViT-B/16 configuration.

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    del config.patches.size
    config.patches.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration.

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration.

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config


def get_red_green_config():
    """Returns the vit configuration for the red/green dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config


def get_pos_config():
    """Returns the vit configuration for the pos dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (1, 1)})
    config.hidden_size = 32
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 256
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config


def get_cifar_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 6
    config.transformer.num_layers = 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_cifar_gani_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_cifar_ganiv2_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_cifar_ganiv2_dropout_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_cifar_ganiv2_cls_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_cifar_ganiv2_dropout_cls_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_cifar_ganiv2_224px_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_cifar_v1_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 8 # 6
    config.transformer.num_layers = 6 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.1 # 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_cifar_v2_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)}) # {'size': (16, 16)}
    config.hidden_size = 512 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 512 # 3072
    config.transformer.num_heads = 8 # 6
    config.transformer.num_layers = 6 # 6
    config.transformer.attention_dropout_rate = 0.1 # 0.1
    config.transformer.dropout_rate = 0.1 # 0.1
    # config.classifier = 'avg'
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_cifar_v3_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)}) # {'size': (16, 16)}
    config.hidden_size = 512 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 512 # 3072
    config.transformer.num_heads = 8 # 6
    config.transformer.num_layers = 6 # 6
    config.transformer.attention_dropout_rate = 0.1 # 0.1
    config.transformer.dropout_rate = 0.1 # 0.1
    # config.classifier = 'avg'
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_cifar_ganiv2_thin_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)}) # {'size': (16, 16)}
    config.hidden_size = 32 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 48 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_cifar_ganiv2_p2_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_cifar_ganiv2_p8_config():
    """Returns the vit configuration for the cifar dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_imagenet_tiny_config():
    """Returns the vit configuration for the imagenet dataset

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 2048
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_eurosat_config():
    """Returns the vit configuration for the EuroSAT dataset, adapted from CIFAR
    gani v2.

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_eurosat_p4_config():
    """Returns the vit configuration for the EuroSAT dataset, adapted from CIFAR
    gani v2.

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_eurosat_dropout_config():
    """Returns the vit configuration for the EuroSAT dataset, adapted from CIFAR
    gani v2.

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_eurosat_p4_dropout_config():
    """Returns the vit configuration for the EuroSAT dataset, adapted from CIFAR
    gani v2.

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_eurosat_p4_dropout_cls_config():
    """Returns the vit configuration for the EuroSAT dataset, adapted from CIFAR
    gani v2.

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_nih_config():
    """Returns the vit configuration for the NIH dataset, adapted from CIFAR
    gani v2.

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_nih_dropout_config():
    """Returns the vit configuration for the NIH dataset, adapted from CIFAR
    gani v2.

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_nih_dropout_cls_config():
    """Returns the vit configuration for the NIH dataset, adapted from CIFAR
    gani v2.

    Originally, this config used token classification, but this was changed to
    avg pooling without changing the workings of the config. See the original
    config below.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config




def get_b16_original_config():
    """Returns the ViT-B/16 configuration.

    This is the original config inherited from previous code, using token
    classification. See above for the legacy config as it was before
    config.classification was ignored."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_l16_original_config():
    """Returns the ViT-L/16 configuration.

    This is the original config inherited from previous code, using token
    classification. See above for the legacy config as it was before
    config.classification was ignored."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config




def get_h14_original_config():
    """Returns the ViT-L/16 configuration.

    This is the original config inherited from previous code, using token
    classification. See above for the legacy config as it was before
    config.classification was ignored."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_red_green_original_config():
    """Returns the vit configuration for the red/green dataset

    This is the original config inherited from previous code, using token
    classification. See above for the legacy config as it was before
    config.classification was ignored."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_pos_original_config():
    """Returns the vit configuration for the pos dataset

    This is the original config inherited from previous code, using token
    classification. See above for the legacy config as it was before
    config.classification was ignored."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (1, 1)})
    config.hidden_size = 32
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 256
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_cifar_original_config():
    """Returns the vit configuration for the cifar dataset

    This is the original config inherited from previous code, using token
    classification. See above for the legacy config as it was before
    config.classification was ignored."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 6
    config.transformer.num_layers = 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_cifar_gani_original_config():
    """Returns the vit configuration for the cifar dataset

    This is the original config inherited from previous code, using token
    classification. See above for the legacy config as it was before
    config.classification was ignored."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_cifar_ganiv2_original_config():
    """Returns the vit configuration for the cifar dataset

    This is the original config inherited from previous code, using token
    classification. See above for the legacy config as it was before
    config.classification was ignored."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_cifar_ganiv2_original_p2_config():
    """Returns the vit configuration for the cifar dataset

    This is the original config inherited from previous code, using token
    classification. See above for the legacy config as it was before
    config.classification was ignored."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_cifar_ganiv2_original_p8_config():
    """Returns the vit configuration for the cifar dataset

    This is the original config inherited from previous code, using token
    classification. See above for the legacy config as it was before
    config.classification was ignored."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)}) # {'size': (16, 16)}
    config.hidden_size = 192 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 384 # 3072
    config.transformer.num_heads = 12 # 6
    config.transformer.num_layers = 9 # 6
    config.transformer.attention_dropout_rate = 0.0 # 0.1
    config.transformer.dropout_rate = 0.0 # 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_imagenet_tiny_original_config():
    """Returns the vit configuration for the imagenet dataset

    This is the original config inherited from previous code, using token
    classification. See above for the legacy config as it was before
    config.classification was ignored."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 2048
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_toy_config():
    """Returns the ViT-B/16 configuration, adapted to the toy setting."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'avg'
    config.representation_size = None
    return config

def get_toy_cls_config():
    """Returns the ViT-B/16 configuration, adapted to the toy setting."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config